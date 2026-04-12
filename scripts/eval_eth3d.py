#!/usr/bin/env python3
"""
scripts/eval_eth3d.py
=====================
ETH3D disparity accuracy evaluation for SGM-ViT.

Evaluates three predictors against ETH3D sparse ground-truth disparity on
the 27 training scenes:

  1. SGM (pre-computed)    — raw stereo baseline from sgm_hole/<scene>/<scene>.pfm
  2. Dense DA2 + align     — DepthAnythingV2 monocular depth, per-image
                             least-squares aligned to SGM disparity space
  3. Sparse DA2 + align    — SGM-ViT: token-pruned + re-assembled depth,
                             aligned to SGM disparity space

Alignment: argmin_{s,t} Σ_i || s·d_i + t − disp_sgm_i ||²
at pixels where alignment confidence >= conf_threshold.
GT disparity is NOT used for alignment.

Metrics:
  EPE   — mean absolute error at GT-valid pixels
  D1    — % pixels where |pred − gt| > max(1.0, 0.05·gt)
           (ETH3D criterion; KITTI uses 3.0 instead of 1.0)
  bad1  — % pixels where |pred − gt| > 1.0   (ETH3D standard)
  bad2  — % pixels where |pred − gt| > 2.0
  delta_1 — fraction of GT-valid pixels where max(pred/gt, gt/pred) < 1.25

Dataset:
  root/
    <scene>/im0.png, im1.png, disp0GT.pfm
    training/sgm_hole/<scene>/
      <scene>.pfm
      <scene>_mismatches.npy
      <scene>_occlusion.npy

Usage
-----
  python scripts/eval_eth3d.py
  python scripts/eval_eth3d.py --max-samples 5
  python scripts/eval_eth3d.py --eth3d-root /your/path/to/eth3d
  python scripts/eval_eth3d.py --no-sparse
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import core._paths  # noqa: F401
from core.fusion import fuse_sgm_da2
from core.pipeline import (
    DA2_MODEL_CONFIGS,
    EMBED_DIM_MAP,
    TOKEN_GRID_SIZE,
    align_depth_to_sgm,
    load_da2_model,
    run_masked_sparse_da2,
)
from core.sgm_wrapper import run_sgm_with_confidence
from core.token_router import SGMConfidenceTokenRouter
from scripts.common_config import (
    DEFAULT_ALIGN_CONF_THRESHOLD,
    DEFAULT_CONF_SIGMA,
    DEFAULT_DISPARITY_RANGE,
    DEFAULT_ENCODER,
    DEFAULT_ETH3D_ROOT,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_PKRN_MIN_DIST,
    DEFAULT_PRUNE_LAYER,
    DEFAULT_PRUNE_THRESHOLD,
    DEFAULT_WEIGHTS,
    default_results_dir,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_pfm(path: str) -> np.ndarray:
    """Read a .pfm disparity file → (H, W) float32."""
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        assert header in (b"PF", b"Pf"), f"Not a PFM file: {path}"
        w, h = map(int, f.readline().split())
        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f").reshape(h, w)
    return np.ascontiguousarray(np.flipud(data)).astype(np.float32)


# ---------------------------------------------------------------------------
# Sample list builder
# ---------------------------------------------------------------------------

def build_sample_list_eth3d(eth3d_root: str) -> list[dict]:
    """Build sorted list of sample dicts for the ETH3D training set.

    Uses scene-name matching to avoid misalignment with the stray
    training/sgm_hole/im0.pfm file.

    Each dict:
        left      : left image path
        right     : right image path
        gt_pfm    : sparse GT disparity (.pfm)
        sgm_pfm   : pre-computed SGM disparity (.pfm)
        mismatch  : mismatch mask (.npy)
        occlusion : occlusion mask (.npy)
        pkrn_cache: where to cache PKRN confidence
        scene     : scene name string
    """
    sgm_base = os.path.join(eth3d_root, "training", "sgm_hole")
    image_paths = sorted(glob(os.path.join(eth3d_root, "*/im0.png")))

    samples: list[dict] = []
    skipped = []

    for img0 in image_paths:
        scene = Path(img0).parent.name
        if scene == "training":
            continue  # skip the training/ directory itself

        img1    = os.path.join(Path(img0).parent, "im1.png")
        gt_pfm  = os.path.join(Path(img0).parent, "disp0GT.pfm")
        sgm_dir = os.path.join(sgm_base, scene)
        sgm_pfm = os.path.join(sgm_dir, f"{scene}.pfm")
        mis_npy = os.path.join(sgm_dir, f"{scene}_mismatches.npy")
        occ_npy = os.path.join(sgm_dir, f"{scene}_occlusion.npy")

        missing = [p for p in (img1, gt_pfm, sgm_pfm, mis_npy, occ_npy)
                   if not os.path.isfile(p)]
        if missing:
            skipped.append(scene)
            continue

        pkrn_cache = sgm_pfm.replace(".pfm", "_pkrn.npy")
        samples.append({
            "left":      str(img0),
            "right":     str(img1),
            "gt_pfm":    gt_pfm,
            "sgm_pfm":   sgm_pfm,
            "mismatch":  mis_npy,
            "occlusion": occ_npy,
            "pkrn_cache":pkrn_cache,
            "scene":     scene,
        })

    if skipped:
        logging.warning(
            f"ETH3D: {len(skipped)} scenes skipped (missing files): {skipped}"
        )
    logging.info(f"ETH3D: {len(samples)} scenes loaded")
    return samples


# ---------------------------------------------------------------------------
# Confidence helpers (mirrors eval_kitti.py pattern)
# ---------------------------------------------------------------------------

def load_pkrn_confidence(
    s: dict,
    disparity_range: int,
    pkrn_min_dist: int,
    smooth_sigma: float,
) -> np.ndarray:
    cache_path = s["pkrn_cache"]
    if os.path.exists(cache_path):
        pkrn_raw = np.load(cache_path)
    else:
        _, pkrn_raw, _ = run_sgm_with_confidence(
            left_path       = s["left"],
            right_path      = s["right"],
            disparity_range = disparity_range,
            smooth_sigma    = 0.0,
            pkrn_min_dist   = pkrn_min_dist,
            verbose         = False,
        )
        np.save(cache_path, pkrn_raw)
        logging.info(f"Cached PKRN: {cache_path}")
    if smooth_sigma > 0:
        return gaussian_filter(pkrn_raw, sigma=smooth_sigma).clip(0.0, 1.0)
    return pkrn_raw


def load_confidence_maps(
    s: dict,
    disparity_range: int,
    pkrn_min_dist: int,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    mismatch  = np.load(s["mismatch"]).astype(bool)
    occlusion = np.load(s["occlusion"]).astype(bool)
    conf_align = (~(mismatch | occlusion)).astype(np.float32)
    conf_align = gaussian_filter(conf_align, sigma=smooth_sigma)
    conf_route = load_pkrn_confidence(s, disparity_range, pkrn_min_dist, smooth_sigma)
    return conf_align, conf_route


# ---------------------------------------------------------------------------
# Metrics (ETH3D-specific thresholds)
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_gt: np.ndarray,
) -> dict:
    """ETH3D metrics at GT-valid pixels.

    bad1    : |pred − gt| > 1.0
    bad2    : |pred − gt| > 2.0
    D1      : |pred − gt| > max(1.0, 0.05·gt)   (ETH3D convention)
    EPE     : mean absolute error
    delta_1 : max(pred/gt, gt/pred) < 1.25
    """
    mask = valid_gt & np.isfinite(pred) & (pred >= 0)
    n = int(mask.sum())
    if n == 0:
        return {"epe": float("nan"), "d1": float("nan"),
                "bad1": float("nan"), "bad2": float("nan"),
                "delta1": float("nan"), "n": 0}
    err = np.abs(pred[mask] - gt[mask])
    epe  = float(err.mean())
    bad1 = float((err > 1.0).mean() * 100.0)
    bad2 = float((err > 2.0).mean() * 100.0)
    d1   = float((err > np.maximum(1.0, 0.05 * gt[mask])).mean() * 100.0)

    # delta_1: avoid division by zero for gt == 0 (already filtered by valid_gt)
    gt_m   = gt[mask]
    pred_m = pred[mask].clip(1e-6)
    ratio  = np.maximum(pred_m / gt_m, gt_m / pred_m)
    delta1 = float((ratio < 1.25).mean())

    return {"epe": epe, "d1": d1, "bad1": bad1, "bad2": bad2,
            "delta1": delta1, "n": n}


def aggregate(records: list[dict]) -> dict:
    keys = ["epe", "d1", "bad1", "bad2", "delta1"]
    out = {}
    for k in keys:
        vals = [r[k] for r in records if not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n"] = len([r for r in records if not np.isnan(r["epe"])])
    out["pixel_n"] = sum(r["n"] for r in records if not np.isnan(r["epe"]))
    return out


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, "eval_eth3d.log")),
            logging.StreamHandler(),
        ],
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logging.info("=" * 66)
    logging.info("SGM-ViT ETH3D Disparity Evaluation")
    logging.info("=" * 66)
    logging.info(f"  ETH3D root    : {args.eth3d_root}")
    logging.info(f"  Encoder       : {args.encoder}")
    logging.info(f"  Prune layer   : {args.prune_layer}")
    logging.info(f"  Threshold θ   : {args.threshold}")
    logging.info(f"  Device        : {device}")
    logging.info(f"  Output        : {args.out_dir}")
    logging.info("=" * 66)

    model = load_da2_model(args.encoder, args.weights, device)
    embed_dim = EMBED_DIM_MAP[args.encoder]
    N = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE
    router = SGMConfidenceTokenRouter(
        token_grid_size=TOKEN_GRID_SIZE,
        confidence_threshold=args.threshold,
        learnable_threshold=False,
    )

    samples = build_sample_list_eth3d(args.eth3d_root)
    if not samples:
        logging.error("No ETH3D samples found — check --eth3d-root.")
        return
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    logging.info(f"Total scenes: {len(samples)}\n")

    methods = ("sgm", "dense", "sparse", "fused", "fused_sparse")
    records    = {m: [] for m in methods}
    records_sv = {m: [] for m in methods}
    csv_rows: list[dict] = []
    t_dense_total = t_sparse_total = 0.0

    for idx, s in enumerate(tqdm(samples, desc="Evaluating ETH3D", unit="scene")):
        left_bgr = cv2.imread(s["left"])
        if left_bgr is None:
            logging.warning(f"Cannot read: {s['left']}"); continue

        gt_disp  = read_pfm(s["gt_pfm"])
        valid_gt = (gt_disp > 0) & np.isfinite(gt_disp)
        sgm_disp = read_pfm(s["sgm_pfm"])

        conf_align, conf_route = load_confidence_maps(
            s, args.disparity_range, args.pkrn_min_dist, args.conf_sigma
        )

        # SGM-valid mask
        sgm_for_mask = sgm_disp
        if sgm_disp.shape != gt_disp.shape:
            sgm_for_mask = cv2.resize(sgm_disp, (gt_disp.shape[1], gt_disp.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        valid_sgm = valid_gt & (sgm_for_mask > 0)

        # SGM metrics
        sgm_m    = compute_metrics(sgm_disp,     gt_disp, valid_gt)
        sgm_m_sv = compute_metrics(sgm_for_mask, gt_disp, valid_sgm)
        records["sgm"].append(sgm_m)
        records_sv["sgm"].append(sgm_m_sv)

        # Dense DA2
        t0 = time.perf_counter()
        depth_dense = model.infer_image(left_bgr, input_size=518)
        t_dense_total += time.perf_counter() - t0

        disp_dense, scale_d, shift_d = align_depth_to_sgm(
            depth_mono=depth_dense,
            disparity_raw=sgm_disp,
            confidence_map=conf_align,
            conf_threshold=args.conf_threshold,
        )
        if disp_dense.shape != gt_disp.shape:
            disp_dense = cv2.resize(disp_dense, (gt_disp.shape[1], gt_disp.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
        dense_m    = compute_metrics(disp_dense, gt_disp, valid_gt)
        dense_m_sv = compute_metrics(disp_dense, gt_disp, valid_sgm)
        records["dense"].append(dense_m)
        records_sv["dense"].append(dense_m_sv)

        # Fused SGM + Dense
        conf_fuse = conf_route
        if conf_fuse.shape != gt_disp.shape:
            conf_fuse = cv2.resize(conf_fuse, (gt_disp.shape[1], gt_disp.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        disp_fused = fuse_sgm_da2(
            sgm_disp=sgm_for_mask, da2_aligned=disp_dense,
            confidence_map=conf_fuse, conf_threshold=args.conf_threshold,
        )
        fused_m    = compute_metrics(disp_fused, gt_disp, valid_gt)
        fused_m_sv = compute_metrics(disp_fused, gt_disp, valid_sgm)
        records["fused"].append(fused_m)
        records_sv["fused"].append(fused_m_sv)

        # Sparse DA2
        if not args.no_sparse:
            conf_tensor  = torch.from_numpy(conf_route).unsqueeze(0).unsqueeze(0)
            dummy_tokens = torch.zeros(1, N, embed_dim)
            with torch.no_grad():
                routing = router(conf_tensor, dummy_tokens)
            prune_mask_1d = torch.zeros(N, dtype=torch.bool)
            prune_mask_1d[routing["prune_idx"][0]] = True

            t0 = time.perf_counter()
            depth_sparse = run_masked_sparse_da2(
                model=model, image_bgr=left_bgr,
                prune_mask=prune_mask_1d, input_size=518,
                prune_layer=args.prune_layer,
                do_reassembly=not args.no_reassembly,
            )
            t_sparse_total += time.perf_counter() - t0

            disp_sparse, _, _ = align_depth_to_sgm(
                depth_mono=depth_sparse, disparity_raw=sgm_disp,
                confidence_map=conf_align, conf_threshold=args.conf_threshold,
            )
            if disp_sparse.shape != gt_disp.shape:
                disp_sparse = cv2.resize(disp_sparse,
                                         (gt_disp.shape[1], gt_disp.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
            sparse_m    = compute_metrics(disp_sparse, gt_disp, valid_gt)
            sparse_m_sv = compute_metrics(disp_sparse, gt_disp, valid_sgm)
            prune_ratio = float(routing["prune_ratio"])

            conf_for_sp = conf_route
            if conf_for_sp.shape != gt_disp.shape:
                conf_for_sp = cv2.resize(conf_for_sp,
                                         (gt_disp.shape[1], gt_disp.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
            disp_fused_sp = fuse_sgm_da2(
                sgm_disp=sgm_for_mask, da2_aligned=disp_sparse,
                confidence_map=conf_for_sp, conf_threshold=args.conf_threshold,
            )
            fused_sp_m    = compute_metrics(disp_fused_sp, gt_disp, valid_gt)
            fused_sp_m_sv = compute_metrics(disp_fused_sp, gt_disp, valid_sgm)
        else:
            nan_m = {"epe": float("nan"), "d1": float("nan"), "bad1": float("nan"),
                     "bad2": float("nan"), "delta1": float("nan"), "n": 0}
            sparse_m = sparse_m_sv = nan_m
            fused_sp_m = fused_sp_m_sv = nan_m
            prune_ratio = float("nan")

        records["sparse"].append(sparse_m)
        records_sv["sparse"].append(sparse_m_sv)
        records["fused_sparse"].append(fused_sp_m)
        records_sv["fused_sparse"].append(fused_sp_m_sv)

        csv_rows.append({
            "idx": idx, "scene": s["scene"],
            "sgm_epe":        f"{sgm_m['epe']:.4f}",
            "sgm_bad1":       f"{sgm_m['bad1']:.4f}",
            "sgm_d1":         f"{sgm_m['d1']:.4f}",
            "dense_epe":      f"{dense_m['epe']:.4f}",
            "dense_bad1":     f"{dense_m['bad1']:.4f}",
            "dense_delta1":   f"{dense_m['delta1']:.4f}",
            "fused_epe":      f"{fused_m['epe']:.4f}",
            "fused_bad1":     f"{fused_m['bad1']:.4f}",
            "sparse_epe":     f"{sparse_m['epe']:.4f}",
            "sparse_bad1":    f"{sparse_m['bad1']:.4f}",
            "fused_sp_epe":   f"{fused_sp_m['epe']:.4f}",
            "fused_sp_bad1":  f"{fused_sp_m['bad1']:.4f}",
            "prune_pct":      f"{prune_ratio*100:.1f}" if not np.isnan(prune_ratio) else "n/a",
            "scale_d": f"{scale_d:.4f}", "shift_d": f"{shift_d:.4f}",
        })

        if (idx + 1) % max(1, len(samples) // 5) == 0 or idx < 3:
            logging.info(
                f"[{idx+1:2d}/{len(samples)}] {s['scene']:<30s}  "
                f"SGM EPE={sgm_m['epe']:.3f}  Dense EPE={dense_m['epe']:.3f}  "
                f"Fused EPE={fused_m['epe']:.3f}"
            )

    # ---- Aggregate ----
    def _row(name: str, rec: dict) -> str:
        a = aggregate(rec)
        return (f"  {name:<28s}  EPE={a['epe']:7.4f}  "
                f"bad1={a['bad1']:6.2f}%  bad2={a['bad2']:6.2f}%  "
                f"D1={a['d1']:6.2f}%  δ₁={a['delta1']:.4f}  "
                f"({a['n']} scenes)")

    header = f"\n{'='*70}\n  SGM-ViT ETH3D Evaluation Results\n{'='*70}\n"
    lines = [header,
             _row("SGM (pre-computed)",    records["sgm"]),
             _row("Dense DA2 + align",     records["dense"]),
             _row("Fused SGM + Dense DA2", records["fused"])]
    if not args.no_sparse:
        lines += [_row("Sparse DA2 (GAS)",       records["sparse"]),
                  _row("Fused SGM + Sparse DA2",  records["fused_sparse"])]
    lines += [
        f"\n  SGM-valid subset (SGM coverage region):",
        _row("SGM (pre-computed)",    records_sv["sgm"]),
        _row("Dense DA2 + align",     records_sv["dense"]),
        _row("Fused SGM + Dense DA2", records_sv["fused"]),
    ]
    if not args.no_sparse:
        lines += [_row("Fused SGM + Sparse DA2", records_sv["fused_sparse"])]

    timing = (
        f"\n  Timing:\n"
        f"    Dense  total={t_dense_total:.1f}s  "
        f"avg={t_dense_total/max(1,len(samples))*1e3:.1f}ms/scene\n"
    )
    if not args.no_sparse:
        timing += (f"    Sparse total={t_sparse_total:.1f}s  "
                   f"avg={t_sparse_total/max(1,len(samples))*1e3:.1f}ms/scene\n")
    lines.append(timing)

    result_text = "\n".join(lines)
    logging.info(result_text)

    txt_path = os.path.join(args.out_dir, "eval_results.txt")
    with open(txt_path, "w") as f:
        f.write(result_text)
    logging.info(f"Results: {txt_path}")

    if csv_rows:
        csv_path = os.path.join(args.out_dir, "eval_per_scene.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logging.info(f"Per-scene CSV: {csv_path}")

    logging.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SGM-ViT ETH3D evaluation.")
    p.add_argument("--eth3d-root",    default=DEFAULT_ETH3D_ROOT)
    p.add_argument("--weights",       default=DEFAULT_WEIGHTS)
    p.add_argument("--encoder",       default=DEFAULT_ENCODER,
                   choices=list(DA2_MODEL_CONFIGS.keys()))
    p.add_argument("--threshold",     type=float, default=DEFAULT_PRUNE_THRESHOLD)
    p.add_argument("--prune-layer",   type=int,   default=DEFAULT_PRUNE_LAYER)
    p.add_argument("--no-reassembly", action="store_true")
    p.add_argument("--no-sparse",     action="store_true")
    p.add_argument("--conf-threshold",type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD)
    p.add_argument("--conf-sigma",    type=float, default=DEFAULT_CONF_SIGMA)
    p.add_argument("--disparity-range", type=int, default=DEFAULT_DISPARITY_RANGE)
    p.add_argument("--pkrn-min-dist", type=int,   default=DEFAULT_PKRN_MIN_DIST)
    p.add_argument("--max-samples",   type=int,   default=DEFAULT_MAX_SAMPLES)
    p.add_argument("--out-dir",       default=default_results_dir("eval_eth3d"))
    p.add_argument("--cpu",           action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
