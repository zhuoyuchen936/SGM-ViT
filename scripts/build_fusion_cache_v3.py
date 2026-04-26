#!/usr/bin/env python3
"""Phase 7 Cache v3 builder (revised to match v1 call signatures).

Reads pre-computed tuned SGM from /tmp/sgm_tuned_all/<dataset>/<name>.npz and
produces cache at artifacts/fusion_cache_v3/<dataset>/<split>/<name>.npz.

Key diffs vs v1:
- SGM source = pre-computed tuned files (disp_raw, disp_filled, hole_mask, conf_raw)
- Cache stores `sgm_disp` = disp_raw with holes zeroed + `sgm_valid` bool field
- `confidence_map` is the unsmoothed raw PKRN * ~hole_mask
- SF driving enumerated across all 8 splits
- `fused_base` recomputed via fuse_edge_aware_residual with tuned SGM
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from scripts.common_config import (
    DEFAULT_SCENEFLOW_ROOT,
    DEFAULT_KITTI_ROOT,
    DEFAULT_ETH3D_ROOT,
    DEFAULT_MIDDLEBURY_ROOT,
    DEFAULT_WEIGHTS,
    DEFAULT_ENCODER,
    DEFAULT_INPUT_SIZE,
    DEFAULT_EDGE_THETA_LOW,
    DEFAULT_EDGE_THETA_HIGH,
    DEFAULT_EDGE_RESIDUAL_GAIN,
    DEFAULT_EDGE_DETAIL_SUPPRESSION,
    DEFAULT_ALIGN_CONF_THRESHOLD,
)
DEFAULT_ADAPTIVE_KEEP_RATIO = 0.814  # mirrors build_fusion_cache.py default
from core.fusion import fuse_edge_aware_residual
from core.fusion_net import compute_disp_scale
from core.pipeline import align_depth_to_sgm, load_da2_model, run_decoder_weight_caps_merged_da2
# Phase 9.6: eval_merge_adaptive.py was removed from repo; inline minimal builders below.
def normalize_middlebury_root(root: str) -> str:
    root = os.path.expanduser(root)
    if os.path.isdir(os.path.join(root, "MiddEval3", "trainingQ")):
        return os.path.join(root, "MiddEval3", "trainingQ")
    if os.path.isdir(os.path.join(root, "trainingQ")):
        return os.path.join(root, "trainingQ")
    return root


def build_sample_list_kitti(kitti_root: str) -> list[dict]:
    """Inlined KITTI-15 + KITTI-12 sample list (replaces deleted eval_merge_adaptive helper)."""
    samples = []
    # KITTI-15
    for left in sorted(glob(os.path.join(kitti_root, "training", "image_2", "*_10.png"))):
        frame = Path(left).name
        right = os.path.join(kitti_root, "training", "image_3", frame)
        gt = os.path.join(kitti_root, "training", "disp_occ_0", frame)
        if os.path.isfile(right) and os.path.isfile(gt):
            samples.append({"left": left, "right": right, "gt_path": gt, "name": f"kitti15/{frame}"})
    # KITTI-12
    root_12 = os.path.join(kitti_root, "kitti2012")
    for left in sorted(glob(os.path.join(root_12, "training", "colored_0", "*_10.png"))):
        frame = Path(left).name
        right = os.path.join(root_12, "training", "colored_1", frame)
        gt = os.path.join(root_12, "training", "disp_occ", frame)
        if os.path.isfile(right) and os.path.isfile(gt):
            samples.append({"left": left, "right": right, "gt_path": gt, "name": f"kitti12/{frame}"})
    return samples


def build_sample_list_middlebury(training_root: str) -> list[dict]:
    """Inlined Middlebury sample list."""
    samples = []
    for scene_dir in sorted(Path(training_root).iterdir()):
        if not scene_dir.is_dir():
            continue
        left = scene_dir / "im0.png"
        right = scene_dir / "im1.png"
        gt = scene_dir / "disp0GT.pfm"
        if left.is_file() and right.is_file() and gt.is_file():
            samples.append({"left": str(left), "right": str(right), "gt_path": str(gt), "name": scene_dir.name})
    return samples


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        color = header == "PF"
        dims = f.readline().decode("latin-1").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("latin-1").strip()
        w, h = map(int, dims.split())
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=endian + "f4").reshape(h, w)
        return np.flipud(data)


def read_disparity(path: str, dataset: str) -> np.ndarray:
    if path.endswith(".pfm"):
        disp = read_pfm(path).astype(np.float32)
        if dataset in ("eth3d", "middlebury"):
            disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        return disp
    if path.endswith(".png"):
        return cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0
    raise ValueError(f"Unknown disparity format: {path}")


def resize_map(arr: np.ndarray, target_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    if arr.shape[:2] == target_hw:
        return arr.astype(arr.dtype)
    h, w = target_hw
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=interpolation)


# ---------- Sample enumeration ----------

def build_samples_sceneflow_driving_full(sf_root: str):
    out = []
    for focal in ("15mm_focallength", "35mm_focallength"):
        for scene in ("scene_backwards", "scene_forwards"):
            for speed in ("fast", "slow"):
                left_dir = os.path.join(sf_root, "frames_cleanpass", focal, scene, speed, "left")
                right_dir = os.path.join(sf_root, "frames_cleanpass", focal, scene, speed, "right")
                disp_dir = os.path.join(sf_root, "disparity", focal, scene, speed, "left")
                if not os.path.isdir(left_dir):
                    continue
                for left in sorted(glob(os.path.join(left_dir, "*.png"))):
                    frame = Path(left).stem
                    right = os.path.join(right_dir, f"{frame}.png")
                    gt = os.path.join(disp_dir, f"{frame}.pfm")
                    if not (os.path.isfile(right) and os.path.isfile(gt)):
                        continue
                    name = f"{focal}__{scene}__{speed}__{frame}"
                    out.append({
                        "left": left, "right": right, "gt_path": gt, "name": name,
                        "sgm_npz": f"/tmp/sgm_tuned_all/sceneflow/{name}.npz",
                    })
    return out


def build_samples_kitti(kitti_root: str):
    out = []
    for s in build_sample_list_kitti(kitti_root):
        nm = s["name"].replace("/", "__")
        out.append({
            "left": s["left"], "right": s["right"], "gt_path": s["gt_path"], "name": nm,
            "sgm_npz": f"/tmp/sgm_tuned_all/kitti/{nm}.npz",
        })
    return out


def build_samples_eth3d(eth3d_root: str):
    out = []
    for im0 in sorted(glob(os.path.join(eth3d_root, "*/im0.png"))):
        scene_dir = Path(im0).parent
        scene = scene_dir.name
        if scene == "training":
            continue
        gt = scene_dir / "disp0GT.pfm"
        right = scene_dir / "im1.png"
        if not (gt.is_file() and right.is_file()):
            continue
        out.append({
            "left": str(im0), "right": str(right), "gt_path": str(gt), "name": scene,
            "sgm_npz": f"/tmp/sgm_tuned_all/eth3d/{scene}.npz",
        })
    return out


def build_samples_monkaa(sf_extract_root: str):
    """monkaa/frames_cleanpass/<scene>/left and monkaa/disparity/<scene>/left"""
    root = os.path.join(sf_extract_root, "monkaa")
    out = []
    frame_root = os.path.join(root, "frames_cleanpass")
    disp_root = os.path.join(root, "disparity")
    for scene in sorted(os.listdir(frame_root)):
        left_dir = os.path.join(frame_root, scene, "left")
        right_dir = os.path.join(frame_root, scene, "right")
        disp_dir = os.path.join(disp_root, scene, "left")
        if not os.path.isdir(left_dir):
            continue
        for left in sorted(glob(os.path.join(left_dir, "*.png"))):
            frame = Path(left).stem
            right = os.path.join(right_dir, f"{frame}.png")
            gt = os.path.join(disp_dir, f"{frame}.pfm")
            if not (os.path.isfile(right) and os.path.isfile(gt)):
                continue
            name = f"monkaa__{scene}__{frame}"
            out.append({
                "left": left, "right": right, "gt_path": gt, "name": name,
                "sgm_npz": f"/tmp/sgm_tuned_all/monkaa/{name}.npz",
            })
    return out


def build_samples_flyingthings(sf_extract_root: str, subsample: int = 4):
    """flyingthings3d/frames_cleanpass/TRAIN/{A,B,C}/<seq>/left; only TRAIN to avoid TEST leak."""
    root = os.path.join(sf_extract_root, "flyingthings3d")
    frame_root = os.path.join(root, "frames_cleanpass", "TRAIN")
    disp_root = os.path.join(root, "disparity", "TRAIN")
    out = []
    idx = 0
    for letter in sorted(os.listdir(frame_root)):
        letter_frame = os.path.join(frame_root, letter)
        letter_disp = os.path.join(disp_root, letter)
        if not os.path.isdir(letter_frame):
            continue
        for seq in sorted(os.listdir(letter_frame)):
            left_dir = os.path.join(letter_frame, seq, "left")
            right_dir = os.path.join(letter_frame, seq, "right")
            disp_dir = os.path.join(letter_disp, seq, "left")
            if not os.path.isdir(left_dir):
                continue
            for left in sorted(glob(os.path.join(left_dir, "*.png"))):
                if idx % subsample != 0:
                    idx += 1
                    continue
                idx += 1
                frame = Path(left).stem
                right = os.path.join(right_dir, f"{frame}.png")
                gt = os.path.join(disp_dir, f"{frame}.pfm")
                if not (os.path.isfile(right) and os.path.isfile(gt)):
                    continue
                name = f"ft__TRAIN__{letter}__{seq}__{frame}"
                out.append({
                    "left": left, "right": right, "gt_path": gt, "name": name,
                    "sgm_npz": f"/tmp/sgm_tuned_all/flyingthings/{name}.npz",
                })
    return out


def build_samples_middlebury(mid_root: str):
    training_root = normalize_middlebury_root(mid_root)
    out = []
    for s in build_sample_list_middlebury(training_root):
        scene = s.get("name", Path(s["left"]).parent.name)
        out.append({
            "left": s["left"], "right": s["right"], "gt_path": s["gt_path"], "name": scene,
            "sgm_npz": f"/tmp/sgm_tuned_all/middlebury/{scene}.npz",
        })
    return out


def split_name(dataset: str, index: int) -> str:
    if dataset == "sceneflow":
        if index % 10 == 0: return "val"
        if index % 10 == 1: return "test"
        return "train"
    if dataset == "kitti":
        return "val" if index % 10 == 0 else "train"
    if dataset in ("monkaa", "flyingthings"):
        # SF-style split: 1/10 val, 1/10 test, rest train
        if index % 10 == 0: return "val"
        if index % 10 == 1: return "test"
        return "train"
    return "eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["kitti", "sceneflow", "eth3d", "middlebury", "monkaa", "flyingthings"])
    ap.add_argument("--out-root", default="artifacts/fusion_cache_v3")
    ap.add_argument("--sceneflow-root", default=DEFAULT_SCENEFLOW_ROOT)
    ap.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    ap.add_argument("--eth3d-root", default=DEFAULT_ETH3D_ROOT)
    ap.add_argument("--middlebury-root", default=DEFAULT_MIDDLEBURY_ROOT)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--keep-ratio", type=float, default=DEFAULT_ADAPTIVE_KEEP_RATIO)
    ap.add_argument("--merge-layer", type=int, default=0)
    ap.add_argument("--align-conf-threshold", type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD)
    ap.add_argument("--decoder-high-precision-ratio", type=float, default=0.75)
    ap.add_argument("--decoder-low-bits", type=int, default=4)
    ap.add_argument("--decoder-stage-policy", default="coarse_only", choices=["all", "coarse_only", "fine_only"])
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"[init] loading DA2 encoder={args.encoder} weights={args.weights}", flush=True)
    model = load_da2_model(args.encoder, args.weights, device)

    if args.dataset == "sceneflow":
        samples = build_samples_sceneflow_driving_full(args.sceneflow_root)
    elif args.dataset == "kitti":
        samples = build_samples_kitti(args.kitti_root)
    elif args.dataset == "eth3d":
        samples = build_samples_eth3d(args.eth3d_root)
    elif args.dataset == "monkaa":
        samples = build_samples_monkaa(os.path.dirname(args.sceneflow_root))
    elif args.dataset == "flyingthings":
        samples = build_samples_flyingthings(os.path.dirname(args.sceneflow_root), subsample=4)
    else:
        samples = build_samples_middlebury(args.middlebury_root)

    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"[init] dataset={args.dataset}  n={len(samples)}", flush=True)

    t0_all = time.perf_counter()
    done = 0; skipped = 0
    for idx, s in enumerate(samples):
        split = split_name(args.dataset, idx)
        out_dir = os.path.join(args.out_root, args.dataset, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{s['name']}.npz")
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue
        if not os.path.isfile(s["sgm_npz"]):
            if idx < 5:
                print(f"[{idx+1}/{len(samples)}] {s['name']}: missing tuned SGM {s['sgm_npz']}", flush=True)
            skipped += 1
            continue

        left_bgr = cv2.imread(s["left"], cv2.IMREAD_COLOR)
        if left_bgr is None:
            print(f"[{idx+1}/{len(samples)}] {s['name']}: left unreadable", flush=True); continue
        gt_disp_raw = read_disparity(s["gt_path"], args.dataset)
        valid_raw = np.isfinite(gt_disp_raw) & (gt_disp_raw > 0)

        sgm_data = np.load(s["sgm_npz"])
        disp_raw = sgm_data["disp_raw"].astype(np.float32)
        disp_filled = sgm_data["disp_filled"].astype(np.float32)
        hole_mask = sgm_data["hole_mask"].astype(bool)
        conf_raw = sgm_data["confidence"].astype(np.float32)

        # Ensure RGB matches SGM shape; resize if needed.
        if left_bgr.shape[:2] != disp_raw.shape:
            left_bgr = cv2.resize(left_bgr, (disp_raw.shape[1], disp_raw.shape[0]), interpolation=cv2.INTER_AREA)
        if gt_disp_raw.shape != disp_raw.shape:
            gt_disp_raw = cv2.resize(gt_disp_raw, (disp_raw.shape[1], disp_raw.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid_raw = cv2.resize(valid_raw.astype(np.uint8), (disp_raw.shape[1], disp_raw.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        # DA2 mono inference (takes conf_raw as routing signal).
        try:
            mono_depth = run_decoder_weight_caps_merged_da2(
                model=model,
                image_bgr=left_bgr,
                confidence_map=conf_raw,
                keep_ratio=args.keep_ratio,
                input_size=args.input_size,
                merge_layer=args.merge_layer,
                decoder_high_precision_ratio=args.decoder_high_precision_ratio,
                low_precision_bits=args.decoder_low_bits,
                decoder_conf_weight=1.0,
                decoder_texture_weight=0.0,
                decoder_variance_weight=0.0,
                decoder_stage_policy=args.decoder_stage_policy,
            )
        except Exception as e:
            print(f"[{idx+1}/{len(samples)}] {s['name']}: DA2 error {e}", flush=True); continue

        # Align mono → SGM space. Use disp_filled (more anchor points) + conf_raw threshold.
        mono_disp_aligned, align_scale, align_shift = align_depth_to_sgm(
            depth_mono=mono_depth,
            disparity_raw=disp_filled,
            confidence_map=conf_raw,
            conf_threshold=args.align_conf_threshold,
        )

        # Resize SGM + auxiliaries to DA2 output resolution if different.
        target_hw = mono_disp_aligned.shape[:2]
        disp_raw_rs = resize_map(disp_raw, target_hw, cv2.INTER_NEAREST)
        disp_filled_rs = resize_map(disp_filled, target_hw, cv2.INTER_NEAREST)
        conf_raw_rs = resize_map(conf_raw, target_hw, cv2.INTER_LINEAR)
        hole_mask_rs = resize_map(hole_mask.astype(np.float32), target_hw, cv2.INTER_NEAREST) > 0.5
        gt_disp = resize_map(gt_disp_raw, target_hw, cv2.INTER_NEAREST).astype(np.float32)
        valid_mask = resize_map(valid_raw.astype(np.float32), target_hw, cv2.INTER_NEAREST) > 0.5

        # Heuristic (edge-aware residual) using tuned (filled) SGM.
        fused_base, debug = fuse_edge_aware_residual(
            sgm_disp=disp_filled_rs,
            da2_aligned=mono_disp_aligned,
            confidence_map=conf_raw_rs,
            image_bgr=left_bgr,
            theta_low=DEFAULT_EDGE_THETA_LOW,
            theta_high=DEFAULT_EDGE_THETA_HIGH,
            detail_suppression=DEFAULT_EDGE_DETAIL_SUPPRESSION,
            residual_gain=DEFAULT_EDGE_RESIDUAL_GAIN,
            return_debug=True,
        )
        fused_base = fused_base.astype(np.float32)
        detail_score = debug.get("detail_score", np.zeros_like(fused_base)).astype(np.float32)

        # Final SGM saved to cache = raw + holes zeroed; sgm_valid explicit bool.
        sgm_disp_save = disp_raw_rs.copy()
        sgm_disp_save[hole_mask_rs] = 0.0
        sgm_valid = (~hole_mask_rs).astype(bool)

        disp_scale = compute_disp_scale(fused_base, valid_mask)

        np.savez_compressed(
            out_path.removesuffix(".npz"),
            rgb=cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8),
            mono_disp_aligned=mono_disp_aligned.astype(np.float32),
            sgm_disp=sgm_disp_save.astype(np.float32),
            confidence_map=conf_raw_rs.astype(np.float32),
            sgm_valid=sgm_valid,
            fused_base=fused_base,
            detail_score=detail_score,
            gt_disp=gt_disp,
            valid_mask=valid_mask,
            disp_scale=np.float32(disp_scale),
            align_scale=np.float32(align_scale),
            align_shift=np.float32(align_shift),
            sample_name=s["name"],
            dataset_name=args.dataset,
        )
        done += 1

        if done % 20 == 0 or idx == len(samples) - 1:
            el = time.perf_counter() - t0_all
            rate = done / max(el, 1.0)
            eta = (len(samples) - idx - 1) / max(rate, 1e-6)
            print(
                f"[{idx+1}/{len(samples)}] split={split} {s['name']:60s} "
                f"done={done} skip={skipped} elapsed={el/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )

    print(f"\n[done] dataset={args.dataset}  done={done}  skipped={skipped}  total={(time.perf_counter()-t0_all)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
