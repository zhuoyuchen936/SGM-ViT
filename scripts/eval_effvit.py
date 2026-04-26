#!/usr/bin/env python3
"""Phase 6 Iter 1: Evaluate EffViTDepthNet on 4 datasets using existing v1 fusion_cache.

Produces summary.json with per-dataset per-method (mono, heuristic_fused, effvit) EPE / D1-bad / flat_noise.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.effvit_depth import EffViTDepthNet, build_effvit_depth_inputs, TAU_BY_DATASET


DATASET_SPLITS = {
    "kitti": "val",
    "sceneflow": "test",
    "eth3d": "eval",
    "middlebury": "eval",
}

# D1 / bad threshold and key label per dataset (matches eval_phase5 convention).
DATASET_D1 = {
    "kitti": (3.0, "d1"),
    "sceneflow": (1.0, "bad1"),
    "eth3d": (1.0, "bad1"),
    "middlebury": (2.0, "bad2"),
}


def _pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    H, W = x.shape[-2:]
    new_h = math.ceil(H / multiple) * multiple
    new_w = math.ceil(W / multiple) * multiple
    pt = (new_h - H) // 2
    pb = new_h - H - pt
    pl = (new_w - W) // 2
    pr = new_w - W - pl
    if (pt, pb, pl, pr) == (0, 0, 0, 0):
        return x, (0, 0, 0, 0)
    return F.pad(x, (pl, pr, pt, pb), mode="constant", value=0), (pt, pb, pl, pr)


def _unpad(x: torch.Tensor, pads):
    pt, pb, pl, pr = pads
    if (pt, pb, pl, pr) == (0, 0, 0, 0):
        return x
    H, W = x.shape[-2:]
    return x[..., pt:H - pb if pb else H, pl:W - pr if pr else W]


def _flat_region_noise(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, win: int = 9) -> float:
    """Simple proxy for visual smoothness: std of (pred - local mean) on flat regions (low GT gradient)."""
    import cv2
    gx = cv2.Sobel(gt, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gt, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    flat = (gmag < 1.0) & valid
    if flat.sum() < 50:
        return float("nan")
    blur = cv2.blur(pred.astype(np.float32), (win, win))
    residual = pred - blur
    return float(np.std(residual[flat]))


def evaluate_dataset(model, device, cache_dir: str, dataset: str) -> dict:
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    print(f"[{dataset}] evaluating {len(files)} samples", flush=True)
    if not files:
        return {"count": 0}
    tau, d1_key = DATASET_D1[dataset]

    agg = {
        "mono":      {"epe": 0.0, d1_key: 0.0, "flat": 0.0, "flat_n": 0, "n": 0},
        "heuristic": {"epe": 0.0, d1_key: 0.0, "flat": 0.0, "flat_n": 0, "n": 0},
        "effvit":    {"epe": 0.0, d1_key: 0.0, "flat": 0.0, "flat_n": 0, "n": 0},
    }

    for i, path in enumerate(files):
        data = np.load(path)
        rgb_np = data["rgb"].astype(np.float32) / 255.0
        mono_np = data["mono_disp_aligned"].astype(np.float32)
        sgm_np = data["sgm_disp"].astype(np.float32)
        conf_np = data["confidence_map"].astype(np.float32)
        gt_np = np.nan_to_num(data["gt_disp"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid_np = data["valid_mask"].astype(bool)
        disp_scale = float(data["disp_scale"])

        # Heuristic baseline = fused_base is the "heuristic_v1" fused product saved by build_fusion_cache.
        heuristic_np = data["fused_base"].astype(np.float32) if "fused_base" in data.files else mono_np

        rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        mono = torch.from_numpy(mono_np).float().unsqueeze(0).unsqueeze(0).to(device)
        sgm = torch.from_numpy(sgm_np).float().unsqueeze(0).unsqueeze(0).to(device)
        conf = torch.from_numpy(conf_np).float().unsqueeze(0).unsqueeze(0).to(device)
        scale_t = torch.tensor([disp_scale]).float().to(device)
        # Phase 7: read sgm_valid if present (v3 cache), else fall back to sgm>0.
        if "sgm_valid" in data.files:
            sgm_valid_np = data["sgm_valid"].astype(bool)
        else:
            sgm_valid_np = (sgm_np > 0)
        sgm_valid_t = torch.from_numpy(sgm_valid_np).to(torch.bool).unsqueeze(0).unsqueeze(0).to(device)

        rgb_p, pads = _pad_to_multiple(rgb, 32)
        mono_p, _ = _pad_to_multiple(mono, 32)
        sgm_p, _ = _pad_to_multiple(sgm, 32)
        conf_p, _ = _pad_to_multiple(conf, 32)
        sgmv_p, _ = _pad_to_multiple(sgm_valid_t, 32)
        x = build_effvit_depth_inputs(rgb_p, mono_p, sgm_p, conf_p, scale_t, sgm_valid=sgmv_p)
        with torch.no_grad():
            residual_n = model(x)
            scale_view = scale_t.view(1, 1, 1, 1).clamp_min(1.0)
            pred_pad = mono_p + residual_n * scale_view
            pred = _unpad(pred_pad, pads)
        effvit_np = pred.squeeze(0).squeeze(0).cpu().float().numpy()

        for name, pred_np in (("mono", mono_np), ("heuristic", heuristic_np), ("effvit", effvit_np)):
            err = np.abs(pred_np - gt_np)
            m = valid_np
            denom = max(m.sum(), 1)
            epe = float(err[m].mean()) if m.any() else 0.0
            d1 = float(((err > tau) & m).sum()) / denom * 100.0
            agg[name]["epe"] += epe
            agg[name][d1_key] += d1
            fl = _flat_region_noise(pred_np, gt_np, m)
            if not np.isnan(fl):
                agg[name]["flat"] += fl
                agg[name]["flat_n"] += 1
            agg[name]["n"] += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    out = {"count": len(files), "d1_key": d1_key, "tau": tau, "methods": {}}
    for name, bucket in agg.items():
        n = bucket["n"]
        avg = {
            "epe": bucket["epe"] / max(n, 1),
            d1_key: bucket[d1_key] / max(n, 1),
            "flat_noise": bucket["flat"] / max(bucket["flat_n"], 1) if bucket["flat_n"] > 0 else None,
        }
        out["methods"][name] = avg
        print(f"  {name:10s}: epe={avg['epe']:.4f}  {d1_key}={avg[d1_key]:.2f}%  flat={avg['flat_noise']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-root", default="artifacts/fusion_cache")
    ap.add_argument("--out-dir", default="results/eval_phase6_iter1")
    ap.add_argument("--variant", default="b1")
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    state = torch.load(args.ckpt, map_location="cpu")
    variant = state.get("variant", args.variant)
    head_ch = state.get("head_ch", 48)  # Phase 8 metadata; default 48 for pre-Phase-8 ckpts
    in_channels = state.get("in_channels", 8)  # pre-Phase-8 ckpts were 8-channel
    model = EffViTDepthNet(variant=variant, head_ch=head_ch, in_channels=in_channels).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"[ckpt] {args.ckpt}  variant={variant}  head_ch={head_ch}  in_channels={in_channels}", flush=True)

    results = {"ckpt": args.ckpt, "per_dataset": {}}
    for dataset, split in DATASET_SPLITS.items():
        cache_dir = os.path.join(args.cache_root, dataset, split)
        if not os.path.isdir(cache_dir):
            print(f"[{dataset}] skip (no cache at {cache_dir})", flush=True)
            continue
        results["per_dataset"][dataset] = evaluate_dataset(model, device, cache_dir, dataset)

    # Save summary
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n==== Summary (EffViT vs heuristic vs mono) ====", flush=True)
    for ds, res in results["per_dataset"].items():
        if not res or "methods" not in res:
            continue
        d1k = res["d1_key"]
        m = res["methods"]
        print(
            f"{ds:12s} "
            f"mono: epe={m['mono']['epe']:.3f} {d1k}={m['mono'][d1k]:.2f}% | "
            f"heur: epe={m['heuristic']['epe']:.3f} {d1k}={m['heuristic'][d1k]:.2f}% | "
            f"effvit: epe={m['effvit']['epe']:.3f} {d1k}={m['effvit'][d1k]:.2f}%",
            flush=True,
        )
    print(f"\nSaved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
