#!/usr/bin/env python3
"""Ablation A1 — Confidence Signal Unification.

Evaluates the EffViT fusion pipeline under four confidence-signal regimes on all
4 datasets, producing EPE / D1 / bad1 / bad2 per variant per dataset. This is the
inference-time variant of the study; no retraining. Results feed Table VII.C and
Section VII-C of the TCAS-I paper.

Variants:
  V0  Ours               unified SGM PKRN confidence broadcast to all SCUs
  V1  Random mask        PKRN replaced by a random mask of matched mean / density
  V2  Independent signals per SCU (image-gradient pseudo-conf drives fusion,
                                    SGM valid-mask drives encoder path)
  V3  No confidence      uniform constant 0.5 everywhere; disables threshold/top-k logic

For V2 we approximate the "each SCU has its own signal" hypothesis at inference:
  - fusion input conf = normalized image gradient magnitude (replaces PKRN)
  - sgm_valid used as-is (substitutes CRM/GSU conf)
The control-area cost of supporting 4 independent signals is estimated separately
by scripts/ablation_a1_area_accounting.py.

Usage:
  python scripts/ablation_a1_conf_unification.py \
      --ckpt ckpts/phase9_b1h24_qat_int8/best.pt \
      --variant V0 \
      --out-dir results/ablation_a1/V0

Run all 4 variants sequentially in a tmux session (see plan §Batch 2).
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

from core.effvit_depth import EffViTDepthNet, build_effvit_depth_inputs


DATASET_SPLITS = {
    "kitti": "val",
    "sceneflow": "test",
    "eth3d": "eval",
    "middlebury": "eval",
}

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


def _transform_confidence(conf: np.ndarray, rgb: np.ndarray, sgm_valid: np.ndarray,
                          variant: str, rng: np.random.Generator) -> np.ndarray:
    """Apply variant-specific confidence transform. conf/rgb are H x W; rgb is H x W x 3 [0,1]."""
    if variant == "V0":
        return conf

    if variant == "V1":
        # Random mask preserving global density (mean value).
        target_mean = float(conf.mean())
        # Shuffle pixel values instead of pure noise to preserve distribution shape.
        flat = conf.flatten().copy()
        rng.shuffle(flat)
        return flat.reshape(conf.shape).astype(np.float32)

    if variant == "V2":
        # Image-gradient-magnitude pseudo-confidence (independent-signal surrogate).
        import cv2
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx ** 2 + gy ** 2)
        # Normalize to [0, 1] and invert (low gradient -> high "confidence"): textureless regions
        # traditionally signal weak stereo coverage; this is a plausible per-SCU local signal.
        gmag /= (gmag.max() + 1e-6)
        pseudo_conf = 1.0 - np.clip(gmag, 0.0, 1.0)
        # Mask to sgm_valid region (outside valid -> 0, independent signal choice).
        pseudo_conf = pseudo_conf * sgm_valid.astype(np.float32)
        return pseudo_conf.astype(np.float32)

    if variant == "V3":
        return np.full_like(conf, 0.5, dtype=np.float32)

    raise ValueError(f"Unknown variant: {variant}")


def evaluate_dataset(model, device, cache_dir: str, dataset: str, variant: str,
                     seed: int = 0) -> dict:
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    print(f"[{dataset}] variant={variant}  {len(files)} samples", flush=True)
    if not files:
        return {"count": 0}
    tau, d1_key = DATASET_D1[dataset]
    rng = np.random.default_rng(seed)

    agg = {"effvit": {"epe": 0.0, d1_key: 0.0, "n": 0}}

    for i, path in enumerate(files):
        data = np.load(path)
        rgb_np = data["rgb"].astype(np.float32) / 255.0
        mono_np = data["mono_disp_aligned"].astype(np.float32)
        sgm_np = data["sgm_disp"].astype(np.float32)
        conf_np = data["confidence_map"].astype(np.float32)
        gt_np = np.nan_to_num(data["gt_disp"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid_np = data["valid_mask"].astype(bool)
        disp_scale = float(data["disp_scale"])
        if "sgm_valid" in data.files:
            sgm_valid_np = data["sgm_valid"].astype(bool)
        else:
            sgm_valid_np = (sgm_np > 0)

        # Variant-specific confidence transform.
        conf_transformed = _transform_confidence(conf_np, rgb_np, sgm_valid_np, variant, rng)

        rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        mono = torch.from_numpy(mono_np).float().unsqueeze(0).unsqueeze(0).to(device)
        sgm = torch.from_numpy(sgm_np).float().unsqueeze(0).unsqueeze(0).to(device)
        conf = torch.from_numpy(conf_transformed).float().unsqueeze(0).unsqueeze(0).to(device)
        scale_t = torch.tensor([disp_scale]).float().to(device)
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

        err = np.abs(effvit_np - gt_np)
        m = valid_np
        denom = max(m.sum(), 1)
        epe = float(err[m].mean()) if m.any() else 0.0
        d1 = float(((err > tau) & m).sum()) / denom * 100.0
        agg["effvit"]["epe"] += epe
        agg["effvit"][d1_key] += d1
        agg["effvit"]["n"] += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(files)}] running EPE={agg['effvit']['epe']/(i+1):.3f}", flush=True)

    n = agg["effvit"]["n"]
    return {
        "count": len(files),
        "d1_key": d1_key,
        "tau": tau,
        "variant": variant,
        "epe": agg["effvit"]["epe"] / max(n, 1),
        d1_key: agg["effvit"][d1_key] / max(n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-root", default="artifacts/fusion_cache_v3")
    ap.add_argument("--variant", required=True, choices=["V0", "V1", "V2", "V3"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", default="all", help="comma-separated list or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    state = torch.load(args.ckpt, map_location="cpu")
    variant_bb = state.get("variant", "b1")
    head_ch = state.get("head_ch", 24)
    in_channels = state.get("in_channels", 7)
    model = EffViTDepthNet(variant=variant_bb, head_ch=head_ch, in_channels=in_channels).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"[ckpt] {args.ckpt}  backbone={variant_bb}  head_ch={head_ch}  in={in_channels}", flush=True)
    print(f"[variant] {args.variant}", flush=True)

    datasets = list(DATASET_SPLITS.keys()) if args.datasets == "all" else args.datasets.split(",")

    results = {"ckpt": args.ckpt, "variant": args.variant, "per_dataset": {}}
    for dataset in datasets:
        cache_dir = os.path.join(args.cache_root, dataset, DATASET_SPLITS[dataset])
        if not os.path.isdir(cache_dir):
            print(f"[WARN] cache not found: {cache_dir} — skipping", flush=True)
            continue
        r = evaluate_dataset(model, device, cache_dir, dataset, args.variant, seed=args.seed)
        results["per_dataset"][dataset] = r
        print(f"[{dataset}] epe={r.get('epe', float('nan')):.4f}  {r.get('d1_key', 'd1')}={r.get(r.get('d1_key', 'd1'), float('nan')):.2f}%", flush=True)

    out_path = os.path.join(args.out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
