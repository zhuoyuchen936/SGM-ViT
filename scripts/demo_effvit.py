#!/usr/bin/env python3
"""Phase 6 demo: generate 4-dataset visualizations comparing mono/heuristic/EffViT predictions."""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import cv2
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


def colorize_disp(disp: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    d = disp.astype(np.float32).copy()
    valid = np.isfinite(d) & (d > 0)
    if vmin is None:
        vmin = float(np.percentile(d[valid], 2)) if valid.any() else 0.0
    if vmax is None:
        vmax = float(np.percentile(d[valid], 98)) if valid.any() else 1.0
    d = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    d = (d * 255).astype(np.uint8)
    out = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    invalid = ~valid
    out[invalid] = (40, 40, 40)
    return out, (vmin, vmax)


def error_map(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, tau: float) -> np.ndarray:
    err = np.abs(pred - gt)
    bad = (err > tau) & valid
    img = np.zeros((*err.shape, 3), dtype=np.uint8)
    img[valid] = (0, 120, 0)
    img[bad] = (0, 0, 220)
    img[~valid] = (30, 30, 30)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-root", default="artifacts/fusion_cache")
    ap.add_argument("--out-dir", default="results/demo_phase6_iter2")
    ap.add_argument("--sample-idx", type=int, default=0, help="which sample in the sorted list")
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    state = torch.load(args.ckpt, map_location="cpu")
    variant = state.get("variant", "b1")
    model = EffViTDepthNet(variant=variant).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"[ckpt] {args.ckpt}  variant={variant}", flush=True)

    taus = {"kitti": 3.0, "sceneflow": 1.0, "eth3d": 1.0, "middlebury": 2.0}

    for dataset, split in DATASET_SPLITS.items():
        cache_dir = os.path.join(args.cache_root, dataset, split)
        files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if not files:
            print(f"[{dataset}] no samples")
            continue
        idx = min(args.sample_idx, len(files) - 1)
        path = files[idx]
        data = np.load(path)
        rgb_np = data["rgb"].astype(np.uint8)
        mono_np = data["mono_disp_aligned"].astype(np.float32)
        sgm_np = data["sgm_disp"].astype(np.float32)
        conf_np = data["confidence_map"].astype(np.float32)
        gt_np = np.nan_to_num(data["gt_disp"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid_np = data["valid_mask"].astype(bool)
        disp_scale = float(data["disp_scale"])
        heuristic_np = data["fused_base"].astype(np.float32) if "fused_base" in data.files else mono_np.copy()

        # Run EffViT
        rgb_t = torch.from_numpy(rgb_np.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        mono_t = torch.from_numpy(mono_np).float().unsqueeze(0).unsqueeze(0).to(device)
        sgm_t = torch.from_numpy(sgm_np).float().unsqueeze(0).unsqueeze(0).to(device)
        conf_t = torch.from_numpy(conf_np).float().unsqueeze(0).unsqueeze(0).to(device)
        scale_t = torch.tensor([disp_scale]).float().to(device)
        rgb_p, pads = _pad_to_multiple(rgb_t, 32)
        mono_p, _ = _pad_to_multiple(mono_t, 32)
        sgm_p, _ = _pad_to_multiple(sgm_t, 32)
        conf_p, _ = _pad_to_multiple(conf_t, 32)
        with torch.no_grad():
            x = build_effvit_depth_inputs(rgb_p, mono_p, sgm_p, conf_p, scale_t)
            residual_n = model(x)
            pred_pad = mono_p + residual_n * scale_t.view(1, 1, 1, 1).clamp_min(1.0)
            pred = _unpad(pred_pad, pads)
        effvit_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        # Compute vmin/vmax from GT valid region for consistent coloring.
        if valid_np.any():
            vmin = float(np.percentile(gt_np[valid_np], 2))
            vmax = float(np.percentile(gt_np[valid_np], 98))
        else:
            vmin, vmax = None, None

        out_sub = os.path.join(args.out_dir, dataset)
        os.makedirs(out_sub, exist_ok=True)
        cv2.imwrite(os.path.join(out_sub, "01_left.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        gt_col, _ = colorize_disp(gt_np, vmin, vmax)
        mono_col, _ = colorize_disp(mono_np, vmin, vmax)
        heur_col, _ = colorize_disp(heuristic_np, vmin, vmax)
        effvit_col, _ = colorize_disp(effvit_np, vmin, vmax)
        cv2.imwrite(os.path.join(out_sub, "02_gt.png"), gt_col)
        cv2.imwrite(os.path.join(out_sub, "03_mono_aligned.png"), mono_col)
        cv2.imwrite(os.path.join(out_sub, "04_heuristic_fused.png"), heur_col)
        cv2.imwrite(os.path.join(out_sub, "05_effvit_phase6.png"), effvit_col)

        # Error maps at dataset's tau (green=ok, red=bad, black=invalid).
        tau = taus[dataset]
        cv2.imwrite(os.path.join(out_sub, "06_effvit_error.png"), error_map(effvit_np, gt_np, valid_np, tau))
        cv2.imwrite(os.path.join(out_sub, "07_heuristic_error.png"), error_map(heuristic_np, gt_np, valid_np, tau))

        # Compute per-method metrics (for summary titles).
        def _metric(p):
            if not valid_np.any():
                return None, None
            e = float(np.abs(p[valid_np] - gt_np[valid_np]).mean())
            b = float(((np.abs(p - gt_np) > tau) & valid_np).sum()) / max(valid_np.sum(), 1) * 100.0
            return e, b

        sgm_col, _ = colorize_disp(sgm_np, vmin, vmax)
        m_mono = _metric(mono_np)
        m_sgm = _metric(sgm_np)
        m_heur = _metric(heuristic_np)
        m_effvit = _metric(effvit_np)

        badk = f"bad{int(tau)}" if float(tau).is_integer() else f"bad{tau}"

        def _title(name: str, mt):
            if mt is None or mt[0] is None:
                return name
            return f"{name}   EPE={mt[0]:.3f}   {badk}={mt[1]:.1f}%"

        # Save SGM panel as its own file too.
        cv2.imwrite(os.path.join(out_sub, "03b_sgm.png"), sgm_col)

        # Combined summary (VERTICAL stack with per-panel title).
        panels = [
            cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR), gt_col, mono_col, sgm_col, heur_col, effvit_col,
        ]
        titles = [
            "left",
            "gt",
            _title("mono (DA2)", m_mono),
            _title("sgm", m_sgm),
            _title("heuristic", m_heur),
            _title("effvit-phase6", m_effvit),
        ]
        # Match widths (pad-right to max width) and add a label header bar above each.
        W_max = max(p.shape[1] for p in panels)
        header_h = 36
        rows = []
        for p, t in zip(panels, titles):
            if p.shape[1] < W_max:
                p = cv2.copyMakeBorder(p, 0, 0, 0, W_max - p.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            header = np.zeros((header_h, W_max, 3), dtype=np.uint8)
            cv2.putText(header, t, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(np.vstack([header, p]))
        sep = np.zeros((3, W_max, 3), dtype=np.uint8)
        sep[:] = (60, 60, 60)
        stacked = []
        for i, r in enumerate(rows):
            if i:
                stacked.append(sep)
            stacked.append(r)
        summary = np.vstack(stacked)
        cv2.imwrite(os.path.join(out_sub, "00_summary.png"), summary)

        # Log line.
        if valid_np.any():
            print(
                f"[{dataset}] sample={os.path.basename(path)}  "
                f"mono: epe={m_mono[0]:.3f} {badk}={m_mono[1]:.1f}% | "
                f"sgm: epe={m_sgm[0]:.3f} {badk}={m_sgm[1]:.1f}% | "
                f"heur: epe={m_heur[0]:.3f} {badk}={m_heur[1]:.1f}% | "
                f"effvit: epe={m_effvit[0]:.3f} {badk}={m_effvit[1]:.1f}%  "
                f"(saved to {out_sub})",
                flush=True,
            )
        else:
            print(f"[{dataset}] sample={os.path.basename(path)} (no valid)")

    print(f"\nDone. All outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
