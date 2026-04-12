#!/usr/bin/env python3
"""
scripts/eval_confidence_fusion_consistency.py
=============================================
Confidence-fusion weight consistency analysis on KITTI.

Proves that CRU-pruned tokens (high SGM confidence) correspond to fusion
weight α ≈ 1 (SGM-dominant), meaning DA2 computation at those locations
would be suppressed by fusion anyway.

Core insight: "CRU skips computing features that fusion would discard."

Output
------
  results/eval_confidence/fusion_consistency.csv
  paper/figures/fig_fusion_consistency.png

Usage
-----
  python scripts/eval_confidence_fusion_consistency.py
  python scripts/eval_confidence_fusion_consistency.py --max-samples 20
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import core._paths  # noqa: F401
from core.eval_utils import pool_confidence
from scripts.eval_kitti import (
    build_sample_list,
    load_pkrn_confidence,
    read_pfm,
)
from scripts.common_config import (
    DEFAULT_CONF_SIGMA,
    DEFAULT_DISPARITY_RANGE,
    DEFAULT_FUSION_CONF_THRESHOLD,
    DEFAULT_KITTI_ROOT,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_PKRN_MIN_DIST,
    DEFAULT_PRUNE_THRESHOLD,
    default_results_dir,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

TOKEN_GRID_SIZE = 37  # 518 // 14


def compute_fusion_alpha(conf_map: np.ndarray, conf_threshold: float,
                         sgm_disp: np.ndarray) -> np.ndarray:
    """Compute soft-blend fusion weight α (same formula as core/fusion.py).

    α = clip(conf / θ, 0, 1) * (sgm > 0)
    High α → SGM dominates; low α → DA2 dominates.
    """
    sgm_valid = (sgm_disp > 0).astype(np.float32)
    alpha = np.clip(conf_map / max(conf_threshold, 1e-8), 0.0, 1.0) * sgm_valid
    return alpha


def main():
    p = argparse.ArgumentParser(description="Confidence-fusion consistency analysis")
    p.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument("--disparity-range", type=int, default=DEFAULT_DISPARITY_RANGE)
    p.add_argument("--pkrn-min-dist", type=int, default=DEFAULT_PKRN_MIN_DIST)
    p.add_argument("--conf-sigma", type=float, default=DEFAULT_CONF_SIGMA)
    p.add_argument("--conf-threshold", type=float, default=DEFAULT_FUSION_CONF_THRESHOLD,
                   help="Fusion confidence threshold θ (best from fusion sweep)")
    p.add_argument("--prune-threshold", type=float, default=DEFAULT_PRUNE_THRESHOLD,
                   help="CRU pruning threshold (tokens with conf > θ are pruned)")
    p.add_argument("--out-dir", default=default_results_dir("eval_confidence"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)

    samples = build_sample_list(args.kitti_root)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    logging.info(f"Evaluating {len(samples)} samples")
    logging.info(f"  Fusion θ : {args.conf_threshold}")
    logging.info(f"  Prune θ  : {args.prune_threshold}")

    # Per-image results
    records = []

    for s in tqdm(samples, desc="Fusion consistency"):
        sgm_disp = read_pfm(s["sgm_pfm"])
        conf_map = load_pkrn_confidence(
            s,
            disparity_range=args.disparity_range,
            pkrn_min_dist=args.pkrn_min_dist,
            smooth_sigma=args.conf_sigma,
        )

        # ---- Per-pixel fusion alpha ----
        alpha = compute_fusion_alpha(conf_map, args.conf_threshold, sgm_disp)

        # ---- Pool to token grid ----
        import torch
        conf_grid = pool_confidence(conf_map, TOKEN_GRID_SIZE)  # (G, G)

        alpha_t = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0).float()
        alpha_grid = torch.nn.functional.adaptive_avg_pool2d(
            alpha_t, TOKEN_GRID_SIZE
        ).squeeze().numpy()

        # ---- CRU prune mask: True = pruned (high confidence) ----
        prune_mask = conf_grid > args.prune_threshold
        keep_mask = ~prune_mask

        n_pruned = int(prune_mask.sum())
        n_kept = int(keep_mask.sum())
        n_total = n_pruned + n_kept
        prune_ratio = n_pruned / max(n_total, 1)

        # ---- Consistency metrics ----
        if n_pruned > 0:
            alpha_pruned = alpha_grid[prune_mask]
            frac_pruned_alpha_gt09 = float((alpha_pruned > 0.9).mean())
            frac_pruned_alpha_gt08 = float((alpha_pruned > 0.8).mean())
            mean_alpha_pruned = float(alpha_pruned.mean())
        else:
            frac_pruned_alpha_gt09 = float("nan")
            frac_pruned_alpha_gt08 = float("nan")
            mean_alpha_pruned = float("nan")

        if n_kept > 0:
            alpha_kept = alpha_grid[keep_mask]
            frac_kept_alpha_lt05 = float((alpha_kept < 0.5).mean())
            frac_kept_alpha_lt03 = float((alpha_kept < 0.3).mean())
            mean_alpha_kept = float(alpha_kept.mean())
        else:
            frac_kept_alpha_lt05 = float("nan")
            frac_kept_alpha_lt03 = float("nan")
            mean_alpha_kept = float("nan")

        records.append({
            "sample": os.path.basename(s["left"]),
            "prune_ratio": f"{prune_ratio:.3f}",
            "n_pruned": n_pruned,
            "n_kept": n_kept,
            "mean_alpha_pruned": f"{mean_alpha_pruned:.4f}",
            "frac_pruned_alpha_gt09": f"{frac_pruned_alpha_gt09:.4f}",
            "frac_pruned_alpha_gt08": f"{frac_pruned_alpha_gt08:.4f}",
            "mean_alpha_kept": f"{mean_alpha_kept:.4f}",
            "frac_kept_alpha_lt05": f"{frac_kept_alpha_lt05:.4f}",
            "frac_kept_alpha_lt03": f"{frac_kept_alpha_lt03:.4f}",
        })

    # ---------------------------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------------------------
    csv_path = os.path.join(args.out_dir, "fusion_consistency.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    logging.info(f"CSV saved: {csv_path}")

    # ---------------------------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------------------------
    def safe_mean(vals):
        vals = [float(v) for v in vals if not np.isnan(float(v))]
        return np.mean(vals) if vals else float("nan")

    avg_prune_ratio = safe_mean([r["prune_ratio"] for r in records])
    avg_alpha_pruned = safe_mean([r["mean_alpha_pruned"] for r in records])
    avg_frac_gt09 = safe_mean([r["frac_pruned_alpha_gt09"] for r in records])
    avg_frac_gt08 = safe_mean([r["frac_pruned_alpha_gt08"] for r in records])
    avg_alpha_kept = safe_mean([r["mean_alpha_kept"] for r in records])
    avg_frac_lt05 = safe_mean([r["frac_kept_alpha_lt05"] for r in records])
    avg_frac_lt03 = safe_mean([r["frac_kept_alpha_lt03"] for r in records])

    logging.info(f"\n{'='*60}")
    logging.info(f"Aggregate Results ({len(records)} images)")
    logging.info(f"{'='*60}")
    logging.info(f"  Average prune ratio           : {avg_prune_ratio:.1%}")
    logging.info(f"  Pruned tokens — mean α        : {avg_alpha_pruned:.4f}")
    logging.info(f"  Pruned tokens — α > 0.9       : {avg_frac_gt09:.1%}")
    logging.info(f"  Pruned tokens — α > 0.8       : {avg_frac_gt08:.1%}")
    logging.info(f"  Kept tokens   — mean α        : {avg_alpha_kept:.4f}")
    logging.info(f"  Kept tokens   — α < 0.5       : {avg_frac_lt05:.1%}")
    logging.info(f"  Kept tokens   — α < 0.3       : {avg_frac_lt03:.1%}")
    logging.info(f"\n  → {avg_frac_gt09:.1%} of CRU-pruned tokens have fusion α > 0.9")
    logging.info("     (SGM-dominant — DA2 computation would be discarded)")
    logging.info(f"  → {avg_frac_lt05:.1%} of kept tokens have fusion α < 0.5")
    logging.info("     (DA2-dominant — ViT computation is needed here)")
    logging.info(f"{'='*60}")

    # ---------------------------------------------------------------------------
    # Generate figure
    # ---------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Distribution of α for pruned vs kept tokens
    alpha_pruned_all = []
    alpha_kept_all = []
    for s_rec, s_data in zip(records, samples):
        sgm_disp = read_pfm(s_data["sgm_pfm"])
        conf_map = load_pkrn_confidence(
            s_data,
            disparity_range=args.disparity_range,
            pkrn_min_dist=args.pkrn_min_dist,
            smooth_sigma=args.conf_sigma,
        )
        alpha = compute_fusion_alpha(conf_map, args.conf_threshold, sgm_disp)
        import torch
        conf_grid = pool_confidence(conf_map, TOKEN_GRID_SIZE)
        alpha_t = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0).float()
        alpha_grid = torch.nn.functional.adaptive_avg_pool2d(
            alpha_t, TOKEN_GRID_SIZE
        ).squeeze().numpy()
        prune_mask = conf_grid > args.prune_threshold
        if prune_mask.sum() > 0:
            alpha_pruned_all.extend(alpha_grid[prune_mask].tolist())
        if (~prune_mask).sum() > 0:
            alpha_kept_all.extend(alpha_grid[~prune_mask].tolist())

    bins = np.linspace(0, 1, 50)
    ax1.hist(alpha_pruned_all, bins=bins, alpha=0.7, density=True,
             color="#FF5722", label=f"Pruned tokens (n={len(alpha_pruned_all)})")
    ax1.hist(alpha_kept_all, bins=bins, alpha=0.7, density=True,
             color="#2196F3", label=f"Kept tokens (n={len(alpha_kept_all)})")
    ax1.set_xlabel("Fusion Weight α (higher = more SGM)")
    ax1.set_ylabel("Density")
    ax1.set_title("Fusion α Distribution: Pruned vs Kept Tokens")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axvline(x=0.9, color="red", linestyle="--", alpha=0.5, label="α=0.9")

    # Right: Summary bar chart
    categories = ["Pruned\nα > 0.9", "Pruned\nα > 0.8", "Kept\nα < 0.5", "Kept\nα < 0.3"]
    values = [avg_frac_gt09 * 100, avg_frac_gt08 * 100, avg_frac_lt05 * 100, avg_frac_lt03 * 100]
    colors = ["#FF5722", "#FF8A65", "#2196F3", "#64B5F6"]
    bars = ax2.bar(categories, values, color=colors)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("CRU-Fusion Consistency")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    fig_path = "paper/figures/fig_fusion_consistency.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
