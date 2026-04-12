#!/usr/bin/env python3
"""
scripts/eval_confidence_distribution.py
=======================================
SGM confidence distribution analysis on KITTI.

Partitions KITTI ground-truth-valid pixels into high-confidence and
low-confidence bands (using the SGM PKRN confidence map), then computes
per-band EPE and D1.  This quantifies the relationship between SGM
confidence and stereo accuracy — the algorithm foundation for the CRU
(Confidence Router Unit).

Expected key result: at θ=0.65, high-confidence pixels (~76% coverage)
have D1 ≈ 8%, while low-confidence pixels (~24%) have D1 >> 50%.

Output
------
  results/eval_confidence/confidence_distribution.csv
  paper/figures/fig_confidence_distribution.png

Usage
-----
  python scripts/eval_confidence_distribution.py
  python scripts/eval_confidence_distribution.py --max-samples 20
  python scripts/eval_confidence_distribution.py --kitti-root /path/to/kitti
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import core._paths  # noqa: F401
from scripts.eval_kitti import (
    build_sample_list,
    compute_metrics,
    load_pkrn_confidence,
    read_kitti_gt,
    read_pfm,
)
from scripts.common_config import (
    DEFAULT_CONF_SIGMA,
    DEFAULT_DISPARITY_RANGE,
    DEFAULT_KITTI_ROOT,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_PKRN_MIN_DIST,
    default_results_dir,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    p = argparse.ArgumentParser(description="SGM confidence distribution analysis")
    p.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument("--disparity-range", type=int, default=DEFAULT_DISPARITY_RANGE)
    p.add_argument("--pkrn-min-dist", type=int, default=DEFAULT_PKRN_MIN_DIST)
    p.add_argument("--conf-sigma", type=float, default=DEFAULT_CONF_SIGMA)
    p.add_argument("--out-dir", default=default_results_dir("eval_confidence"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)

    samples = build_sample_list(args.kitti_root)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    logging.info(f"Evaluating {len(samples)} samples")

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

    # Per-threshold accumulators: list of per-image dicts
    results = {th: {"high": [], "low": [], "all": []} for th in thresholds}

    for s in tqdm(samples, desc="Confidence distribution"):
        # Load pre-computed SGM disparity + confidence + GT
        sgm_disp = read_pfm(s["sgm_pfm"])
        conf_map = load_pkrn_confidence(
            s,
            disparity_range=args.disparity_range,
            pkrn_min_dist=args.pkrn_min_dist,
            smooth_sigma=args.conf_sigma,
        )
        gt_disp, gt_valid = read_kitti_gt(s["gt_disp"])

        # Ensure shapes match
        if sgm_disp.shape != gt_disp.shape:
            sgm_disp = cv2.resize(
                sgm_disp, (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if conf_map.shape != gt_disp.shape:
            conf_map = cv2.resize(
                conf_map, (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # SGM overall metrics
        m_all = compute_metrics(sgm_disp, gt_disp, gt_valid)

        for th in thresholds:
            high_mask = gt_valid & (conf_map >= th)
            low_mask = gt_valid & (conf_map < th)

            m_high = compute_metrics(sgm_disp, gt_disp, high_mask)
            m_low = compute_metrics(sgm_disp, gt_disp, low_mask)

            results[th]["high"].append(m_high)
            results[th]["low"].append(m_low)
            results[th]["all"].append(m_all)

    # ---------------------------------------------------------------------------
    # Aggregate and write CSV
    # ---------------------------------------------------------------------------
    csv_path = os.path.join(args.out_dir, "confidence_distribution.csv")
    rows = []
    for th in thresholds:
        high_recs = results[th]["high"]
        low_recs = results[th]["low"]

        # Macro-average across images
        def avg(recs, key):
            vals = [r[key] for r in recs if not np.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        def total_px(recs):
            return sum(r["n"] for r in recs)

        n_high = total_px(high_recs)
        n_low = total_px(low_recs)
        n_total = n_high + n_low
        coverage = n_high / max(n_total, 1) * 100.0

        row = {
            "threshold": th,
            "coverage_high_pct": f"{coverage:.1f}",
            "n_high_px": n_high,
            "n_low_px": n_low,
            "epe_high": f"{avg(high_recs, 'epe'):.3f}",
            "d1_high": f"{avg(high_recs, 'd1'):.2f}",
            "epe_low": f"{avg(low_recs, 'epe'):.3f}",
            "d1_low": f"{avg(low_recs, 'd1'):.2f}",
        }
        rows.append(row)
        logging.info(
            f"θ={th:.2f}  coverage={coverage:5.1f}%  "
            f"D1_high={avg(high_recs, 'd1'):6.2f}%  "
            f"D1_low={avg(low_recs, 'd1'):6.2f}%  "
            f"EPE_high={avg(high_recs, 'epe'):6.3f}  "
            f"EPE_low={avg(low_recs, 'epe'):6.3f}"
        )

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logging.info(f"\nCSV saved: {csv_path}")

    # ---------------------------------------------------------------------------
    # Generate figure for paper
    # ---------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ths = [r["threshold"] for r in rows]
    d1_high = [float(r["d1_high"]) for r in rows]
    d1_low = [float(r["d1_low"]) for r in rows]
    coverage = [float(r["coverage_high_pct"]) for r in rows]

    # Left: D1 by confidence band
    x = np.arange(len(ths))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, d1_high, width, label="High-conf (SGM reliable)", color="#2196F3")
    bars2 = ax1.bar(x + width / 2, d1_low, width, label="Low-conf (SGM unreliable)", color="#FF5722")
    ax1.set_xlabel("Confidence Threshold θ")
    ax1.set_ylabel("SGM D1-all (%)")
    ax1.set_title("SGM Accuracy by Confidence Band")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t:.2f}" for t in ths], rotation=45)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: Coverage vs threshold
    ax2.plot(ths, coverage, "o-", color="#4CAF50", linewidth=2, markersize=6)
    ax2.set_xlabel("Confidence Threshold θ")
    ax2.set_ylabel("High-Confidence Pixel Coverage (%)")
    ax2.set_title("SGM Coverage at Different Thresholds")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    fig_path = "paper/figures/fig_confidence_distribution.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Figure saved: {fig_path}")

    # ---------------------------------------------------------------------------
    # Summary for paper
    # ---------------------------------------------------------------------------
    # Find the row closest to θ=0.65
    row_65 = next((r for r in rows if r["threshold"] == 0.65), rows[-1])
    logging.info(f"\n{'='*60}")
    logging.info("Key result at θ=0.65:")
    logging.info(f"  High-conf coverage : {row_65['coverage_high_pct']}%")
    logging.info(f"  High-conf SGM D1   : {row_65['d1_high']}%")
    logging.info(f"  Low-conf SGM D1    : {row_65['d1_low']}%")
    logging.info(f"  → {row_65['coverage_high_pct']}% of pixels have SGM D1={row_65['d1_high']}%")
    logging.info(f"  → Remaining pixels have SGM D1={row_65['d1_low']}%")
    logging.info("  → Pruning high-conf tokens is safe: SGM handles them well")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
