#!/usr/bin/env python3
"""F6: SGM PKRN confidence distribution histogram (4 subplots, one per dataset)."""
import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CACHE_ROOT = "/home/pdongaa/workspace/SGM-ViT/artifacts/fusion_cache_v3"
OUT_PATH = "/home/pdongaa/workspace/SGM-ViT/paper/tcasi/fig_conf_dist.pdf"

DATASETS = [
    ("kitti",      "KITTI-2015",       "#4a86e8"),
    ("sceneflow",  "SceneFlow",        "#d79b00"),
    ("eth3d",      "ETH3D",            "#82b366"),
    ("middlebury", "Middlebury v3",    "#b85450"),
]
SPLIT = "val"
N_SAMPLES = 10

random.seed(42)

fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.6), sharey=False)
for ax, (dset, title, color) in zip(axes, DATASETS):
    dset_dir = None
    for sub in (SPLIT, "eval", "train"):
        cand = os.path.join(CACHE_ROOT, dset, sub)
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*.npz")):
            dset_dir = cand
            break
    if dset_dir is None:
        dset_dir = os.path.join(CACHE_ROOT, dset, SPLIT)
    npz_files = sorted(glob.glob(os.path.join(dset_dir, "*.npz")))
    if len(npz_files) == 0:
        ax.set_title(f"{title}\n(no data)", fontsize=9)
        continue
    picks = random.sample(npz_files, min(N_SAMPLES, len(npz_files)))
    conf_vals = []
    for p in picks:
        try:
            d = np.load(p)
            cm = d["confidence_map"]
            valid = d["sgm_valid"] if "sgm_valid" in d.files else np.ones_like(cm, dtype=bool)
            vals = cm[valid.astype(bool)].astype(np.float32).ravel()
            # subsample to cap memory
            if vals.size > 200_000:
                idx = np.random.choice(vals.size, 200_000, replace=False)
                vals = vals[idx]
            conf_vals.append(vals)
        except Exception as e:
            print(f"Skipping {p}: {e}")
    if not conf_vals:
        ax.set_title(f"{title}\n(empty)", fontsize=9)
        continue
    all_vals = np.concatenate(conf_vals)
    # Histogram
    ax.hist(all_vals, bins=40, range=(0, 1), color=color, edgecolor="black",
            linewidth=0.3, alpha=0.85)
    # Percentile lines
    for q, ls, lab in [(25, ":", "25%"), (50, "--", "50%"), (75, "-.", "75%")]:
        v = np.percentile(all_vals, q)
        ax.axvline(v, color="black", linestyle=ls, linewidth=0.9)
        ax.text(v, ax.get_ylim()[1] * 0.92, f" {lab}={v:.2f}",
                fontsize=7, rotation=90, va="top", ha="left")
    ax.set_title(f"{title} (n={len(picks)})", fontsize=9)
    ax.set_xlabel("SGM PKRN confidence", fontsize=8)
    ax.set_xlim(0, 1)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)

axes[0].set_ylabel("Pixel count", fontsize=8)
fig.suptitle("SGM PKRN Confidence Distribution per Dataset", fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.05)
print(f"Wrote {OUT_PATH}")
