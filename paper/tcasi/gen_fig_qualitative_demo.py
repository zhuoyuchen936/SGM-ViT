#!/usr/bin/env python3
"""F10: 4x5 qualitative demo grid (rows=datasets, cols=[left,gt,mono,heur,effvit]).

Falls back from demo_phase8_pareto (has only 00_summary.png) to demo_phase7_iter1
which contains per-stage PNGs we need.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

BASE = "/home/pdongaa/workspace/SGM-ViT"
OUT_PATH = f"{BASE}/paper/tcasi/fig_qualitative_demo.pdf"

# (label, demo_dir, scene_name)
# Use phase7_iter1 because phase8_pareto only contains summary.png
ROWS = [
    ("KITTI",      f"{BASE}/results/demo_phase7_iter1/kitti",      "kitti12__000010_10.png"),
    ("SceneFlow",  f"{BASE}/results/demo_phase7_iter1/sceneflow",  "15mm_focallength__scene_backwards__fast__0012"),
    ("ETH3D",      f"{BASE}/results/demo_phase7_iter1/eth3d",      "delivery_area_1l"),
    ("Middlebury", f"{BASE}/results/demo_phase7_iter1/middlebury", "ArtL"),
]

COLS = [
    ("Left",       "01_left.png"),
    ("GT Disp",    "02_gt.png"),
    ("Mono (DA2)", "03_mono.png"),
    ("SGM+Heur.",  "05_heuristic.png"),
    ("EffViT (Ours)", "06_effvit.png"),
]

fig, axes = plt.subplots(len(ROWS), len(COLS),
                         figsize=(2.05 * len(COLS), 1.55 * len(ROWS)))
if len(ROWS) == 1:
    axes = axes[None, :]

for r, (row_label, root, scene) in enumerate(ROWS):
    scene_dir = os.path.join(root, scene)
    for c, (col_label, fname) in enumerate(COLS):
        ax = axes[r, c]
        fp = os.path.join(scene_dir, fname)
        if os.path.exists(fp):
            try:
                img = mpimg.imread(fp)
                ax.imshow(img, aspect="auto")
            except Exception as e:
                ax.text(0.5, 0.5, f"[load err]\n{e}", ha="center", va="center",
                        fontsize=6, transform=ax.transAxes)
        else:
            # try alt filename (phase7 sometimes varies)
            alt = fname.replace("_heuristic", "_heuristic_fused").replace("_mono", "_mono_aligned")
            fp2 = os.path.join(scene_dir, alt)
            if os.path.exists(fp2):
                img = mpimg.imread(fp2)
                ax.imshow(img, aspect="auto")
            else:
                ax.text(0.5, 0.5, "[missing]", ha="center", va="center", fontsize=7,
                        transform=ax.transAxes, color="grey")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if r == 0:
            ax.set_title(col_label, fontsize=9)
        if c == 0:
            ax.set_ylabel(row_label, fontsize=10, rotation=90, labelpad=6)

fig.suptitle("EffViT-Depth B1-h24 qualitative results across 4 Datasets",
             fontsize=10.5, y=1.0)
fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.04)
print(f"Wrote {OUT_PATH}")
