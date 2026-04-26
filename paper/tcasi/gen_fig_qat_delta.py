#!/usr/bin/env python3
"""F8: INT8 - FP32 delta bar chart (2 variants x 4 datasets)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/pdongaa/workspace/SGM-ViT"
OUT_PATH = f"{BASE}/paper/tcasi/fig_qat_delta.pdf"

FP32 = {
    "b1_h24": f"{BASE}/results/phase8_pareto/eval/b1_h24.json",
    "b0_h24": f"{BASE}/results/phase8_pareto/eval/b0_h24.json",
}
INT8 = {
    "b1_h24": f"{BASE}/results/eval_phase9_qat_b1h24/summary.json",
    "b0_h24": f"{BASE}/results/eval_phase9_qat_b0h24/summary.json",
}

DATASETS = [
    ("kitti",      "KITTI",      "d1"),
    ("sceneflow",  "SceneFlow",  "bad1"),
    ("eth3d",      "ETH3D",      "bad1"),
    ("middlebury", "Middlebury", "bad2"),
]

def load(path):
    with open(path) as f:
        return json.load(f)

def effvit_key(methods_dict):
    # FP32 file uses 'effvit', INT8 file uses 'effvit_qat'
    for k in ("effvit", "effvit_qat"):
        if k in methods_dict:
            return k
    raise KeyError(str(methods_dict.keys()))

fig, axes = plt.subplots(2, 4, figsize=(11.5, 4.6), sharey=False)

for row, variant in enumerate(("b1_h24", "b0_h24")):
    fp = load(FP32[variant])
    iq = load(INT8[variant])
    for col, (ds, ds_label, d_key) in enumerate(DATASETS):
        ax = axes[row, col]
        fp_methods = fp["per_dataset"][ds]["methods"]
        iq_methods = iq["per_dataset"][ds]["methods"]
        fp_m = fp_methods[effvit_key(fp_methods)]
        iq_m = iq_methods[effvit_key(iq_methods)]
        metrics = ["epe", d_key]
        labels  = ["EPE", d_key.upper()]
        deltas = []
        for m in metrics:
            fp_v = fp_m.get(m, np.nan)
            iq_v = iq_m.get(m, np.nan)
            deltas.append(iq_v - fp_v)
        colors = ["#b85450" if d > 0 else "#82b366" for d in deltas]
        x = np.arange(len(metrics))
        bars = ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
        for bar, d in zip(bars, deltas):
            h = bar.get_height()
            ax.annotate(f"{d:+.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                        ha="center", fontsize=7.5)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        if col == 0:
            ax.set_ylabel(f"{variant}\nINT8 - FP32", fontsize=8.5)
        if row == 0:
            ax.set_title(ds_label, fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#b85450", edgecolor="black", label="INT8 regression (worse)"),
    Patch(facecolor="#82b366", edgecolor="black", label="INT8 improvement (better)"),
]
fig.legend(handles=legend_handles, ncol=2, fontsize=8.5, loc="lower center",
           bbox_to_anchor=(0.5, -0.03), frameon=False)

fig.suptitle("INT8 QAT vs FP32 baseline -- lower EPE / lower error-rate is better",
             fontsize=10.5, y=1.0)
fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.08)
print(f"Wrote {OUT_PATH}")
