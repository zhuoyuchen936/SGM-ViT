#!/usr/bin/env python3
"""F9 v2: SOTA scatter - 2-panel (Params, Weight Size) vs KITTI D1-all.
Adds measured MSNet points (our protocol), Phase 9.6 measured points,
FPGA/ASIC accelerator baselines.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/pdongaa/workspace/SGM-ViT"
RAW_PATH = f"{BASE}/results/phase9_sota/sota_raw.json"
OUT_PATH = f"{BASE}/paper/tcasi/fig_sota_scatter.pdf"

with open(RAW_PATH) as f:
    raw = f.read()

def _eval_arith(m):
    try:
        return str(eval(m.group(0), {"__builtins__": None}, {}))
    except Exception:
        return m.group(0)
raw = re.sub(r"\b\d+(\.\d+)?\s*[-+]\s*\d+(\.\d+)?\b", _eval_arith, raw)
data = json.loads(raw)

# (color, marker, label, z)
STYLE = {
    "learning_stereo":      ("#4a86e8", "o",  "Learning Stereo (reported)",          2),
    "lightweight_stereo":   ("#d79b00", "s",  "Lightweight Stereo (reported)",       2),
    "msnet_measured":       ("#e69138", "v",  "MSNet (measured, our val)",           3),
    "fpga_accelerator":     ("#82b366", "D",  "FPGA/ASIC Accelerator",               2),
    "traditional":          ("#888888", "X",  "Traditional (SGM)",                   1),
    "ours_heuristic":       ("#b85450", "P",  "Ours (heuristic)",                    3),
    "ours_fp32_measured":   ("#c0392b", "*",  "Ours FP32 (Ph9.6 measured)",          5),
    "ours_int8_measured":   ("#27ae60", "*",  "Ours INT8 W8A8 (measured)",           5),
    "ours_int4_measured":   ("#8e44ad", "*",  "Ours INT4 W4A8 (measured)",           5),
    "ours_kitti_ft":        ("#2980b9", "*",  "Ours KITTI-FT FP32 (measured)",       5),
}

# ---- Helpers ----
def _weight_mb_for_reported(name, params_M):
    """FP32 model weight in MB = params * 4 bytes."""
    if params_M is None or params_M <= 0:
        return None
    return params_M * 4.0

def _collect():
    pts = []  # (cat, name, params_M, weight_mb, d1, gflops)
    # Reference learning
    for e in data.get("sota_learning_stereo", []):
        p = e.get("params_M"); d1 = e.get("kitti_d1_all")
        if p is None or d1 is None: continue
        w = _weight_mb_for_reported(e["name"], p)
        pts.append(("learning_stereo", e["name"], p, w, d1, e.get("gflops")))
    # Lightweight (reported from paper)
    for e in data.get("sota_lightweight_stereo", []):
        p = e.get("params_M"); d1 = e.get("kitti_d1_all")
        if p is None or d1 is None: continue
        w = _weight_mb_for_reported(e["name"], p)
        pts.append(("lightweight_stereo", e["name"], p, w, d1, e.get("gflops")))
    # MSNet measured on our protocol
    for e in data.get("sota_measured_protocol", []):
        p = e.get("params_M"); d1 = e.get("kitti_d1_all")
        if p is None or d1 is None: continue
        w = e.get("model_weight_MB") or _weight_mb_for_reported(e["name"], p)
        pts.append((e["category"], e["name"], p, w, d1, e.get("gflops")))
    # Accelerators (often no params)
    for e in data.get("sota_accelerator", []):
        try: d1 = float(e.get("kitti_d1_all"))
        except (TypeError, ValueError): continue
        p = e.get("params_M") or 0.06   # draw on left edge
        w = e.get("weight_MB") or 0.06
        cat = "traditional" if "SGM" in e["name"] or "Hirsch" in e["name"] else "fpga_accelerator"
        pts.append((cat, e["name"], p, w, d1, e.get("gflops")))
    # Our measured points (Phase 9.6 b1_h24 FP32 / W8 / W4 / KITTI-FT)
    for e in data.get("ours_measured", []):
        pts.append((e["category"], e["name"], e["params_M"], e["model_weight_MB"],
                    e["kitti_d1_all"], e.get("gflops")))
    # Legacy "ours" (Phase 8) — still useful context
    for e in data.get("ours", []):
        cat = e["category"]
        p = e.get("params_M"); d1 = e.get("kitti_d1_all")
        if p is None or d1 is None: continue
        if p <= 0: p = 0.06
        # map old cats to a compatible set
        if cat == "ours_fp32":     cat2 = "ours_fp32_measured"
        elif cat == "ours_int8":   cat2 = "ours_int8_measured"
        elif cat == "ours_heuristic": cat2 = "ours_heuristic"
        else:                      cat2 = cat
        # compute weight_mb: prefer INT8 kb if given
        if "weight_bytes_KB_int8_conv" in e:
            w = e["weight_bytes_KB_int8_conv"] / 1024.0
        elif "weight_bytes_KB_fp32" in e:
            w = e["weight_bytes_KB_fp32"] / 1024.0
        else:
            w = _weight_mb_for_reported(e["name"], p)
        pts.append((cat2, e["name"], p, w, d1, e.get("gflops_at_384x768")))
    return pts

points = _collect()

# Deduplicate: prefer the measured Phase 9.6 over legacy ours; drop ours_heuristic duplicate
seen = set()
dedup = []
for t in points:
    key = (t[0], t[1])
    if key in seen:
        continue
    seen.add(key)
    dedup.append(t)
points = dedup

# ---- Plot ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0))

plotted_cats = {"L": set(), "R": set()}

def _scatter(ax, x, y, cat, name, gf, side):
    color, marker, label, z = STYLE.get(cat, ("black", "o", cat, 1))
    size = 60
    if gf is not None and gf > 0:
        size = max(30, min(320, gf * 4))
    if cat.startswith("ours"):
        size = max(size, 180)   # make ours stand out
    lab = label if cat not in plotted_cats[side] else None
    plotted_cats[side].add(cat)
    ax.scatter(x, y, s=size, c=color, marker=marker, alpha=0.85,
               edgecolors="black", linewidths=0.7, label=lab, zorder=z)

def _annotate(ax, x, y, short, cat):
    color, _, _, _ = STYLE.get(cat, ("black", "o", cat, 1))
    ax.annotate(short, (x, y), xytext=(5, 3), textcoords="offset points",
                fontsize=7, color=color)

# Panel L: Params vs D1
for cat, name, p, w, d1, gf in points:
    if p is None or p <= 0: continue
    _scatter(axL, p, d1, cat, name, gf, "L")
    if (cat.startswith("ours") or "HITNet" in name or "StereoNet" in name
        or "FP-Stereo" in name or "StereoVAE" in name or "SGM" in name
        or "MSNet" in name or "Li-JSSC" in name or "BNN-FPGA" in name):
        short = name.split(" (")[0].replace("EffViT-", "")
        # collapse our long names
        if cat.startswith("ours_") and "Ours" in short:
            short = short.replace("Ours ", "")
        _annotate(axL, p, d1, short, cat)

axL.set_xscale("log")
axL.set_xlabel("Parameters (M, log scale)", fontsize=11)
axL.set_ylabel("KITTI-2015 D1-all (%)  - lower is better", fontsize=11)
axL.set_title("(a) Accuracy vs. Model Size (Params)", fontsize=11)
axL.grid(True, which="both", alpha=0.3)
axL.set_ylim(0, 22)

# Panel R: Weight Size (MB) vs D1
for cat, name, p, w, d1, gf in points:
    if w is None or w <= 0: continue
    _scatter(axR, w, d1, cat, name, gf, "R")
    if (cat.startswith("ours") or "HITNet" in name or "FP-Stereo" in name
        or "StereoVAE" in name or "SGM" in name or "MSNet" in name
        or "Li-JSSC" in name or "BNN-FPGA" in name or "PSMNet" in name):
        short = name.split(" (")[0].replace("EffViT-", "")
        if cat.startswith("ours_") and "Ours" in short:
            short = short.replace("Ours ", "")
        _annotate(axR, w, d1, short, cat)

axR.set_xscale("log")
axR.set_xlabel("Weight Memory (MB, log scale; INT4/INT8 for quantized)", fontsize=11)
axR.set_ylabel("KITTI-2015 D1-all (%)  - lower is better", fontsize=11)
axR.set_title("(b) Accuracy vs. Weight Memory Footprint", fontsize=11)
axR.grid(True, which="both", alpha=0.3)
axR.set_ylim(0, 22)

# Single combined legend under both panels
handles, labels = axL.get_legend_handles_labels()
# Also pull any legend entries only on right panel
h_r, l_r = axR.get_legend_handles_labels()
extra = [(h, l) for h, l in zip(h_r, l_r) if l not in labels]
for h, l in extra:
    handles.append(h); labels.append(l)

fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03),
           ncol=5, fontsize=8.5, framealpha=0.92)

# Caveat note
fig.text(0.5, -0.10,
         "Reference points use KITTI test-server D1 reported by original papers; "
         "\"measured\" points (triangles, stars) use the in-house 40-image KITTI val. "
         "Marker size ~ GFLOPs where reported.",
         ha="center", fontsize=8, style="italic", color="#444")

fig.suptitle("SOTA Accuracy-Efficiency Scatter (KITTI): stereo networks, "
             "hardware accelerators, and our EdgeStereoDAv2 fusion model",
             fontsize=12, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.98])

fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.25)
fig.savefig(OUT_PATH.replace(".pdf", ".png"), dpi=150, bbox_inches="tight", pad_inches=0.25)
print(f"Wrote {OUT_PATH}")
print(f"Wrote {OUT_PATH.replace('.pdf','.png')}")
print(f"Total points plotted: L={len([t for t in points if t[2] and t[2]>0])}  R={len([t for t in points if t[3] and t[3]>0])}")
