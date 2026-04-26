#!/usr/bin/env python3
"""Generate Fig. op-mix pie chart for Section VII-E (A3 ablation)."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
OP_MIX_PATH = os.path.join(_ROOT, "results", "ablation_a3", "op_mix.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "fig_isa_coverage.pdf")


def main():
    if not os.path.isfile(OP_MIX_PATH):
        print(f"[err] {OP_MIX_PATH} not found; run scripts/ablation_a3_isa_coverage.py first", file=sys.stderr)
        sys.exit(1)

    with open(OP_MIX_PATH) as f:
        data = json.load(f)
    shares = data["cycle_share_pct"]

    # Group the "<1%" tail into one slice for readability.
    ordered = sorted(shares.items(), key=lambda kv: -kv[1])
    big = [(k, v) for k, v in ordered if v >= 1.0]
    small = [(k, v) for k, v in ordered if v < 1.0]
    labels = [k for k, _ in big]
    sizes = [v for _, v in big]
    if small:
        labels.append("Others (<1% each)")
        sizes.append(sum(v for _, v in small))

    # Paper-palette greens/blues/oranges.
    colors = [
        "#4a86e8",  # deep blue — largest (CONV_3X3_DW)
        "#6c8ebf",
        "#82b366",
        "#cce5ff",
        "#d5e8d4",
        "#ffe6cc",
        "#f8cecc",
        "#dae8fc",
        "#d79b00",
        "#b85450",
        "#fff2cc",
    ][: len(labels)]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= 2.0 else "",
        startangle=90,
        pctdistance=0.72,
        colors=colors,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")

    ax.legend(
        wedges,
        [f"{l}  ({s:.1f}%)" for l, s in zip(labels, sizes)],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
        frameon=False,
    )
    ax.set_title("FusionEngineV2 op-mix — B1-h24 INT8 per-frame cycle share", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.02)
    print(f"[gen] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
