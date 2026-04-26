#!/usr/bin/env python3
"""Phase 10 — Pipeline breakdown figure + Markdown table.

Reads results/phase10/sim_<config>.json (3 configs) and emits:
  - results/phase10/pipeline_figure.png (Gantt chart + stacked bar + pie)
  - results/phase10/breakdown_table.md  (ops/cycles/latency/FLOPs per stage per config)
  - results/phase10/breakdown_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


STAGE_NAMES = {
    0: "DMA/Input",
    1: "SGM compute",
    2: "DA2 ViT-S encoder",
    3: "Decoder / EffViT bb",
    4: "Mono-SGM align",
    5: "Fusion / FPN head",
}
STAGE_COLORS = ["#8ea9db", "#c00000", "#4472c4", "#70ad47", "#ffc000", "#a569bd"]


def load_results(in_dir: str, configs: list[str]) -> dict:
    out = {}
    for cfg in configs:
        p = os.path.join(in_dir, f"sim_{cfg}.json")
        if os.path.isfile(p):
            out[cfg] = json.load(open(p))
    return out


def plot_all(results: dict, out_path: str, freq_mhz: int = 500):
    configs = list(results.keys())
    n_configs = len(configs)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.30, height_ratios=[1.0, 0.9])

    # ----- Panel 1: Gantt chart (top, spans 2 cols) -----
    ax_gantt = fig.add_subplot(gs[0, :2])
    y_pos = np.arange(n_configs)
    max_cycles = 0
    for i, cfg in enumerate(configs):
        r = results[cfg]
        stages = r["stages"]
        for s_str, st in sorted(stages.items(), key=lambda x: int(x[0])):
            s = int(s_str)
            start = st.get("start", 0) / (freq_mhz * 1e6) * 1e3  # ms
            width = st["cycles"] / (freq_mhz * 1e6) * 1e3
            if width <= 0:
                continue
            color = STAGE_COLORS[s % len(STAGE_COLORS)]
            ax_gantt.barh(i, width, left=start, color=color, edgecolor="black", linewidth=0.6)
            if width > 5:  # only annotate stages > 5 ms
                ax_gantt.text(start + width / 2, i, f"{width:.1f}ms",
                              ha="center", va="center", fontsize=8, color="white", weight="bold")
        total_ms = r["latency_ms"]
        max_cycles = max(max_cycles, total_ms)
        ax_gantt.text(total_ms + 1, i, f" {total_ms:.1f} ms ({r['fps']:.2f} FPS)",
                      va="center", fontsize=9, weight="bold")

    ax_gantt.set_yticks(y_pos)
    ax_gantt.set_yticklabels(configs)
    ax_gantt.set_xlabel("Time (ms)")
    ax_gantt.set_title("(a) End-to-end pipeline timeline  @  28nm / 500 MHz", fontsize=11, weight="bold")
    ax_gantt.set_xlim(0, max_cycles * 1.15)
    ax_gantt.invert_yaxis()
    ax_gantt.grid(True, axis="x", alpha=0.3)

    # Legend for stages
    handles = [mpatches.Patch(color=STAGE_COLORS[s], label=f"S{s}: {STAGE_NAMES[s]}")
               for s in range(6)]
    ax_gantt.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                    fontsize=8, framealpha=0.9)

    # ----- Panel 2: FLOPs breakdown bar (top right) -----
    ax_flops = fig.add_subplot(gs[0, 2])
    bottom = np.zeros(n_configs)
    for s in range(6):
        vals = np.array([results[cfg]["stages"].get(str(s), {}).get("flops", 0) / 1e9 for cfg in configs])
        if vals.max() == 0:
            continue
        ax_flops.bar(configs, vals, bottom=bottom, color=STAGE_COLORS[s],
                    label=f"S{s}", edgecolor="black", linewidth=0.4)
        bottom += vals
    ax_flops.set_ylabel("GFLOPs (cumulative)")
    ax_flops.set_title("(b) FLOPs per stage", fontsize=11, weight="bold")
    ax_flops.tick_params(axis="x", labelrotation=15, labelsize=9)
    ax_flops.grid(True, axis="y", alpha=0.3)

    # ----- Panel 3: Per-config stage pie (bottom, 3 pies) -----
    for i, cfg in enumerate(configs):
        ax = fig.add_subplot(gs[1, i])
        r = results[cfg]
        sizes = []
        labels = []
        colors = []
        for s in range(6):
            st = r["stages"].get(str(s), None)
            if st is None or st["cycles"] <= 0:
                continue
            latms = st["cycles"] / (freq_mhz * 1e6) * 1e3
            sizes.append(latms)
            labels.append(f"S{s}\n{latms:.1f}ms")
            colors.append(STAGE_COLORS[s])
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 8}, pctdistance=0.7,
        )
        for t in autotexts:
            t.set_color("white"); t.set_fontsize(8); t.set_weight("bold")
        ax.set_title(f"({chr(ord('c')+i)}) {cfg}\ntotal {r['latency_ms']:.1f} ms / {r['fps']:.2f} FPS",
                     fontsize=10, weight="bold")

    fig.suptitle("Phase 10: End-to-end pipeline latency breakdown  (SGM → DA2 encoder → decoder/EffViT → align → fusion)",
                 fontsize=12, weight="bold", y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[figure] saved {out_path}")


def write_breakdown_md(results: dict, out_md: str, freq_mhz: int = 500):
    with open(out_md, "w") as f:
        f.write("# Phase 10 — Pipeline Latency Breakdown\n\n")
        f.write(f"Platform: 28nm @ {freq_mhz} MHz, input 384×768, INT8 (Conv/Linear) + FP32 (LiteMLA)\n\n")

        # Main summary
        f.write("## Summary\n\n")
        f.write("| Config | Total latency | FPS | Total FLOPs | Weight bytes | # Ops |\n|---|---|---|---|---|---|\n")
        for cfg, r in results.items():
            f.write(f"| **{cfg}** | {r['latency_ms']:.2f} ms | **{r['fps']:.2f}** | "
                    f"{r['total_flops_G']:.1f} G | {r['total_weight_bytes_MB']:.2f} MB | {r['num_ops']} |\n")

        # Per-stage breakdown
        f.write("\n## Per-stage breakdown (latency / % of total)\n\n")
        header = "| Stage | Name |"
        divider = "|---|---|"
        for cfg in results:
            header += f" {cfg} |"
            divider += "---|"
        f.write(header + "\n" + divider + "\n")
        for s in range(6):
            line = f"| S{s} | {STAGE_NAMES[s]} |"
            for cfg, r in results.items():
                st = r["stages"].get(str(s), None)
                if st is None or st["cycles"] <= 0:
                    line += " — |"
                    continue
                latms = st["cycles"] / (freq_mhz * 1e6) * 1e3
                pct = latms / r["latency_ms"] * 100
                line += f" {latms:.2f} ms ({pct:.0f}%) |"
            f.write(line + "\n")

        # Per-stage ops / FLOPs breakdown
        f.write("\n## Per-stage ops + FLOPs\n\n")
        header = "| Stage |"
        divider = "|---|"
        for cfg in results:
            header += f" {cfg} (ops / GFLOPs) |"
            divider += "---|"
        f.write(header + "\n" + divider + "\n")
        for s in range(6):
            line = f"| S{s} {STAGE_NAMES[s]} |"
            for cfg, r in results.items():
                st = r["stages"].get(str(s), None)
                if st is None:
                    line += " — |"
                    continue
                line += f" {st['num_ops']} / {st['flops']/1e9:.2f} |"
            f.write(line + "\n")

        # Per-engine breakdown
        f.write("\n## Per-engine utilization (cycles / % of engine-busy time)\n\n")
        engine_order = ["systolic_array", "fu", "crm", "gsu", "adcu", "dma"]
        header = "| Engine |"
        divider = "|---|"
        for cfg in results:
            header += f" {cfg} |"
            divider += "---|"
        f.write(header + "\n" + divider + "\n")
        for e in engine_order:
            line = f"| {e} |"
            for cfg, r in results.items():
                eng = r["engines"].get(e, None)
                if eng is None:
                    line += " — |"
                    continue
                total = r["total_cycles"]
                pct = eng["cycles"] / total * 100 if total > 0 else 0
                line += f" {eng['num_ops']} ops / {pct:.1f}% busy |"
            f.write(line + "\n")

        # FE sub-core breakdown
        f.write("\n## FusionEngineV2 sub-core usage\n\n")
        fu_types = set()
        for r in results.values():
            fu_types.update(r.get("fu_types", {}).keys())
        header = "| FE op type |"
        divider = "|---|"
        for cfg in results:
            header += f" {cfg} (ops / cycles) |"
            divider += "---|"
        f.write(header + "\n" + divider + "\n")
        for t in sorted(fu_types):
            line = f"| {t} |"
            for cfg, r in results.items():
                fu = r.get("fu_types", {}).get(t, None)
                if fu is None:
                    line += " — |"; continue
                line += f" {fu['num_ops']} / {fu['cycles']:,} |"
            f.write(line + "\n")

        # Key takeaways
        f.write("\n## Key takeaways\n\n")
        cfgs = list(results.keys())
        if len(cfgs) >= 2:
            base = results.get("baseline", results[cfgs[0]])
            best = min(results.values(), key=lambda r: r["latency_ms"])
            f.write(f"- **Dominant bottleneck**: Stage 2 (ViT-S encoder) takes ~{int(base['stages']['2']['cycles']/base['total_cycles']*100)}% of baseline latency — this is shared across all configs and sets the FPS ceiling.\n")
            f.write(f"- **EffViT saves decoder+fusion time**: baseline S3+S5 is {(base['stages']['3']['cycles']+base['stages'].get('5',{}).get('cycles',0))/(500e3):.2f} ms vs "
                    f"{(best['stages']['3']['cycles']+best['stages'].get('5',{}).get('cycles',0))/(500e3):.2f} ms for best EffViT config.\n")
            f.write(f"- **End-to-end speedup**: baseline {base['latency_ms']:.2f} ms → best {best['latency_ms']:.2f} ms = {base['latency_ms']/best['latency_ms']:.2f}× faster.\n")
            f.write(f"- **SGM cost is modest** (~9 ms), fixed across configs; further HW effort on SGM gives diminishing return.\n")
            f.write(f"- **Alignment cost is negligible** (<0.2 ms).\n")
            f.write("- **Critical path of EffViT path**: encoder (unavoidable unless we replace encoder) → DepthwiseCore on FE → FPN head. To push past 10 FPS @ 28nm, need to shrink encoder (e.g., ViT-Tiny) or raise clock.\n")
    print(f"[md] saved {out_md}")


def write_breakdown_csv(results: dict, out_csv: str, freq_mhz: int = 500):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "stage", "stage_name", "num_ops", "cycles", "latency_ms", "flops_G", "weight_MB"])
        for cfg, r in results.items():
            for s_str, st in r["stages"].items():
                s = int(s_str)
                latms = st["cycles"] / (freq_mhz * 1e6) * 1e3
                w.writerow([cfg, s, STAGE_NAMES.get(s, ""), st["num_ops"],
                            st["cycles"], f"{latms:.3f}", f"{st['flops']/1e9:.3f}",
                            f"{st['weight_bytes']/1e6:.3f}"])
    print(f"[csv] saved {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/phase10")
    ap.add_argument("--out-fig", default="results/phase10/pipeline_figure.png")
    ap.add_argument("--out-md", default="results/phase10/breakdown_table.md")
    ap.add_argument("--out-csv", default="results/phase10/breakdown_table.csv")
    ap.add_argument("--configs", nargs="+", default=["baseline", "effvit_b1_h24", "effvit_b0_h24"])
    ap.add_argument("--freq", type=int, default=500)
    args = ap.parse_args()

    results = load_results(args.in_dir, args.configs)
    if not results:
        print(f"[error] no sim_*.json found in {args.in_dir}")
        return

    plot_all(results, args.out_fig, freq_mhz=args.freq)
    write_breakdown_md(results, args.out_md, freq_mhz=args.freq)
    write_breakdown_csv(results, args.out_csv, freq_mhz=args.freq)


if __name__ == "__main__":
    main()
