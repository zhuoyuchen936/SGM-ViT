#!/usr/bin/env python3
"""Phase 8 Stage 4: aggregate Pareto 6 variants into a table and a plot.

Reads per-variant:
  - profile JSON : results/phase8_pareto/profile/<variant>.json
  - eval summary : results/phase8_pareto/eval/<variant>.json

Writes:
  - results/phase8_pareto/summary_table.md
  - results/phase8_pareto/summary_table.csv
  - results/phase8_pareto/pareto_plot.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path


VARIANTS = [
    ("b0", 24), ("b0", 48),
    ("b1", 24), ("b1", 48),
    ("b2", 24), ("b2", 48),
]

DATASET_BAD_KEY = {"kitti": "d1", "sceneflow": "bad1", "eth3d": "bad1", "middlebury": "bad2"}


def load_variant(profile_dir: str, eval_dir: str, backbone: str, head_ch: int) -> dict | None:
    name = f"{backbone}_h{head_ch}"
    p = os.path.join(profile_dir, f"{name}.json")
    e = os.path.join(eval_dir, f"{name}.json")
    if not (os.path.isfile(p) and os.path.isfile(e)):
        return None
    with open(p) as f:
        prof = json.load(f)
    with open(e) as f:
        ev = json.load(f)
    row = {
        "variant": name,
        "backbone": backbone,
        "head_ch": head_ch,
        "params_M": prof.get("params_M"),
        "gflops": prof.get("gflops"),
    }
    for ds, cfg in ev.get("per_dataset", {}).items():
        if "methods" not in cfg:
            continue
        m = cfg["methods"].get("effvit")
        if not m:
            continue
        badk = DATASET_BAD_KEY.get(ds, "d1")
        row[f"{ds}_epe"] = round(m["epe"], 4)
        row[f"{ds}_{badk}"] = round(m.get(badk, 0.0), 3)
        # heuristic reference
        h = cfg["methods"].get("heuristic", {})
        row[f"{ds}_heur_epe"] = round(h.get("epe", 0.0), 4)
        row[f"{ds}_heur_{badk}"] = round(h.get(badk, 0.0), 3)
    # Aggregate: mean EPE over 4 datasets
    eps = [row[f"{ds}_epe"] for ds in ("kitti", "sceneflow", "eth3d", "middlebury") if f"{ds}_epe" in row]
    row["avg_epe"] = round(sum(eps) / max(len(eps), 1), 4) if eps else None
    # Dominates heuristic on all 8 metrics?
    beats_all = True
    for ds in ("kitti", "sceneflow", "eth3d", "middlebury"):
        badk = DATASET_BAD_KEY[ds]
        if row.get(f"{ds}_epe") is None or row.get(f"{ds}_heur_epe") is None:
            beats_all = False; break
        if row[f"{ds}_epe"] > row[f"{ds}_heur_epe"]:
            beats_all = False
        if row[f"{ds}_{badk}"] > row[f"{ds}_heur_{badk}"]:
            beats_all = False
    row["beats_heur_8of8"] = beats_all
    return row


def write_markdown(rows: list[dict], out_path: str):
    if not rows:
        open(out_path, "w").write("(no data)\n"); return
    with open(out_path, "w") as f:
        f.write("# Phase 8 Pareto Summary\n\n")
        f.write("| variant | params (M) | GFLOPs | KITTI EPE/D1 | SF EPE/bad1 | ETH3D EPE/bad1 | Mid EPE/bad2 | avg EPE | 8/8 > heur? |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            def fmt(a, b):
                if a is None or b is None: return "—"
                return f"{a:.3f} / {b:.2f}%"
            f.write(
                f"| {r['variant']} | {r.get('params_M','?')} | {r.get('gflops','?')} | "
                f"{fmt(r.get('kitti_epe'), r.get('kitti_d1'))} | "
                f"{fmt(r.get('sceneflow_epe'), r.get('sceneflow_bad1'))} | "
                f"{fmt(r.get('eth3d_epe'), r.get('eth3d_bad1'))} | "
                f"{fmt(r.get('middlebury_epe'), r.get('middlebury_bad2'))} | "
                f"{r.get('avg_epe','?')} | "
                f"{'✅' if r.get('beats_heur_8of8') else '❌'} |\n"
            )
        f.write("\n(EffViT row; heuristic kept in the v3 cache's fused_base field for reference.)\n")


def write_csv(rows: list[dict], out_path: str):
    if not rows:
        return
    keys = [
        "variant", "backbone", "head_ch", "params_M", "gflops", "avg_epe",
        "kitti_epe", "kitti_d1", "sceneflow_epe", "sceneflow_bad1",
        "eth3d_epe", "eth3d_bad1", "middlebury_epe", "middlebury_bad2", "beats_heur_8of8",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def plot_pareto(rows: list[dict], out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping plot", file=sys.stderr)
        return

    valid = [r for r in rows if r.get("params_M") is not None and r.get("avg_epe") is not None]
    if not valid:
        return

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"b0": "tab:green", "b1": "tab:blue", "b2": "tab:purple"}
    markers = {24: "o", 48: "s"}

    # Left: params vs avg EPE
    for r in valid:
        ax[0].scatter(r["params_M"], r["avg_epe"],
                      color=colors.get(r["backbone"], "k"),
                      marker=markers.get(r["head_ch"], "x"),
                      s=120, edgecolors="black", linewidths=0.6,
                      label=f"{r['variant']}")
        ax[0].annotate(r["variant"], (r["params_M"], r["avg_epe"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax[0].set_xlabel("Parameters (M)")
    ax[0].set_ylabel("avg EPE (4 datasets)")
    ax[0].set_title("Pareto: params vs avg EPE")
    ax[0].grid(True, alpha=0.3)

    # Right: GFLOPs vs avg EPE
    valid_flop = [r for r in valid if r.get("gflops") is not None]
    for r in valid_flop:
        ax[1].scatter(r["gflops"], r["avg_epe"],
                      color=colors.get(r["backbone"], "k"),
                      marker=markers.get(r["head_ch"], "x"),
                      s=120, edgecolors="black", linewidths=0.6)
        ax[1].annotate(r["variant"], (r["gflops"], r["avg_epe"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax[1].set_xlabel("GFLOPs @ 384×768")
    ax[1].set_ylabel("avg EPE (4 datasets)")
    ax[1].set_title("Pareto: GFLOPs vs avg EPE")
    ax[1].grid(True, alpha=0.3)

    fig.suptitle("Phase 8 — EffViTDepthNet efficiency Pareto")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-dir", default="results/phase8_pareto/profile")
    ap.add_argument("--eval-dir", default="results/phase8_pareto/eval")
    ap.add_argument("--out-dir", default="results/phase8_pareto")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for backbone, head_ch in VARIANTS:
        r = load_variant(args.profile_dir, args.eval_dir, backbone, head_ch)
        if r:
            rows.append(r)
        else:
            print(f"[miss] {backbone}_h{head_ch}: profile or eval missing", file=sys.stderr)

    if not rows:
        print("[error] no complete variants found", file=sys.stderr)
        return

    write_markdown(rows, os.path.join(args.out_dir, "summary_table.md"))
    write_csv(rows, os.path.join(args.out_dir, "summary_table.csv"))
    plot_pareto(rows, os.path.join(args.out_dir, "pareto_plot.png"))

    print(f"[done] {len(rows)} variants aggregated → {args.out_dir}/")


if __name__ == "__main__":
    main()
