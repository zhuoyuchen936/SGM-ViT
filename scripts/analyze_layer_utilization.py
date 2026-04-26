#!/usr/bin/env python3
"""Per-layer systolic-array utilization & compute breakdown for the end-to-end pipeline.

Utilization model (32x32 = 1024 PEs, output-stationary matmul):
  * SA ops: ideal_cycles = M*K*N/1024, actual_cycles = ceil(M/32)*ceil(N/32)*K
    SA_util = ideal/actual = (M*N) / (ceil(M/32)*32 * ceil(N/32)*32)
    Interpretation: fraction of the 32x32 PE fabric that holds useful work each cycle.
  * FE Conv1x1Core (32 lanes): util = (pixels*out_channels) / (ceil(pixels*out_channels/32)*32)
  * FE DepthwiseCore (16 lanes × 9 MAC/pixel): util = (H*W*C*9) / (ceil(H*W*C*9 / (16*16))*16*16)
  * FE ElementwiseCore (32 lanes): util = pixels / (ceil(pixels/32)*32)

Also reports arithmetic intensity (FLOPs per byte of memory traffic) and the roofline
regime (compute-bound vs memory-bound) assuming peak_GOPS=1024 GOPS @ 500 MHz
and peak_BW=3 GB/s (single-channel LPDDR4 sustained).

Usage:
    python scripts/analyze_layer_utilization.py --config effvit_b1_h24
Outputs:
    results/phase10/layer_util_{config}.csv
    results/phase10/layer_util_{config}.md
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulator.core.event_simulator import SimConfig
from simulator.core.pipeline_model import build_pipeline_workload


# SA geometry / platform constants.
SA_ROWS = SA_COLS = 32
PE_COUNT = SA_ROWS * SA_COLS
CLOCK_MHZ = 500
PEAK_MACS_PER_CYC = PE_COUNT              # 1024
PEAK_GFLOPS = 2 * PEAK_MACS_PER_CYC * CLOCK_MHZ / 1e3  # 1024 * 2 * 500MHz = 1024 GFLOPs
PEAK_DRAM_BW_GBS = 3.0                     # LPDDR4 single-channel sustained
FU_CONV1X1_LANES = 32
FU_DW_LANES = 16                          # × 9 MAC per-pixel inside each lane
FU_ELEM_LANES = 32


def sa_tiled_cycles(M: int, K: int, N: int) -> tuple[int, int, float]:
    """Return (ideal_cyc, actual_cyc, util) for an SA matmul of shape MxKxN.

    actual_cyc assumes output-stationary tiling with 32x32 tiles, no K-tiling.
    """
    ideal = max(1, (M * K * N + PE_COUNT - 1) // PE_COUNT)
    n_m_tiles = (M + SA_ROWS - 1) // SA_ROWS
    n_n_tiles = (N + SA_COLS - 1) // SA_COLS
    actual = n_m_tiles * n_n_tiles * K
    # Spatial occupancy: how full is each tile on average
    occupancy_M = M / (n_m_tiles * SA_ROWS)
    occupancy_N = N / (n_n_tiles * SA_COLS)
    util = occupancy_M * occupancy_N
    return ideal, max(1, actual), util


def fe_dw_cycles(H: int, W: int, C: int) -> tuple[int, int, float]:
    """DepthwiseCore: 16 lanes × 9 MAC per pixel for 3x3 DW. Process 16x16 pixel tiles."""
    # Each cycle issues 16 channels × 16 pixel-positions × 9 MAC-accum
    # Approx: total_macs / (16 * 16 * 9) where each cycle does 16*16 MACs in 1 lane slice
    # Simpler: cycles = ceil(H*W*C * 9 / (16 * 16))
    total_macs = H * W * C * 9
    actual = (total_macs + FU_DW_LANES * FU_DW_LANES - 1) // (FU_DW_LANES * FU_DW_LANES)
    ideal = max(1, total_macs // PE_COUNT)  # if it could use the SA
    # Fabric util relative to FU's own capacity
    fu_peak_per_cyc = FU_DW_LANES * FU_DW_LANES  # 256 MACs/cycle at 100%
    util_fu = total_macs / (actual * fu_peak_per_cyc)
    return ideal, actual, util_fu


def fe_conv1x1_cycles(pixels: int, out_ch: int) -> tuple[int, int, float]:
    total_ops = pixels * out_ch
    actual = (total_ops + FU_CONV1X1_LANES - 1) // FU_CONV1X1_LANES
    ideal = max(1, total_ops // PE_COUNT)
    util_fu = total_ops / (actual * FU_CONV1X1_LANES)
    return ideal, actual, util_fu


def fe_elem_cycles(pixels: int, ops_per_pixel: int = 1) -> tuple[int, int, float]:
    total_ops = pixels * ops_per_pixel
    actual = (total_ops + FU_ELEM_LANES - 1) // FU_ELEM_LANES
    ideal = max(1, total_ops // PE_COUNT)
    util_fu = total_ops / (actual * FU_ELEM_LANES)
    return ideal, actual, util_fu


STAGE_NAMES = {
    0: "DMA / Input",
    1: "SGM compute",
    2: "DA2 ViT-S encoder",
    3: "EffViT backbone / decoder",
    4: "Mono-SGM alignment",
    5: "Fusion / FPN head",
}


def analyze_op(op) -> dict:
    md = op.metadata
    stage = md.get("stage", -1)
    row = {
        "op_id": op.id,
        "name": op.name[:50],
        "stage": stage,
        "stage_name": STAGE_NAMES.get(stage, f"stage{stage}"),
        "engine": op.engine,
        "op_type": md.get("sa_op_type") or md.get("fu_op_type") or "-",
        "flops_M": round(op.flops / 1e6, 3),
        "weight_KB": round(op.weight_bytes / 1024, 2),
        "input_KB": round(md.get("input_bytes", 0) / 1024, 2),
        "output_KB": round(md.get("output_bytes", 0) / 1024, 2),
    }
    M = md.get("M"); K = md.get("K"); N = md.get("N")
    H = md.get("H"); W = md.get("W"); C_in = md.get("in_channels"); C_out = md.get("out_channels")
    if op.engine == "systolic_array" and M and K and N:
        row["shape"] = f"M={M} K={K} N={N}"
        ideal, actual, util = sa_tiled_cycles(M, K, N)
        row["ideal_cyc"] = ideal
        row["actual_cyc"] = actual
        row["util_pct"] = round(util * 100, 1)
        row["fabric"] = "SA 32x32"
    elif op.engine == "fu" and md.get("fu_op_type") == "CONV_3X3_DW" and H and W:
        row["shape"] = f"H={H} W={W} C={C_in}"
        ideal, actual, util = fe_dw_cycles(H, W, C_in or 1)
        row["ideal_cyc"] = ideal
        row["actual_cyc"] = actual
        row["util_pct"] = round(util * 100, 1)
        row["fabric"] = "FE DW 16L"
    elif op.engine == "fu" and md.get("fu_op_type") == "CONV_1X1" and H and W:
        row["shape"] = f"H={H} W={W} Cin={C_in} Cout={C_out}"
        pixels = H * W
        ideal, actual, util = fe_conv1x1_cycles(pixels, C_out or 1)
        row["ideal_cyc"] = ideal
        row["actual_cyc"] = actual
        row["util_pct"] = round(util * 100, 1)
        row["fabric"] = "FE 1x1 32L"
    elif op.engine == "fu":
        pixels = md.get("total_pixels", 1)
        ops_per_pixel = md.get("ops_per_pixel", 1)
        ideal, actual, util = fe_elem_cycles(pixels, ops_per_pixel)
        row["shape"] = f"pix={pixels}"
        row["ideal_cyc"] = ideal
        row["actual_cyc"] = actual
        row["util_pct"] = round(util * 100, 1)
        row["fabric"] = f"FE Elem 32L"
    else:
        row["shape"] = "-"
        row["ideal_cyc"] = 1
        row["actual_cyc"] = 1
        row["util_pct"] = 0.0
        row["fabric"] = "non-compute"

    # Arithmetic intensity and roofline
    memory_traffic = (md.get("input_bytes", 0) + md.get("output_bytes", 0) + op.weight_bytes)
    if memory_traffic > 0:
        ai = op.flops / memory_traffic
        row["ai_flops_per_B"] = round(ai, 2)
        # Roofline ridge: peak_flops / peak_bw = 1024 GFLOPs / 3 GB/s = 341
        ridge = PEAK_GFLOPS / PEAK_DRAM_BW_GBS  # B/flop ratio
        row["bound"] = "compute" if ai > ridge else "memory"
    else:
        row["ai_flops_per_B"] = 0.0
        row["bound"] = "-"
    # Latency @ 500 MHz
    row["lat_us"] = round(row["actual_cyc"] / (CLOCK_MHZ), 2)
    return row


def write_markdown(rows, out_path, config_name):
    # Aggregate per-stage totals
    stage_tot = defaultdict(lambda: {"ops": 0, "flops_M": 0.0, "weight_KB": 0.0,
                                     "actual_cyc": 0, "ideal_cyc": 0})
    for r in rows:
        s = r["stage"]
        stage_tot[s]["ops"] += 1
        stage_tot[s]["flops_M"] += r["flops_M"]
        stage_tot[s]["weight_KB"] += r["weight_KB"]
        stage_tot[s]["actual_cyc"] += r["actual_cyc"]
        stage_tot[s]["ideal_cyc"] += r["ideal_cyc"]

    with open(out_path, "w") as f:
        f.write(f"# Per-Layer Compute & Utilization — `{config_name}`\n\n")
        f.write(f"Platform: 32×32 SA + FE sub-cores @ {CLOCK_MHZ} MHz (peak {PEAK_GFLOPS:.0f} GFLOPs)\n")
        f.write(f"Memory: LPDDR4 3 GB/s sustained. Roofline ridge = {PEAK_GFLOPS/PEAK_DRAM_BW_GBS:.1f} FLOPs/B\n\n")

        f.write("## Per-stage summary\n\n")
        f.write("| Stage | Engine(s) | #ops | Σ FLOPs (M) | Σ weights (KB) | Σ ideal_cyc | Σ actual_cyc | avg util | lat @500MHz |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for s in sorted(stage_tot):
            st = stage_tot[s]
            util = st["ideal_cyc"] / max(st["actual_cyc"], 1) * 100
            lat_us = st["actual_cyc"] / CLOCK_MHZ
            engines = sorted(set(r["fabric"] for r in rows if r["stage"] == s))
            f.write(f"| {s}: {STAGE_NAMES.get(s, f'stage{s}')} | {', '.join(engines)} | "
                    f"{st['ops']} | {st['flops_M']:.1f} | {st['weight_KB']:.1f} | "
                    f"{st['ideal_cyc']:,} | {st['actual_cyc']:,} | {util:.1f}% | {lat_us/1000:.2f} ms |\n")

        f.write("\n## Per-layer breakdown\n\n")
        f.write("| op_id | name | stage | fabric | shape | FLOPs (M) | wgt (KB) | in+out (KB) | ideal cyc | actual cyc | util % | AI (F/B) | bound | lat (µs) |\n")
        f.write("|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|\n")
        for r in rows:
            f.write(f"| {r['op_id']} | {r['name']} | {r['stage']} | {r['fabric']} | {r['shape']} | "
                    f"{r['flops_M']:.2f} | {r['weight_KB']:.1f} | "
                    f"{r['input_KB']+r['output_KB']:.1f} | "
                    f"{r['ideal_cyc']:,} | {r['actual_cyc']:,} | {r['util_pct']:.1f} | "
                    f"{r['ai_flops_per_B']:.1f} | {r['bound']} | {r['lat_us']:.1f} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="effvit_b1_h24",
                    choices=["baseline", "effvit_b1_h24", "effvit_b0_h24"])
    ap.add_argument("--img-h", type=int, default=384)
    ap.add_argument("--img-w", type=int, default=768)
    ap.add_argument("--out-dir", default="results/phase10")
    ap.add_argument("--top-by-flops", type=int, default=0,
                    help="If >0, only emit the top-N ops by FLOPs in the detailed table")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sim_cfg = SimConfig(clock_freq_mhz=CLOCK_MHZ, process_node_nm=28,
                        keep_ratio=1.0, stage_policy="coarse_only")
    dag = build_pipeline_workload(args.config, sim_cfg, img_h=args.img_h, img_w=args.img_w)
    print(f"[dag] {args.config}  total ops = {len(dag.operations)}")

    rows = [analyze_op(op) for op in dag.operations.values()]
    rows.sort(key=lambda r: (r["stage"], r["op_id"]))

    if args.top_by_flops > 0:
        # Keep per-stage summary based on all ops, but trim the long per-layer table
        rows_full = list(rows)
        top = sorted(rows_full, key=lambda r: -r["flops_M"])[:args.top_by_flops]
        top_ids = {r["op_id"] for r in top}
        rows_trimmed = [r for r in rows_full if r["op_id"] in top_ids]
        rows_trimmed.sort(key=lambda r: (r["stage"], r["op_id"]))
        out_md = os.path.join(args.out_dir, f"layer_util_{args.config}_top{args.top_by_flops}.md")
        out_csv = os.path.join(args.out_dir, f"layer_util_{args.config}_top{args.top_by_flops}.csv")
        write_markdown(rows_trimmed, out_md, args.config + f" (top-{args.top_by_flops})")
    else:
        out_md = os.path.join(args.out_dir, f"layer_util_{args.config}.md")
        out_csv = os.path.join(args.out_dir, f"layer_util_{args.config}.csv")
        write_markdown(rows, out_md, args.config)

    keys = ["op_id", "name", "stage", "stage_name", "engine", "op_type", "fabric", "shape",
            "flops_M", "weight_KB", "input_KB", "output_KB",
            "ideal_cyc", "actual_cyc", "util_pct", "ai_flops_per_B", "bound", "lat_us"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    # Print quick summary to stdout
    total_flops = sum(r["flops_M"] for r in rows) / 1e3  # → GFLOPs
    total_actual = sum(r["actual_cyc"] for r in rows)
    total_ideal = sum(r["ideal_cyc"] for r in rows)
    overall_util = total_ideal / max(total_actual, 1) * 100
    print(f"[result] {args.config}: {len(rows)} ops, {total_flops:.1f} GFLOPs, "
          f"{total_actual:,} actual cyc, {total_ideal:,} ideal cyc, overall util = {overall_util:.1f}%")
    print(f"[saved] {out_md}")
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
