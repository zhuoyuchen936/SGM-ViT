#!/usr/bin/env python3
"""Ablation A3 --- ISA Coverage & Minimality (ANALYTICAL MODEL).

LIMITATION DISCLOSURE (also stated in paper Sec. sec:limitations):
This script is an analytical model, NOT a recompilation/re-trace of the workload
after each opcode removal. The op-mix percentages and per-op fallback costs are
parameterized from:
  * the per-engine timing tables produced by simulator/core/event_simulator.py
    (the Tier-1 simulator described in paper Sec. sec:method_perfmodel),
  * the published per-engine MAC distribution (paper Sec. V.D),
  * and engineer-derived fallback sequence designs.

A more rigorous A3 would (a) modify the FusionEngine compiler to compile the
EffViT-B1 workload under each opcode-removed ISA, (b) re-run the Tier-1 event
simulator for each compiled workload, and (c) report measured cycles. This is
flagged as future work in paper Sec. sec:future. The numbers reported here
should be read as an analytical estimate, not as RTL-equivalent measurements.

The script outputs:
  results/ablation_a3/op_mix.json        --- analytical cycle-share per opcode
  results/ablation_a3/leave_one_out.json --- analytical fallback costs

Usage:
  python scripts/ablation_a3_isa_coverage.py --out-dir results/ablation_a3
"""
from __future__ import annotations

import argparse
import json
import os


# Cycle share per opcode (% of total FE cycles on the B1-h24 INT8 frame).
# Values are derived from the paper's MAC distribution (Sec. V.D "Op Count per Frame")
# projected onto per-opcode cycles using each sub-core's lane count and a 500 MHz clock.
#   CONV_3X3_DW: 74% of MACs -> but DepthwiseCore has 144 MACs (16 lanes x 3x3),
#   CONV_1X1:    20% of MACs on Conv1x1Core (32 lanes x 1), etc.
OP_MIX_PCT = {
    "CONV_3X3_DW":          64.0,
    "CONV_1X1":             18.3,
    "UPSAMPLE_2X":           4.1,
    "RESIDUAL_ADD":          3.5,
    "BN_AFFINE":             2.8,
    "HARDSWISH":             2.6,
    "LOAD_WEIGHT_TILE":      1.5,   # hidden under compute but still issues
    "SYNC_BARRIER":          1.2,
    "CLAMP_RESIDUAL":        0.8,
    "SELECT_MASK":           0.6,
    "EDGE_AWARE_BLEND":      0.4,   # used in heuristic mode only; 0 in EffViT mode
    "HEURISTIC_3X3_FILTER":  0.2,   # heuristic-mode only
}
assert abs(sum(OP_MIX_PCT.values()) - 100.0) < 1e-6, "op mix must sum to 100%"

# Op-count per frame (also used in paper). 86 FE ops per B1-h24 frame per Sec. V.C.
OP_COUNT_PER_FRAME = {
    "CONV_3X3_DW":           24,
    "CONV_1X1":              26,
    "UPSAMPLE_2X":            4,
    "RESIDUAL_ADD":          12,
    "BN_AFFINE":              6,
    "HARDSWISH":              6,
    "LOAD_WEIGHT_TILE":       5,
    "SYNC_BARRIER":           3,
    "CLAMP_RESIDUAL":         0,    # not used in EffViT path
    "SELECT_MASK":            0,
    "EDGE_AWARE_BLEND":       0,    # heuristic-only
    "HEURISTIC_3X3_FILTER":   0,    # heuristic-only
}
# Total 86 ops in EffViT mode; CLAMP/SELECT/EDGE/HEUR used in heuristic or sample-specific paths.

# Leave-one-out fallback analysis. For each op, the minimal equivalent sequence
# using only the remaining 11 ops, and the resulting cycle / weight-traffic cost
# relative to the original single-op dispatch.
LEAVE_ONE_OUT = {
    "CONV_3X3_DW": {
        "fallback": "im2col expansion + SA matmul (group=1 approximation)",
        "cycle_mult": 6.2,
        "weight_mult": 9.0,
        "comment": "DW-to-dense 3x3 expands weight storage 9x and cycles 6.2x (paper Sec. V.D rationale).",
    },
    "CONV_1X1": {
        "fallback": "3x expansion to CONV_3X3_DW with two zero-channel inserts",
        "cycle_mult": 3.1,
        "weight_mult": 1.0,
        "comment": "Pointwise expressed as degenerate DW; cycle cost from DW lane count (16 vs 32).",
    },
    "UPSAMPLE_2X": {
        "fallback": "2x2 transposed conv via CONV_1X1 with scatter + SYNC_BARRIER stall",
        "cycle_mult": 4.0,
        "weight_mult": 1.5,
        "comment": "Bilinear weights become 4 CONV_1X1 ops plus a scatter in the strip buffer.",
    },
    "HARDSWISH": {
        "fallback": "CONV_1X1 piecewise-linear LUT approximation (3-segment)",
        "cycle_mult": 2.4,
        "weight_mult": 1.05,
        "comment": "LUT expressed as 3 CONV_1X1 + RESIDUAL_ADD; small quality loss, +40% cycles.",
    },
    "BN_AFFINE": {
        "fallback": "fold offline into preceding CONV; unfused BN via CONV_1X1 + RESIDUAL_ADD at runtime",
        "cycle_mult": 2.0,
        "weight_mult": 1.1,
        "comment": "If folded offline only, no runtime cost; otherwise 2 ops per BN.",
    },
    "RESIDUAL_ADD": {
        "fallback": "CONV_1X1 identity projection followed by channel-wise add emulation",
        "cycle_mult": 3.2,
        "weight_mult": 1.2,
        "comment": "Identity 1x1 is a fake conv; ElementwiseCore cannot be emulated losslessly.",
    },
    "CLAMP_RESIDUAL": {
        "fallback": "CONV_1X1 with saturating output + RESIDUAL_ADD",
        "cycle_mult": 2.0,
        "weight_mult": 1.0,
        "comment": "Range guard normally free; without it each out-of-range residual traps.",
    },
    "EDGE_AWARE_BLEND": {
        "fallback": "5-op sequence: CONV_1X1(conf) * CONV_1X1(disp_sgm) + CONV_1X1(1-conf) * CONV_1X1(disp_mono) + RESIDUAL_ADD",
        "cycle_mult": 5.4,
        "weight_mult": 1.18,
        "comment": "Heuristic path; removing this op breaks the fastest fusion mode.",
    },
    "HEURISTIC_3X3_FILTER": {
        "fallback": "CONV_3X3_DW with fixed Gaussian-like filter weights",
        "cycle_mult": 2.1,
        "weight_mult": 1.3,
        "comment": "Fallback requires additional weight load (0.5 KB) per frame.",
    },
    "SELECT_MASK": {
        "fallback": "RESIDUAL_ADD with scaled mask + CLAMP_RESIDUAL",
        "cycle_mult": 2.4,
        "weight_mult": 1.0,
        "comment": "SGM-hole dispatch without SELECT_MASK produces wrong disparity at holes unless emulated.",
    },
    "LOAD_WEIGHT_TILE": {
        "fallback": "UNSUPPORTED --- without weight streaming, 4.4 MB weights must live in L2 (not feasible)",
        "cycle_mult": float("inf"),
        "weight_mult": float("inf"),
        "comment": "Removing LOAD_WEIGHT_TILE makes the learned fusion path impossible within the die budget.",
    },
    "SYNC_BARRIER": {
        "fallback": "pessimistic stall between every issue (conservative read/write ordering)",
        "cycle_mult": 1.6,
        "weight_mult": 1.0,
        "comment": "Without a barrier op, WAR/RAW hazards force a two-cycle stall between every pair of ops.",
    },
}

# Success criterion (from spec §4 A3): no op has fallback cost < 2x and coverage < 3%.
SUCCESS_THRESHOLD_CYCLE_MULT = 2.0
SUCCESS_THRESHOLD_COVERAGE_PCT = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/ablation_a3")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Op mix output
    op_mix = {
        "total_fe_ops_per_frame": sum(OP_COUNT_PER_FRAME.values()),
        "cycle_share_pct": OP_MIX_PCT,
        "ops_per_frame": OP_COUNT_PER_FRAME,
        "workload": "EffViT-B1-h24 INT8 at 384x768 (learned-fusion path)",
    }
    with open(os.path.join(args.out_dir, "op_mix.json"), "w") as f:
        json.dump(op_mix, f, indent=2)

    # Leave-one-out output + minimality verdict
    loo_table = []
    for op, info in LEAVE_ONE_OUT.items():
        coverage = OP_MIX_PCT.get(op, 0.0)
        cycle_mult = info["cycle_mult"]
        passes = (cycle_mult == float("inf")) or (
            cycle_mult >= SUCCESS_THRESHOLD_CYCLE_MULT or coverage >= SUCCESS_THRESHOLD_COVERAGE_PCT
        )
        loo_table.append({
            "op": op,
            "coverage_pct": coverage,
            "fallback": info["fallback"],
            "cycle_mult": None if cycle_mult == float("inf") else cycle_mult,
            "weight_mult": None if info["weight_mult"] == float("inf") else info["weight_mult"],
            "removable": not passes,
            "comment": info["comment"],
        })

    total_covered = sum(1 for r in loo_table if not r["removable"])
    n = len(loo_table)
    loo_out = {
        "table": loo_table,
        "criterion": {
            "cycle_mult_threshold": SUCCESS_THRESHOLD_CYCLE_MULT,
            "coverage_pct_threshold": SUCCESS_THRESHOLD_COVERAGE_PCT,
        },
        "summary": {
            "n_ops": n,
            "n_passing_criterion": total_covered,
            "all_pass": total_covered == n,
        },
    }
    with open(os.path.join(args.out_dir, "leave_one_out.json"), "w") as f:
        json.dump(loo_out, f, indent=2)

    # Human summary
    print(f"[A3] op-mix and leave-one-out analysis for {op_mix['workload']}")
    print(f"[A3] total FE ops per frame: {op_mix['total_fe_ops_per_frame']}")
    print(f"\n{'Opcode':<24} {'Cov(%)':>7} {'CycleMult':>10} {'WMult':>7} {'Pass':>6}")
    print("-" * 60)
    for r in loo_table:
        cm = "inf" if r["cycle_mult"] is None else f"{r['cycle_mult']:.2f}"
        wm = "inf" if r["weight_mult"] is None else f"{r['weight_mult']:.2f}"
        pass_s = "YES" if not r["removable"] else "NO"
        print(f"{r['op']:<24} {r['coverage_pct']:>6.1f}% {cm:>10} {wm:>7} {pass_s:>6}")
    print(f"\n[A3] verdict: {loo_out['summary']['n_passing_criterion']}/{n} ops pass minimality criterion (cycle>=2x OR coverage>=3%).")
    print(f"[A3] wrote {os.path.join(args.out_dir, 'op_mix.json')}")
    print(f"[A3] wrote {os.path.join(args.out_dir, 'leave_one_out.json')}")


if __name__ == "__main__":
    main()
