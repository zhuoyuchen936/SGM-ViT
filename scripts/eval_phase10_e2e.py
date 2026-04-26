#!/usr/bin/env python3
"""Phase 10 end-to-end evaluation — Unified Fusion Engine v2 + WeightStreamer.

Builds a synthetic op-list for two workloads and runs them through the new
hardware modules' cycle estimators:

  1. **heuristic** — the original FU edge-aware-residual fusion pipeline
     (KITTI-sized 384x768 image)
  2. **effvit_b1_int8** — EffViT-B1-h24 INT8 QAT model (Phase 9 sweet-spot):
     full backbone + FPN decoder mapped to UnifiedSA (matmul/conv) +
     FusionEngineV2 (FPN, BN, Hardswish, residual) + WeightStreamer
     for tile-level weight streaming.

Reports per workload:
  - total cycles
  - per-op breakdown
  - FPS @ 500 MHz
  - area / power
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from hardware.fusion_engine import FusionEngineV2, FusionEngineConfig
from hardware.architecture.weight_streamer import WeightStreamer, WeightStreamerConfig


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------

def heuristic_workload(H: int = 384, W: int = 768) -> list[dict]:
    """Original FU edge-aware-residual fusion (1 op covers full pipeline)."""
    return [{"id": "heur:edge_aware_blend", "fu_op_type": "EDGE_AWARE_BLEND", "H": H, "W": W}]


def effvit_b1_int8_workload(H: int = 384, W: int = 768) -> list[dict]:
    """EffViT-B1-h24 INT8 (sweet-spot) op list.

    Stages and per-stage feature shapes (input H×W → 4 strides):
      stem(7→16, /2):        H/2 ×W/2  ×16  (1 conv1x1 stub representing stem)
      stage1 (16→32, /4):    H/4 ×W/4  ×32  (2 MBConv DSConv blocks)
      stage2 (32→64, /8):    H/8 ×W/8  ×64  (3 MBConv blocks)
      stage3 (64→128, /16):  H/16×W/16 ×128 (3 MBConv + 3 EffViTBlocks; LiteMLA → SA, MBConv parts → fusion)
      stage4 (128→256, /32): H/32×W/32 ×256 (4 EffViTBlocks; same)

    FPN head (head_ch=24): 4 _UpBlock 1×1 + bilinear + 1×1 lateral + 3×3 fuse + BN+Hardswish
    Final conv 24→1 + clamp + residual_add to mono.

    For paper modelling: only the **non-systolic-array work** runs on FusionEngineV2.
    The 1×1 expand/project convs and the matmul-style attention are routed to SA.
    Here we list ONLY the fusion-engine-bound ops; the SA cycles are added at the end
    as a pre-aggregated estimate (assumed to already exist in baseline simulator).
    """
    ops: list[dict] = []
    sH, sW = H // 4, W // 4  # stage1 res
    # ------ Backbone Hardswish + BN + 3×3 DW (fusion-engine-bound parts) ------
    # Stage 1: 16→32, 2 MBConv blocks. Each block has:
    #   1x1 expand (SA), 3x3 DW (fusion), 1x1 project (SA), Hardswish/BN (fusion), residual (fusion).
    # We include only the fusion-bound parts.
    for i in range(2):
        ops += [
            {"id": f"s1.b{i}.dw", "fu_op_type": "CONV_3X3_DW", "in_channels": 32, "H": sH, "W": sW},
            {"id": f"s1.b{i}.bn1", "fu_op_type": "BN_AFFINE", "in_channels": 32, "H": sH, "W": sW},
            {"id": f"s1.b{i}.act", "fu_op_type": "HARDSWISH", "in_channels": 32, "H": sH, "W": sW},
            {"id": f"s1.b{i}.res", "fu_op_type": "RESIDUAL_ADD", "in_channels": 32, "H": sH, "W": sW},
        ]
    # Stage 2: 32→64, 3 MBConv blocks at H/8.
    sH, sW = H // 8, W // 8
    for i in range(3):
        ops += [
            {"id": f"s2.b{i}.dw", "fu_op_type": "CONV_3X3_DW", "in_channels": 64, "H": sH, "W": sW},
            {"id": f"s2.b{i}.bn1", "fu_op_type": "BN_AFFINE", "in_channels": 64, "H": sH, "W": sW},
            {"id": f"s2.b{i}.act", "fu_op_type": "HARDSWISH", "in_channels": 64, "H": sH, "W": sW},
            {"id": f"s2.b{i}.res", "fu_op_type": "RESIDUAL_ADD", "in_channels": 64, "H": sH, "W": sW},
        ]
    # Stage 3: 64→128, 1 MBConv + 3 EffViTBlocks at H/16. Only MBConv parts hit fusion engine.
    sH, sW = H // 16, W // 16
    ops += [
        {"id": "s3.mbconv.dw",  "fu_op_type": "CONV_3X3_DW", "in_channels": 128, "H": sH, "W": sW},
        {"id": "s3.mbconv.bn",  "fu_op_type": "BN_AFFINE",    "in_channels": 128, "H": sH, "W": sW},
        {"id": "s3.mbconv.act", "fu_op_type": "HARDSWISH",    "in_channels": 128, "H": sH, "W": sW},
    ]
    for i in range(3):
        # EffViTBlock LiteMLA + GLUMBConv: GLUMBConv has DW + BN + Hardswish parts on fusion.
        ops += [
            {"id": f"s3.evb{i}.dw",  "fu_op_type": "CONV_3X3_DW", "in_channels": 128, "H": sH, "W": sW},
            {"id": f"s3.evb{i}.bn",  "fu_op_type": "BN_AFFINE",    "in_channels": 128, "H": sH, "W": sW},
            {"id": f"s3.evb{i}.act", "fu_op_type": "HARDSWISH",    "in_channels": 128, "H": sH, "W": sW},
        ]
    # Stage 4: 128→256, 1 MBConv + 4 EffViTBlocks at H/32.
    sH, sW = H // 32, W // 32
    ops += [
        {"id": "s4.mbconv.dw",  "fu_op_type": "CONV_3X3_DW", "in_channels": 256, "H": sH, "W": sW},
        {"id": "s4.mbconv.bn",  "fu_op_type": "BN_AFFINE",    "in_channels": 256, "H": sH, "W": sW},
        {"id": "s4.mbconv.act", "fu_op_type": "HARDSWISH",    "in_channels": 256, "H": sH, "W": sW},
    ]
    for i in range(4):
        ops += [
            {"id": f"s4.evb{i}.dw",  "fu_op_type": "CONV_3X3_DW", "in_channels": 256, "H": sH, "W": sW},
            {"id": f"s4.evb{i}.bn",  "fu_op_type": "BN_AFFINE",    "in_channels": 256, "H": sH, "W": sW},
            {"id": f"s4.evb{i}.act", "fu_op_type": "HARDSWISH",    "in_channels": 256, "H": sH, "W": sW},
        ]
    # ------ FPN decoder (4 _UpBlock + final head) ------
    # _UpBlock i: project(1x1) + upsample + lateral(1x1) + add + fuse(3x3 SA) + 2× (BN+HS).
    # Resolutions: stage4→stage3, stage3→stage2, stage2→stage1, stage1→stage0.
    fpn_stages = [
        # (in_top, lateral_in, lateral_H, lateral_W, head_ch=24)
        (256, 128, H // 16, W // 16),   # up4_3
        (24,  64,  H // 8,  W // 8),    # up3_2
        (24,  32,  H // 4,  W // 4),    # up2_1
        (24,  16,  H // 2,  W // 2),    # up1_0
    ]
    for i, (top, lat, fH, fW) in enumerate(fpn_stages):
        ops += [
            {"id": f"fpn{i}.proj_top", "fu_op_type": "CONV_1X1", "in_channels": top, "out_channels": 24, "H": fH, "W": fW},
            {"id": f"fpn{i}.upsample", "fu_op_type": "UPSAMPLE_2X", "H": fH, "W": fW, "in_channels": 24},
            {"id": f"fpn{i}.lateral",  "fu_op_type": "CONV_1X1", "in_channels": lat, "out_channels": 24, "H": fH, "W": fW},
            {"id": f"fpn{i}.add",      "fu_op_type": "RESIDUAL_ADD", "in_channels": 24, "H": fH, "W": fW},
            {"id": f"fpn{i}.bn1",      "fu_op_type": "BN_AFFINE", "in_channels": 24, "H": fH, "W": fW},
            {"id": f"fpn{i}.act1",     "fu_op_type": "HARDSWISH", "in_channels": 24, "H": fH, "W": fW},
            {"id": f"fpn{i}.bn2",      "fu_op_type": "BN_AFFINE", "in_channels": 24, "H": fH, "W": fW},
            {"id": f"fpn{i}.act2",     "fu_op_type": "HARDSWISH", "in_channels": 24, "H": fH, "W": fW},
        ]
    # Final upsample ×2 + 3×3 fuse + head conv (24→12) + final residual conv.
    ops += [
        {"id": "head.upsample", "fu_op_type": "UPSAMPLE_2X", "H": H // 2, "W": W // 2, "in_channels": 24},
        {"id": "head.conv1x1_proj", "fu_op_type": "CONV_1X1", "in_channels": 24, "out_channels": 12, "H": H, "W": W},
        {"id": "head.bn",  "fu_op_type": "BN_AFFINE", "in_channels": 12, "H": H, "W": W},
        {"id": "head.act", "fu_op_type": "HARDSWISH", "in_channels": 12, "H": H, "W": W},
        {"id": "head.conv1x1_residual", "fu_op_type": "CONV_1X1", "in_channels": 12, "out_channels": 1, "H": H, "W": W},
        {"id": "head.clamp",   "fu_op_type": "CLAMP_RESIDUAL", "in_channels": 1, "H": H, "W": W},
        {"id": "head.add_mono","fu_op_type": "RESIDUAL_ADD",   "in_channels": 1, "H": H, "W": W},
    ]
    return ops


# ---------------------------------------------------------------------------
# Weight streaming schedule
# ---------------------------------------------------------------------------

def stream_schedule_effvit_b1_int8(streamer: WeightStreamer) -> tuple[int, list[int]]:
    """Schedule weight tiles for EffViT-B1 INT8 (4.85M params ≈ 4.85 MB).

    Stage-sequential: stage 1: 16→32 (~10 KB), 2: 32→64 (~40 KB),
    3: 64→128 (~600 KB w/ 3 EffViTBlocks of LiteMLA + GLUMBConv),
    4: 128→256 (~3.3 MB w/ 4 LiteMLA + GLUMBConv),
    FPN head: ~50 KB (head_ch=24 × few 1x1 lateral/project).
    LiteMLA stays FP32, but for paper streaming model we treat all as INT8 weights.
    """
    # Per-stage tile sizes (bytes), conservatively.
    stage_tiles = {
        "stage1":      10 * 1024,
        "stage2":      40 * 1024,
        "stage3":     600 * 1024,
        "stage4":    3300 * 1024,
        "fpn_head":    50 * 1024,
    }
    total_cycles = 0
    per_stage_cycles = []
    cycle = 0
    for name, tile_bytes in stage_tiles.items():
        # Split each stage into ≤32 KB tiles (double-buffer size).
        n_tiles = max(1, (tile_bytes + 32 * 1024 - 1) // (32 * 1024))
        per_tile = max(1, tile_bytes // n_tiles)
        for _ in range(n_tiles):
            ready, _ = streamer.fetch_tile(per_tile, cycle)
            cycle = ready
        per_stage_cycles.append((name, cycle - total_cycles, n_tiles))
        total_cycles = cycle
    return total_cycles, per_stage_cycles


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_workload(name: str, ops: list[dict], engine: FusionEngineV2) -> dict:
    per_op = []
    total = 0
    for op in ops:
        c = engine.estimate_op_cycles(op)
        per_op.append({"id": op["id"], "fu_op_type": op["fu_op_type"], "cycles": c})
        total += c
    return {"name": name, "total_cycles": total, "n_ops": len(ops), "per_op": per_op}


def summarize(name: str, fusion_cycles: int, sa_cycles: int, stream_cycles: int, freq_mhz: int = 500) -> dict:
    """Combine per-engine cycles into a frame-level estimate.

    Assumes weight streaming overlaps with compute (steady-state pipelined),
    so we take max(fusion+sa, stream) per frame as a first-order upper bound.
    """
    pipelined = max(fusion_cycles + sa_cycles, stream_cycles)
    serial = fusion_cycles + sa_cycles + stream_cycles
    fps_pipelined = freq_mhz * 1e6 / max(pipelined, 1)
    fps_serial = freq_mhz * 1e6 / max(serial, 1)
    return {
        "name": name,
        "fusion_cycles": fusion_cycles,
        "sa_cycles": sa_cycles,
        "stream_cycles": stream_cycles,
        "pipelined_cycles": pipelined,
        "serial_cycles": serial,
        "fps_pipelined_500MHz": fps_pipelined,
        "fps_serial_500MHz": fps_serial,
        "ms_per_frame_pipelined": 1000 / max(fps_pipelined, 1e-6),
        "ms_per_frame_serial": 1000 / max(fps_serial, 1e-6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=384)
    ap.add_argument("--W", type=int, default=768)
    ap.add_argument("--freq-mhz", type=int, default=500)
    ap.add_argument("--node-nm", type=int, default=28)
    ap.add_argument("--out-json", default="results/phase10_e2e/summary.json")
    # Manual SA-side cycle hint per workload (to be replaced by full simulator integration).
    ap.add_argument("--sa-cycles-effvit", type=int, default=2_500_000,
                    help="Estimated SA cycles for EffViT-B1 backbone (matmul/conv)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    engine = FusionEngineV2()
    streamer = WeightStreamer()

    # ------ Workload 1: heuristic ------
    h_ops = heuristic_workload(args.H, args.W)
    h_run = run_workload("heuristic", h_ops, engine)

    # ------ Workload 2: EffViT-B1 INT8 ------
    e_ops = effvit_b1_int8_workload(args.H, args.W)
    e_run = run_workload("effvit_b1_int8", e_ops, engine)

    # Streaming schedule for EffViT (heuristic has no weights → 0).
    stream_cycles_effvit, per_stage = stream_schedule_effvit_b1_int8(streamer)

    # ------ Frame-level summaries ------
    h_summary = summarize("heuristic", h_run["total_cycles"], 0, 0, args.freq_mhz)
    e_summary = summarize("effvit_b1_int8", e_run["total_cycles"], args.sa_cycles_effvit,
                          stream_cycles_effvit, args.freq_mhz)

    # ------ Area / power ------
    engine_area = engine.estimate_area_mm2(args.node_nm)
    engine_power = engine.estimate_power_mw(args.node_nm, args.freq_mhz, utilization=0.30)
    streamer_area = streamer.estimate_area_mm2(args.node_nm)
    streamer_power = streamer.estimate_power_mw(args.node_nm, args.freq_mhz, utilization=0.40)

    # ------ Output ------
    out = {
        "config": {"H": args.H, "W": args.W, "freq_mhz": args.freq_mhz, "node_nm": args.node_nm},
        "fusion_engine": {
            "describe": engine.describe(),
            "area_mm2": engine_area,
            "power_mw": engine_power,
        },
        "weight_streamer": {
            "describe": streamer.describe(),
            "area_mm2": streamer_area,
            "power_mw": streamer_power,
            "stream_cycles_effvit": stream_cycles_effvit,
            "per_stage_cycles": per_stage,
        },
        "workloads": {
            "heuristic": {**h_run, **h_summary},
            "effvit_b1_int8": {**e_run, **e_summary},
        },
        "phase10_summary": {
            "fusion_engine_total_mm2": engine_area["total_mm2"],
            "weight_streamer_total_mm2": streamer_area["total_mm2"],
            "phase10_added_area_mm2": engine_area["total_mm2"] + streamer_area["total_mm2"],
            "fps_heuristic": h_summary["fps_pipelined_500MHz"],
            "fps_effvit_int8": e_summary["fps_pipelined_500MHz"],
            "ms_heuristic": h_summary["ms_per_frame_pipelined"],
            "ms_effvit_int8": e_summary["ms_per_frame_pipelined"],
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    # Console summary.
    print("=" * 70)
    print(f"Phase 10 E2E Summary  (image {args.H}x{args.W}, {args.freq_mhz} MHz, {args.node_nm} nm)")
    print("=" * 70)
    print(f"\nFusionEngineV2 area     : {engine_area['total_mm2']:.4f} mm^2")
    print(f"  - elementwise          : {engine_area['elementwise_mm2']:.4f}")
    print(f"  - conv1x1              : {engine_area['conv1x1_mm2']:.4f}")
    print(f"  - depthwise (NEW)      : {engine_area['depthwise_mm2']:.4f}")
    print(f"  - upsampler/control/L1 : {engine_area['upsampler_mm2']+engine_area['control_mm2']+engine_area['l1_sram_mm2']+engine_area['bilateral_lut_mm2']:.4f}")
    print(f"FusionEngineV2 power    : {engine_power['total_mw']:.2f} mW @ 30% util\n")

    print(f"WeightStreamer area     : {streamer_area['total_mm2']:.4f} mm^2  (64 KB SRAM + ctrl)")
    print(f"WeightStreamer power    : {streamer_power['total_mw']:.2f} mW (active+idle)\n")

    print("Workload cycles:")
    print(f"  heuristic     : {h_run['total_cycles']:>10,d}  fusion-cycles  ({h_summary['ms_per_frame_pipelined']:.2f} ms, {h_summary['fps_pipelined_500MHz']:.1f} fps)")
    print(f"  effvit_b1_int8: {e_run['total_cycles']:>10,d}  fusion-cycles  +  {args.sa_cycles_effvit:>10,d} SA-cycles  +  {stream_cycles_effvit:>10,d} stream-cycles")
    print(f"                  pipelined: {e_summary['pipelined_cycles']:>10,d} cycles → {e_summary['ms_per_frame_pipelined']:.2f} ms / {e_summary['fps_pipelined_500MHz']:.1f} fps")
    print(f"                  serial   : {e_summary['serial_cycles']:>10,d} cycles → {e_summary['ms_per_frame_serial']:.2f} ms / {e_summary['fps_serial_500MHz']:.1f} fps")

    print(f"\nPhase 10 total ADDED area: {out['phase10_summary']['phase10_added_area_mm2']:.4f} mm^2")
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
