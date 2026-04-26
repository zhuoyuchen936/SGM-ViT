"""hardware/fusion_engine.py
============================
Unified Fusion Engine v2 (Phase 10 — replaces hardware/scu/fu.py).

Adds support for EffViT-Depth FPN decoder ops (3x3 depthwise conv, 1x1 conv,
Hardswish, BN affine) on top of the original FU heuristic ops (bilinear
upsample, elementwise, 3x3 spatial filter for bilateral/gaussian/sobel).

Sub-cores:
  1. ElementwiseCore (32 lanes) — add/mul/sub/clamp/Hardswish/BN-like affine
  2. Conv1x1Core (32 lanes)     — 1x1 conv / FPN lateral / project
  3. DepthwiseCore (16 lanes, 3x3) — NEW; per-channel 3x3 DW conv (MBConv)
  4. UpsamplerCore (32 px/cycle)— shift-add bilinear 2x

8-12 fixed instructions; no microcode. Heuristic FU ops re-route via
EDGE_AWARE_BLEND / HEURISTIC_3X3_FILTER (backward-compatible).

Event contract (preserves FusionUnit Spec D for simulator integration):
  accept_op(op, cycle) -> list[Event]
  handle_event(event, cycle) -> list[Event]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from hardware.base_module import Event, HardwareModule, ModuleState
from hardware.interfaces import MemoryPort


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FusionEngineConfig:
    """Phase 10 unified fusion engine configuration."""
    # ElementwiseCore
    ew_lanes: int = 32
    ew_op_cycles_per_pixel: int = 1  # hardswish ~3, BN-affine ~2, simple add 1

    # Conv1x1Core
    c1_lanes: int = 32  # 32 output channels per cycle
    c1_macs_per_lane_per_cycle: int = 1

    # DepthwiseCore (NEW)
    dw_lanes: int = 16  # process 16 channels in parallel
    dw_kernel: int = 3
    dw_macs_per_pixel: int = 9  # 3x3 = 9 MACs per output pixel per channel

    # UpsamplerCore
    bu_parallel_pixels: int = 32
    bu_pipeline_stages: int = 2

    # Streaming
    strip_height: int = 32

    # L1 (on-engine)
    l1_size_bytes: int = 16384  # 16 KB strip + bilateral LUT

    # Bilateral LUT (heuristic compat)
    bilateral_lut_bytes: int = 512


# ---------------------------------------------------------------------------
# Per-core cycle estimators
# ---------------------------------------------------------------------------

def _ew_cycles(total_pixels: int, lanes: int, ops_per_pixel: int = 1) -> int:
    return max(1, math.ceil(total_pixels * ops_per_pixel / max(lanes, 1)))


def _conv1x1_cycles(in_ch: int, out_ch: int, H: int, W: int, lanes: int) -> int:
    """1x1 conv: H*W*in_ch*out_ch MACs; lanes process `lanes` outputs in parallel."""
    macs = H * W * in_ch * out_ch
    return max(1, math.ceil(macs / max(lanes, 1)))


def _dw_cycles(in_ch: int, H: int, W: int, dw_lanes: int, macs_per_pixel: int) -> int:
    """3x3 depthwise: per pixel per channel = 9 MACs; channels parallelized."""
    pixels_per_channel = H * W
    parallel_passes = math.ceil(in_ch / max(dw_lanes, 1))
    return max(1, parallel_passes * pixels_per_channel * macs_per_pixel)


def _upsample_cycles(H_in: int, W_in: int, parallel: int, stages: int) -> int:
    H_out, W_out = H_in * 2, W_in * 2
    pixels = H_out * W_out
    return max(1, math.ceil(pixels / max(parallel, 1)) + stages)


# ---------------------------------------------------------------------------
# Instruction set
# ---------------------------------------------------------------------------

ISA = {
    # FPN / EffViT FPN decoder
    "UPSAMPLE_2X":      "Upsampler",     # bilinear 2x
    "CONV_1X1":         "Conv1x1",       # 1x1 conv (FPN lateral / project)
    "CONV_3X3_DW":      "Depthwise",     # 3x3 depthwise (MBConv)
    "HARDSWISH":        "Elementwise",   # x * relu6(x+3)/6
    "BN_AFFINE":        "Elementwise",   # (x-mu)/sigma * gamma + beta -> fused affine
    "RESIDUAL_ADD":     "Elementwise",   # two-input add
    "CLAMP_RESIDUAL":   "Elementwise",   # clamp(x, -32, +32)
    # Heuristic compat
    "EDGE_AWARE_BLEND":     "Elementwise+Depthwise",  # original FU fusion_pipeline
    "HEURISTIC_3X3_FILTER": "Depthwise",              # gaussian/sobel/bilateral
    "SELECT_MASK":          "Elementwise",            # mask select a/b
    # Control
    "LOAD_WEIGHT_TILE": "Controller",  # triggers WeightStreamer; no compute cycles
    "SYNC_BARRIER":     "Controller",  # 1-cycle stall
}


# ---------------------------------------------------------------------------
# Top-level engine
# ---------------------------------------------------------------------------

class FusionEngineV2(HardwareModule):
    """
    Phase 10 unified fusion engine.

    Replaces FusionUnit (hardware/scu/fu.py). Same module-name convention
    "fu" in simulator dispatch (op["engine"] == "fu") so existing DAG / scheduler
    continues to work. New op type field: op["fu_op_type"] uses ISA keys.

    Backward compat: original FU op types (fusion_pipeline, bilateral_filter,
    upsample_2x, element_blend, alpha_blend) are remapped to the closest ISA op.
    """

    def __init__(self, config: FusionEngineConfig | None = None) -> None:
        self.config = config or FusionEngineConfig()
        super().__init__(name="fu")

    def _declare_ports(self) -> dict[str, Any]:
        return {
            "data_a_in": MemoryPort(name="data_a_in", data_width_bits=256, is_read=True, is_write=False),
            "data_b_in": MemoryPort(name="data_b_in", data_width_bits=256, is_read=True, is_write=False),
            "weight_in": MemoryPort(name="weight_in", data_width_bits=256, is_read=True, is_write=False),
            "result_out": MemoryPort(name="result_out", data_width_bits=256, is_read=False, is_write=True),
        }

    # -- Per-instruction cycle estimators ------------------------------------

    def estimate_op_cycles(self, op: dict) -> int:
        """Dispatch op to the right core's cycle estimator. Returns total cycles."""
        cfg = self.config
        op_type = op.get("fu_op_type", "EDGE_AWARE_BLEND")
        H = op.get("H", 96)
        W = op.get("W", 192)
        in_ch = op.get("in_channels", op.get("channels", 32))
        out_ch = op.get("out_channels", in_ch)

        # ----- Backward-compatible op-type remapping -----
        # Original FU op_types → new ISA
        compat = {
            "fusion_pipeline":   "EDGE_AWARE_BLEND",
            "bilateral_filter":  "HEURISTIC_3X3_FILTER",
            "gaussian_blur":     "HEURISTIC_3X3_FILTER",
            "sobel_gradient":    "HEURISTIC_3X3_FILTER",
            "upsample_2x":       "UPSAMPLE_2X",
            "element_blend":     "RESIDUAL_ADD",
            "alpha_blend":       "RESIDUAL_ADD",
        }
        if op_type in compat:
            op_type = compat[op_type]

        # ----- Cycle estimators per ISA op -----
        if op_type == "UPSAMPLE_2X":
            return _upsample_cycles(H // 2, W // 2, cfg.bu_parallel_pixels, cfg.bu_pipeline_stages) if H % 2 == 0 \
                else _upsample_cycles(H, W, cfg.bu_parallel_pixels, cfg.bu_pipeline_stages)

        if op_type == "CONV_1X1":
            return _conv1x1_cycles(in_ch, out_ch, H, W, cfg.c1_lanes)

        if op_type == "CONV_3X3_DW":
            return _dw_cycles(in_ch, H, W, cfg.dw_lanes, cfg.dw_macs_per_pixel)

        if op_type == "HARDSWISH":
            # Hardswish: x * relu6(x+3)/6 ~ 3 ops per pixel per channel
            return _ew_cycles(H * W * in_ch, cfg.ew_lanes, ops_per_pixel=3)

        if op_type == "BN_AFFINE":
            # Affine: (x * gamma + beta), fused, 2 ops per pixel
            return _ew_cycles(H * W * in_ch, cfg.ew_lanes, ops_per_pixel=2)

        if op_type == "RESIDUAL_ADD":
            return _ew_cycles(H * W * in_ch, cfg.ew_lanes, ops_per_pixel=1)

        if op_type == "CLAMP_RESIDUAL":
            return _ew_cycles(H * W * in_ch, cfg.ew_lanes, ops_per_pixel=1)

        if op_type == "SELECT_MASK":
            return _ew_cycles(H * W * in_ch, cfg.ew_lanes, ops_per_pixel=2)

        if op_type == "HEURISTIC_3X3_FILTER":
            # 3x3 spatial filter on a single-channel image (or per-channel)
            channels = op.get("channels", 1)
            return _dw_cycles(channels, H, W, cfg.dw_lanes, cfg.dw_macs_per_pixel)

        if op_type == "EDGE_AWARE_BLEND":
            # Heuristic edge-aware residual fusion (original FU full pipeline).
            total = H * W
            bilateral = _dw_cycles(1, H, W, cfg.dw_lanes, cfg.dw_macs_per_pixel)
            gaussian = _dw_cycles(1, H, W, cfg.dw_lanes, cfg.dw_macs_per_pixel)
            sobel = 2 * _dw_cycles(1, H, W, cfg.dw_lanes, cfg.dw_macs_per_pixel)
            grad_mag = _ew_cycles(total, cfg.ew_lanes, ops_per_pixel=2)
            detail_score = _ew_cycles(total, cfg.ew_lanes, ops_per_pixel=4)
            alpha_blend = _ew_cycles(total, cfg.ew_lanes, ops_per_pixel=2)
            residual_add = _ew_cycles(total, cfg.ew_lanes, ops_per_pixel=2)
            num_strips = math.ceil(H / cfg.strip_height)
            strip_overhead = num_strips * 2 * max(1, math.ceil(cfg.strip_height * W * 4 / 32))
            return bilateral + gaussian + sobel + grad_mag + detail_score + alpha_blend + residual_add + strip_overhead

        if op_type == "LOAD_WEIGHT_TILE":
            # No compute; controller-only op. Cycles come from WeightStreamer.
            return op.get("streamer_cycles", 100)

        if op_type == "SYNC_BARRIER":
            return 1

        # Unknown op → default 1 cycle (safe noop).
        return op.get("cycles", 1)

    # -- Event-driven interface (preserves FU contract) ----------------------

    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        if not self.is_idle:
            return []

        self.current_op = op
        self.state = ModuleState.FETCH

        total_cycles = self.estimate_op_cycles(op)
        H = op.get("H", 96)
        num_strips = max(1, math.ceil(H / self.config.strip_height))

        events = []
        c = cycle
        if num_strips > 1 and total_cycles >= num_strips:
            strip_cycles = max(1, total_cycles // num_strips)
            for s in range(min(num_strips, 4)):
                c += strip_cycles
                events.append(self._emit(c, "fu:strip_compute_done", {
                    "op_id": op.get("id"), "strip": s,
                }))
            c = cycle + total_cycles
        else:
            c += total_cycles

        events.append(self._emit(c, "fu:op_complete", {
            "op_id": op.get("id"),
            "total_cycles": total_cycles,
            "fu_op_type": op.get("fu_op_type"),
        }))
        return events

    def handle_event(self, event: Event, cycle: int) -> list[Event]:
        action = event.action
        if action == "fu:strip_compute_done":
            self.state = ModuleState.COMPUTE
            return []
        if action == "fu:op_complete":
            self.state = ModuleState.IDLE
            self.stats.ops_completed += 1
            self.current_op = None
            return [self._emit(cycle, "sched:op_complete", event.data)]
        return []

    # -- Spec / area / power -------------------------------------------------

    def describe(self) -> dict:
        cfg = self.config
        return {
            "name": "FusionEngineV2",
            "phase": 10,
            "isa": list(ISA.keys()),
            "sub_cores": {
                "elementwise": {"lanes": cfg.ew_lanes, "ops": ["add", "sub", "mul", "hardswish", "bn_affine", "select"]},
                "conv1x1":     {"lanes": cfg.c1_lanes, "macs_per_lane_per_cycle": cfg.c1_macs_per_lane_per_cycle},
                "depthwise":   {"lanes": cfg.dw_lanes, "kernel": cfg.dw_kernel, "macs_per_pixel": cfg.dw_macs_per_pixel},
                "upsampler":   {"parallel_pixels": cfg.bu_parallel_pixels, "stages": cfg.bu_pipeline_stages},
            },
            "streaming": {"strip_height": cfg.strip_height, "l1_size_bytes": cfg.l1_size_bytes},
            "ports": {k: v.describe() for k, v in self.ports.items()},
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        scale = (node_nm / 28) ** 2
        cfg = self.config
        # Per-core area estimates (28 nm baseline).
        ew_mm2 = cfg.ew_lanes * 500e-6 * scale  # ~500 um^2 per lane
        c1_mm2 = cfg.c1_lanes * 800e-6 * scale  # 1x1 MAC slightly larger than EW lane
        dw_mm2 = cfg.dw_lanes * cfg.dw_macs_per_pixel * 500e-6 * scale  # 3x3 MAC array
        bu_mm2 = 0.005 * scale
        bilateral_lut = cfg.bilateral_lut_bytes * 1e-6 * scale
        l1_sram = cfg.l1_size_bytes * 1e-6 * scale
        control = 0.005 * scale  # ISA dispatcher + state machine
        total = ew_mm2 + c1_mm2 + dw_mm2 + bu_mm2 + bilateral_lut + l1_sram + control
        return {
            "elementwise_mm2": ew_mm2,
            "conv1x1_mm2": c1_mm2,
            "depthwise_mm2": dw_mm2,
            "upsampler_mm2": bu_mm2,
            "bilateral_lut_mm2": bilateral_lut,
            "l1_sram_mm2": l1_sram,
            "control_mm2": control,
            "total_mm2": total,
            "node_nm": node_nm,
        }

    def estimate_power_mw(
        self, node_nm: int = 28, freq_mhz: int = 500, utilization: float = 0.30,
    ) -> dict[str, float]:
        scale = (node_nm / 28) ** 1.1 * (freq_mhz / 500)
        # FusionEngineV2 baseline ~6 mW @ 28nm 500MHz 100% util (3 cores active).
        total = 6.0 * scale * utilization
        return {"total_mw": total, "node_nm": node_nm, "freq_mhz": freq_mhz, "utilization": utilization}


# Public re-export for backward-compat with FU users.
FusionUnit = FusionEngineV2
FUConfig = FusionEngineConfig
