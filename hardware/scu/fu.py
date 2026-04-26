"""
hardware/scu/fu.py
==================
Fusion Unit (FU).

Handles all element-wise, small-kernel spatial, and bilinear upsampling
operations that do not justify going through the systolic array.

Sub-units:
  1. BilinearUpsampleEngine — shift-add only, 32 pixels/cycle
  2. ElementWiseEngine — 32 lanes (mul/add/cmp/clamp/alpha_blend)
  3. SpatialFilterEngine — 3x3 kernel, 4 output pixels/cycle

Streaming rule (Spec C): processes in 32-row strips, L2 budget ~200KB
for 3 concurrent strip buffers.

Event contract (Spec D):
  Per strip: strip_fetch, strip_compute_done, strip_writeback_done
  Per op: fu:op_complete
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from hardware.base_module import Event, HardwareModule, ModuleState
from hardware.interfaces import MemoryPort


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FUConfig:
    """Parameterized FU configuration."""
    # ElementWise engine
    ew_parallel_pixels: int = 32
    ew_alpha_blend_cycles: int = 2   # cycles per alpha_blend op per pixel batch

    # SpatialFilter engine
    sf_output_pixels_per_cycle: int = 4  # 4 output pixels/cycle for 3x3
    sf_macs_per_pixel: int = 9           # 3x3 kernel

    # BilinearUpsample engine
    bu_parallel_pixels: int = 32
    bu_pipeline_stages: int = 2

    # Streaming
    strip_height: int = 32

    # L1
    l1_size_bytes: int = 16384  # 16 KB

    # Bilateral filter range-kernel LUT
    bilateral_lut_entries: int = 256
    bilateral_lut_bytes: int = 512


# ---------------------------------------------------------------------------
# Sub-unit cycle estimators
# ---------------------------------------------------------------------------

def _ew_cycles(total_pixels: int, parallel: int, ops_per_pixel: int = 1) -> int:
    """Element-wise engine cycles."""
    return max(1, math.ceil(total_pixels / parallel) * ops_per_pixel)


def _sf_cycles(total_pixels: int, output_per_cycle: int) -> int:
    """Spatial filter engine cycles."""
    return max(1, math.ceil(total_pixels / output_per_cycle))


def _bu_cycles(total_pixels: int, parallel: int, pipeline: int) -> int:
    """Bilinear upsample engine cycles."""
    return max(1, math.ceil(total_pixels / parallel) + pipeline)


# ---------------------------------------------------------------------------
# FU Module
# ---------------------------------------------------------------------------

class FusionUnit(HardwareModule):
    """
    SCU-5: Fusion Unit.

    Operations:
      - bilateral_filter: SpatialFilter engine
      - gaussian_blur: SpatialFilter engine
      - sobel_gradient: SpatialFilter engine (x2 for gx, gy)
      - element_blend: ElementWise engine (mul + add + clip)
      - alpha_blend: ElementWise engine (a*x + (1-a)*y)
      - bilinear_upsample_2x: BilinearUpsample engine
      - fuse_edge_aware_residual: full fusion pipeline (7 steps)
    """

    def __init__(self, config: FUConfig | None = None):
        self.config = config or FUConfig()
        super().__init__(name="fu")

    def _declare_ports(self) -> dict[str, Any]:
        return {
            "data_a_in": MemoryPort(
                name="data_a_in",
                data_width_bits=256,
                is_read=True, is_write=False,
            ),
            "data_b_in": MemoryPort(
                name="data_b_in",
                data_width_bits=256,
                is_read=True, is_write=False,
            ),
            "result_out": MemoryPort(
                name="result_out",
                data_width_bits=256,
                is_read=False, is_write=True,
            ),
        }

    # -- Per-operation cycle estimates ---------------------------------------

    def estimate_bilateral_filter_cycles(self, H: int, W: int) -> int:
        """Bilateral filter: 3x3 spatial kernel + range kernel LUT."""
        return _sf_cycles(H * W, self.config.sf_output_pixels_per_cycle)

    def estimate_gaussian_blur_cycles(self, H: int, W: int) -> int:
        return _sf_cycles(H * W, self.config.sf_output_pixels_per_cycle)

    def estimate_sobel_cycles(self, H: int, W: int) -> int:
        """Two Sobel passes (gx, gy) + gradient magnitude (element-wise)."""
        sobel = 2 * _sf_cycles(H * W, self.config.sf_output_pixels_per_cycle)
        magnitude = _ew_cycles(H * W, self.config.ew_parallel_pixels, 2)  # sqrt(gx^2+gy^2)
        return sobel + magnitude

    def estimate_element_blend_cycles(self, total_pixels: int) -> int:
        """Simple element-wise blend: a * x + (1-a) * y."""
        return _ew_cycles(total_pixels, self.config.ew_parallel_pixels,
                          self.config.ew_alpha_blend_cycles)

    def estimate_upsample_2x_cycles(self, H_in: int, W_in: int) -> int:
        """Bilinear 2x upsample."""
        H_out, W_out = H_in * 2, W_in * 2
        return _bu_cycles(H_out * W_out, self.config.bu_parallel_pixels,
                          self.config.bu_pipeline_stages)

    def estimate_fusion_pipeline_cycles(self, H: int, W: int) -> dict[str, Any]:
        """
        Full edge_aware_residual fusion pipeline (7 steps from core/fusion.py).

        Processes in strips of strip_height rows for L2 budget compliance.
        """
        cfg = self.config
        total = H * W

        bilateral = self.estimate_bilateral_filter_cycles(H, W)
        gaussian = self.estimate_gaussian_blur_cycles(H, W)
        sobel = self.estimate_sobel_cycles(H, W)
        grad_mag = _ew_cycles(total, cfg.ew_parallel_pixels, 1)
        detail_score = _ew_cycles(total, cfg.ew_parallel_pixels, 4)  # 4 weighted sums
        alpha_blend = _ew_cycles(total, cfg.ew_parallel_pixels, cfg.ew_alpha_blend_cycles)
        residual_add = _ew_cycles(total, cfg.ew_parallel_pixels, 2)  # mul + add + clip

        # Strip-based streaming overhead: read/write overhead per strip
        num_strips = math.ceil(H / cfg.strip_height)
        strip_overhead_per = 2 * max(1, math.ceil(
            cfg.strip_height * W * 4 / self.ports["data_a_in"].bytes_per_beat
        ))
        strip_overhead = num_strips * strip_overhead_per

        grand_total = (bilateral + gaussian + sobel + grad_mag
                       + detail_score + alpha_blend + residual_add
                       + strip_overhead)

        return {
            "bilateral_filter_cycles": bilateral,
            "gaussian_blur_cycles": gaussian,
            "sobel_gradient_cycles": sobel,
            "grad_magnitude_cycles": grad_mag,
            "detail_score_cycles": detail_score,
            "alpha_blend_cycles": alpha_blend,
            "residual_add_cycles": residual_add,
            "strip_overhead_cycles": strip_overhead,
            "total_cycles": grand_total,
            "image_size": (H, W),
            "num_strips": num_strips,
        }

    # -- Event-driven interface (Spec D) ------------------------------------

    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        if not self.is_idle:
            return []

        self.current_op = op
        self.state = ModuleState.FETCH

        fu_op_type = op.get("fu_op_type", "element_blend")
        H = op.get("H", 518)
        W = op.get("W", 518)

        if fu_op_type == "fusion_pipeline":
            est = self.estimate_fusion_pipeline_cycles(H, W)
            total_cycles = est["total_cycles"]
            num_strips = est["num_strips"]
        elif fu_op_type == "bilateral_filter":
            total_cycles = self.estimate_bilateral_filter_cycles(H, W)
            num_strips = math.ceil(H / self.config.strip_height)
        elif fu_op_type == "upsample_2x":
            total_cycles = self.estimate_upsample_2x_cycles(H, W)
            num_strips = 1
        elif fu_op_type == "element_blend":
            total_pixels = op.get("total_pixels", H * W)
            total_cycles = self.estimate_element_blend_cycles(total_pixels)
            num_strips = 1
        elif fu_op_type == "alpha_blend":
            total_pixels = op.get("total_pixels", H * W * op.get("channels", 64))
            total_cycles = self.estimate_element_blend_cycles(total_pixels)
            num_strips = 1
        else:
            total_cycles = op.get("cycles", 1)
            num_strips = 1

        events = []
        c = cycle

        if num_strips > 1:
            strip_cycles = max(1, total_cycles // num_strips)
            for s in range(min(num_strips, 4)):  # emit up to 4 strip events
                c += strip_cycles
                events.append(self._emit(c, "fu:strip_compute_done", {
                    "op_id": op.get("id"), "strip": s,
                }))
            # Jump to final
            c = cycle + total_cycles
        else:
            c += total_cycles

        events.append(self._emit(c, "fu:op_complete", {
            "op_id": op.get("id"),
            "total_cycles": total_cycles,
            "fu_op_type": fu_op_type,
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

    # -- Spec / estimation ---------------------------------------------------

    def describe(self) -> dict:
        cfg = self.config
        return {
            "name": "FusionUnit",
            "sub_units": {
                "bilinear_upsample": {
                    "type": "shift-add",
                    "parallel_pixels": cfg.bu_parallel_pixels,
                    "pipeline_stages": cfg.bu_pipeline_stages,
                },
                "element_wise": {
                    "parallel_lanes": cfg.ew_parallel_pixels,
                    "ops": ["add", "sub", "mul", "max", "min", "clamp", "alpha_blend"],
                },
                "spatial_filter": {
                    "kernel_size": "3x3",
                    "output_pixels_per_cycle": cfg.sf_output_pixels_per_cycle,
                    "supports": ["gaussian", "sobel", "bilateral"],
                },
            },
            "streaming": {
                "strip_height": cfg.strip_height,
                "l1_size_bytes": cfg.l1_size_bytes,
            },
            "ports": {k: v.describe() for k, v in self.ports.items()},
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        scale = (node_nm / 28) ** 2
        cfg = self.config
        bilinear = 0.005 * scale
        element_wise = cfg.ew_parallel_pixels * 500e-6 * scale  # ~500 um^2 per lane
        spatial_filter = cfg.sf_output_pixels_per_cycle * cfg.sf_macs_per_pixel * 400e-6 * scale
        bilateral_lut = cfg.bilateral_lut_bytes * 1e-6 * scale
        l1_sram = cfg.l1_size_bytes * 1e-6 * scale
        control = 0.002 * scale
        total = bilinear + element_wise + spatial_filter + bilateral_lut + l1_sram + control
        return {
            "bilinear_mm2": bilinear,
            "element_wise_mm2": element_wise,
            "spatial_filter_mm2": spatial_filter,
            "bilateral_lut_mm2": bilateral_lut,
            "l1_sram_mm2": l1_sram,
            "control_mm2": control,
            "total_mm2": total,
            "node_nm": node_nm,
        }

    def estimate_power_mw(
        self, node_nm: int = 28, freq_mhz: int = 500, utilization: float = 0.15,
    ) -> dict[str, float]:
        scale = (node_nm / 28) ** 1.1 * (freq_mhz / 500)
        total = 2.0 * scale * utilization
        return {"total_mw": total, "node_nm": node_nm, "freq_mhz": freq_mhz}
