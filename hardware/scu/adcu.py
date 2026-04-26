"""
hardware/scu/adcu.py
====================
Absolute Disparity Calibration Unit (ADCU).

Converts monocular relative depth to absolute stereo disparity via:
  1. Sparse NCC matching (32 keypoints)
  2. Scale-shift least-squares solve (2x2 system)
  3. Pixel-wise depth-to-disparity via reciprocal LUT

Per Spec A: ADCU does NOT produce PKRN confidence.  Confidence is an
external SGM input loaded from DRAM.

Event contract (Spec D):
  sparse_match_start → sparse_match_done → pixel_apply_start → op_complete
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
class ADCUConfig:
    """Parameterized ADCU configuration."""
    num_keypoints: int = 32
    max_disparity: int = 192
    patch_size: int = 11
    ncc_parallel: int = 32           # parallel NCC correlators
    lut_size: int = 4096             # reciprocal LUT entries
    lut_entry_bits: int = 16         # FP16 entries
    parallel_pixels: int = 32        # pixels processed per cycle in applicator
    ls_accumulate_parallel: int = 32 # parallel dot-product width for LS
    l1_size_bytes: int = 8192        # 8 KB (LUT + keypoint buffers)


# ---------------------------------------------------------------------------
# ADCU Module
# ---------------------------------------------------------------------------

class AbsoluteDisparityCU(HardwareModule):
    """
    SCU-4: Absolute Disparity Calibration Unit.

    Sub-units:
      1. Sparse NCC Matcher: 32 keypoints, 11x11 patches, max_disp=192
      2. Scale-Shift Solver: accumulate A^T*A (2x2) + A^T*b (2x1), solve
      3. Pixel Applicator: Z = s*d + t -> 1/Z (LUT) -> disp = B*f/Z
    """

    def __init__(self, config: ADCUConfig | None = None):
        self.config = config or ADCUConfig()
        super().__init__(name="adcu")

    def _declare_ports(self) -> dict[str, Any]:
        return {
            "mono_depth_in": MemoryPort(
                name="mono_depth_in",
                data_width_bits=256,
                is_read=True, is_write=False,
            ),
            "sgm_disp_in": MemoryPort(
                name="sgm_disp_in",
                data_width_bits=256,
                is_read=True, is_write=False,
            ),
            "aligned_out": MemoryPort(
                name="aligned_out",
                data_width_bits=256,
                is_read=False, is_write=True,
            ),
        }

    # -- Cycle estimation ----------------------------------------------------

    def estimate_sparse_match_cycles(self) -> int:
        """NCC on num_keypoints patches along max_disparity search range."""
        cfg = self.config
        ops_per_keypoint = cfg.max_disparity * cfg.patch_size * cfg.patch_size * 5
        total_ops = cfg.num_keypoints * ops_per_keypoint
        return max(1, math.ceil(total_ops / cfg.ncc_parallel))

    def estimate_ls_solve_cycles(self, n_valid: int | None = None) -> int:
        """Accumulate A^T*A and A^T*b over valid pixels + 2x2 inverse."""
        cfg = self.config
        if n_valid is None:
            n_valid = 10000  # typical count
        accumulate = max(1, math.ceil(n_valid / cfg.ls_accumulate_parallel))
        inverse_solve = 10  # hardwired 2x2 inverse (fixed formula)
        return accumulate + inverse_solve

    def estimate_pixel_apply_cycles(self, image_h: int, image_w: int) -> int:
        """Apply s*d + t + LUT reciprocal + multiply for all pixels."""
        total_pixels = image_h * image_w
        return max(1, math.ceil(total_pixels / self.config.parallel_pixels))

    def estimate_total_cycles(
        self,
        image_h: int = 518, image_w: int = 518,
        n_valid: int | None = None,
    ) -> dict[str, Any]:
        sparse_match = self.estimate_sparse_match_cycles()
        ls_solve = self.estimate_ls_solve_cycles(n_valid)
        pixel_apply = self.estimate_pixel_apply_cycles(image_h, image_w)
        total = sparse_match + ls_solve + pixel_apply

        return {
            "sparse_match_cycles": sparse_match,
            "ls_solve_cycles": ls_solve,
            "pixel_apply_cycles": pixel_apply,
            "total_cycles": total,
            "image_size": (image_h, image_w),
        }

    # -- Event-driven interface (Spec D) ------------------------------------

    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        if not self.is_idle:
            return []

        self.current_op = op
        self.state = ModuleState.COMPUTE

        est = self.estimate_total_cycles(
            image_h=op.get("image_h", 518),
            image_w=op.get("image_w", 518),
        )

        events = []
        c = cycle

        events.append(self._emit(c, "adcu:sparse_match_start", {"op_id": op.get("id")}))

        c += est["sparse_match_cycles"]
        events.append(self._emit(c, "adcu:sparse_match_done", {"op_id": op.get("id")}))

        c += est["ls_solve_cycles"]
        events.append(self._emit(c, "adcu:pixel_apply_start", {"op_id": op.get("id")}))

        c += est["pixel_apply_cycles"]
        events.append(self._emit(c, "adcu:op_complete", {
            "op_id": op.get("id"),
            "total_cycles": c - cycle,
            **est,
        }))

        return events

    def handle_event(self, event: Event, cycle: int) -> list[Event]:
        action = event.action

        if action == "adcu:sparse_match_start":
            self.state = ModuleState.COMPUTE
            return []

        if action == "adcu:sparse_match_done":
            return []

        if action == "adcu:pixel_apply_start":
            self.state = ModuleState.WRITEBACK
            return []

        if action == "adcu:op_complete":
            self.state = ModuleState.IDLE
            self.stats.ops_completed += 1
            self.current_op = None
            return [self._emit(cycle, "sched:op_complete", event.data)]

        return []

    # -- Spec / estimation ---------------------------------------------------

    def describe(self) -> dict:
        cfg = self.config
        return {
            "name": "AbsoluteDisparityCU",
            "num_keypoints": cfg.num_keypoints,
            "max_disparity": cfg.max_disparity,
            "patch_size": cfg.patch_size,
            "ncc_parallel": cfg.ncc_parallel,
            "lut_size": cfg.lut_size,
            "parallel_pixels": cfg.parallel_pixels,
            "l1_size_bytes": cfg.l1_size_bytes,
            "confidence_source": "NONE (external, per Spec A)",
            "ports": {k: v.describe() for k, v in self.ports.items()},
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        scale = (node_nm / 28) ** 2
        cfg = self.config
        ncc_area = 0.020 * scale  # 32 correlators
        solver_area = 0.005 * scale
        lut_sram = cfg.lut_size * (cfg.lut_entry_bits // 8) * 1e-6 * scale
        applicator = 0.010 * scale
        control = 0.002 * scale
        total = ncc_area + solver_area + lut_sram + applicator + control
        return {
            "ncc_matcher_mm2": ncc_area,
            "ls_solver_mm2": solver_area,
            "lut_sram_mm2": lut_sram,
            "pixel_applicator_mm2": applicator,
            "control_mm2": control,
            "total_mm2": total,
            "node_nm": node_nm,
        }

    def estimate_power_mw(
        self, node_nm: int = 28, freq_mhz: int = 500, utilization: float = 0.1,
    ) -> dict[str, float]:
        scale = (node_nm / 28) ** 1.1 * (freq_mhz / 500)
        total = 1.5 * scale * utilization
        return {"total_mw": total, "node_nm": node_nm, "freq_mhz": freq_mhz}
