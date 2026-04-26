"""
hardware/scu/dpc.py
===================
Dual Precision Controller (DPC).

Controller that orchestrates dual-path execution for decoder adaptive
precision.  Covers the FULL decoder taxonomy (Spec B): proj_1-4, rn_1-4,
path_1-4, output -- up to 12 dual-path stages depending on stage_policy.

For each affected stage the DPC:
  1. Generates the HP spatial mask (resize sensitivity + histogram threshold)
  2. Programs the SA with HP weight address -> SA runs HP conv -> buf A
  3. Programs the SA with LP weight address -> SA runs LP conv -> buf B
  4. Triggers FU blend: mask * A + (1-mask) * B -> result in A, free B

Event contract (Spec D):
  Per stage: mask_gen_start, mask_gen_done, blend_trigger
  Per op: dpc:op_complete
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from hardware.base_module import Event, HardwareModule, ModuleState
from hardware.interfaces import ConfigPort, MemoryPort


# ---------------------------------------------------------------------------
# Decoder Stage Tags (Spec B)
# ---------------------------------------------------------------------------

COARSE_TAGS = {"proj_3", "proj_4", "rn_3", "rn_4", "path_3", "path_4"}
FINE_TAGS = {"proj_1", "proj_2", "rn_1", "rn_2", "path_1", "path_2", "output"}
ALL_TAGS = COARSE_TAGS | FINE_TAGS

STAGE_RESOLUTIONS = {
    "proj_1": (37, 37), "proj_2": (74, 74), "proj_3": (148, 148), "proj_4": (296, 296),
    "rn_1": (37, 37), "rn_2": (74, 74), "rn_3": (148, 148), "rn_4": (296, 296),
    "path_4": (148, 148), "path_3": (74, 74), "path_2": (37, 37), "path_1": (37, 37),
    "output": (296, 296),
}


def affected_tags(stage_policy: str) -> set[str]:
    if stage_policy == "all":
        return ALL_TAGS
    elif stage_policy == "coarse_only":
        return COARSE_TAGS
    elif stage_policy == "fine_only":
        return FINE_TAGS
    elif stage_policy in ("none", "off", ""):
        return set()
    raise ValueError(f"Unknown stage_policy: {stage_policy}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPCConfig:
    """Parameterized DPC configuration."""
    histogram_bins: int = 256
    histogram_sram_bytes: int = 1024     # 256 bins * 4 bytes
    mask_fragment_bytes: int = 1024      # partial mask buffer
    l1_size_bytes: int = 2048            # 2 KB total internal SRAM
    sensitivity_element_bytes: int = 4   # FP32 sensitivity values
    parallel_pixels: int = 32            # pixels processed per cycle in histogram


# ---------------------------------------------------------------------------
# DPC Module
# ---------------------------------------------------------------------------

class DualPrecisionController(HardwareModule):
    """
    SCU-3: Decoder Precision Controller.

    Per dual-path decoder stage:
      1. mask_gen: resize sensitivity map to stage resolution, histogram
         threshold for top-K selection -> binary HP mask
      2. blend_trigger: after SA finishes HP and LP convolutions, signal
         FU to spatially blend the two outputs
    """

    def __init__(self, config: DPCConfig | None = None):
        self.config = config or DPCConfig()
        super().__init__(name="dpc")

    def _declare_ports(self) -> dict[str, Any]:
        return {
            "sensitivity_in": MemoryPort(
                name="sensitivity_in",
                data_width_bits=256,
                is_read=True, is_write=False,
            ),
            "mask_out": MemoryPort(
                name="mask_out",
                data_width_bits=256,
                is_read=False, is_write=True,
            ),
            "sa_config": ConfigPort(name="dpc_sa_config", data_width_bits=32),
            "blend_trigger": ConfigPort(name="dpc_blend_trigger", data_width_bits=1),
        }

    # -- Cycle estimation ----------------------------------------------------

    def estimate_mask_gen_cycles(self, stage_h: int, stage_w: int) -> dict[str, int]:
        """Cycles to generate the HP mask for one decoder stage."""
        cfg = self.config
        total_pixels = stage_h * stage_w

        # Phase 1: resize sensitivity map to stage resolution (bilinear)
        resize_cycles = max(1, math.ceil(total_pixels / cfg.parallel_pixels))

        # Phase 2: histogram build (1 pass over resized map)
        histogram_build = max(1, math.ceil(total_pixels / cfg.parallel_pixels))

        # Phase 3: scan histogram bins to find threshold (256 bins)
        histogram_scan = cfg.histogram_bins

        # Phase 4: threshold to binary mask + write to L2
        mask_bytes = max(1, math.ceil(total_pixels / 8))  # 1 bit per pixel
        mask_write = max(1, math.ceil(mask_bytes / self.ports["mask_out"].bytes_per_beat))

        total = resize_cycles + histogram_build + histogram_scan + mask_write
        return {
            "resize_cycles": resize_cycles,
            "histogram_build_cycles": histogram_build,
            "histogram_scan_cycles": histogram_scan,
            "mask_write_cycles": mask_write,
            "total_cycles": total,
            "stage_pixels": total_pixels,
        }

    def estimate_blend_cycles(self, stage_h: int, stage_w: int,
                              feature_channels: int = 64) -> int:
        """Cycles for FU to blend HP + LP outputs (delegated to FU)."""
        total_elements = stage_h * stage_w * feature_channels
        # 2-cycle alpha_blend at 32 elements/cycle
        return max(1, math.ceil(total_elements / 32) * 2)

    def estimate_total_cycles(
        self,
        stage_policy: str = "coarse_only",
        patch_h: int = 37,
        patch_w: int = 37,
    ) -> dict[str, Any]:
        """Total DPC overhead across all affected decoder stages."""
        tags = affected_tags(stage_policy)

        # Compute actual resolutions based on patch_h/patch_w
        scale_map = {
            "proj_1": (1, 1), "proj_2": (2, 2), "proj_3": (4, 4), "proj_4": (8, 8),
            "rn_1": (1, 1), "rn_2": (2, 2), "rn_3": (4, 4), "rn_4": (8, 8),
            "path_4": (4, 4), "path_3": (2, 2), "path_2": (1, 1), "path_1": (1, 1),
            "output": (8, 8),
        }

        per_stage = {}
        total_mask_gen = 0
        total_blend = 0

        for tag in sorted(tags):
            sh, sw = scale_map.get(tag, (1, 1))
            h, w = patch_h * sh, patch_w * sw
            mask_est = self.estimate_mask_gen_cycles(h, w)
            blend_est = self.estimate_blend_cycles(h, w)
            per_stage[tag] = {
                "resolution": (h, w),
                "mask_gen_cycles": mask_est["total_cycles"],
                "blend_cycles": blend_est,
            }
            total_mask_gen += mask_est["total_cycles"]
            total_blend += blend_est

        return {
            "stage_policy": stage_policy,
            "num_dual_path_stages": len(tags),
            "per_stage": per_stage,
            "total_mask_gen_cycles": total_mask_gen,
            "total_blend_cycles": total_blend,
            "total_dpc_overhead_cycles": total_mask_gen + total_blend,
        }

    # -- Event-driven interface (Spec D) ------------------------------------

    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        if not self.is_idle:
            return []

        self.current_op = op
        self.state = ModuleState.COMPUTE

        stage_tag = op.get("stage_tag", "rn_3")
        stage_h = op.get("stage_h", 148)
        stage_w = op.get("stage_w", 148)

        mask_est = self.estimate_mask_gen_cycles(stage_h, stage_w)

        events = []
        c = cycle

        events.append(self._emit(c, "dpc:mask_gen_start", {
            "op_id": op.get("id"), "stage_tag": stage_tag,
        }))

        c += mask_est["total_cycles"]
        events.append(self._emit(c, "dpc:mask_gen_done", {
            "op_id": op.get("id"), "stage_tag": stage_tag,
        }))

        # C4 fix: emit blend_trigger immediately after mask_gen_done.
        # The actual FU blend synchronization with SA HP+LP completion is
        # enforced by DAG dependency edges in simulator/event_simulator.py's
        # build_workload() (FU blend depends on both DPC mask and SA LP conv).
        events.append(self._emit(c, "dpc:blend_trigger", {
            "op_id": op.get("id"), "stage_tag": stage_tag,
        }))

        events.append(self._emit(c, "dpc:op_complete", {
            "op_id": op.get("id"),
            "total_cycles": c - cycle,
            **mask_est,
        }))

        return events

    def handle_event(self, event: Event, cycle: int) -> list[Event]:
        action = event.action

        if action == "dpc:mask_gen_start":
            self.state = ModuleState.COMPUTE
            return []

        if action == "dpc:mask_gen_done":
            self.state = ModuleState.WRITEBACK
            return []

        if action == "dpc:blend_trigger":
            # Purely a notification event; state already WRITEBACK
            return []

        if action == "dpc:op_complete":
            self.state = ModuleState.IDLE
            self.stats.ops_completed += 1
            self.current_op = None
            return [self._emit(cycle, "sched:op_complete", event.data)]

        return []

    # -- Spec / estimation ---------------------------------------------------

    def describe(self) -> dict:
        cfg = self.config
        return {
            "name": "DualPrecisionController",
            "histogram_bins": cfg.histogram_bins,
            "l1_size_bytes": cfg.l1_size_bytes,
            "parallel_pixels": cfg.parallel_pixels,
            "supported_policies": ["all", "coarse_only", "fine_only"],
            "decoder_tags": sorted(ALL_TAGS),
            "ports": {k: v.describe() for k, v in self.ports.items()},
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        scale = (node_nm / 28) ** 2
        histogram_sram = self.config.histogram_sram_bytes * 1e-6 * scale
        control = 0.002 * scale
        total = histogram_sram + control
        return {
            "histogram_sram_mm2": histogram_sram,
            "control_mm2": control,
            "total_mm2": total,
            "node_nm": node_nm,
        }

    def estimate_power_mw(
        self, node_nm: int = 28, freq_mhz: int = 500, utilization: float = 0.05,
    ) -> dict[str, float]:
        scale = (node_nm / 28) ** 1.1 * (freq_mhz / 500)
        total = 0.2 * scale * utilization
        return {"total_mw": total, "node_nm": node_nm, "freq_mhz": freq_mhz}
