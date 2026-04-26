"""
hardware/scu/crm.py
===================
Confidence Router & Merge Planner (CRM).

Reads the externally-provided PKRN confidence map from L2, pools it
to the token grid, and generates merge plans (representative selection
+ member-to-representative assignment).

Modes:
  PRUNE       — threshold → binary keep/prune mask (1 cycle core)
  MERGE       — sort → select reps → distance assignment → LUT
  CAPS_MERGE  — MERGE + per-group sensitivity scoring + HP/LP mask

Confidence source: external SGM input (Spec A).  CRM has NO dependency
on ADCU.

Event contract (Spec D):
  fetch_conf → sort_done → assign_done → op_complete
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from hardware.base_module import Event, HardwareModule, ModuleState
from hardware.interfaces import ConfigPort, MemoryPort


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CRMConfig:
    """Parameterized CRM configuration."""
    max_tokens: int = 1369           # 37*37 default
    max_reps: int = 1024             # max representative count
    num_distance_units: int = 32     # parallel Euclidean distance comparators
    num_sort_comparators: int = 16   # parallel comparators in bitonic network
    l1_size_bytes: int = 4096        # 4 KB internal SRAM (index table + rep coords)
    conf_element_bytes: int = 2      # FP16 confidence values

    @property
    def index_entry_bytes(self) -> int:
        """Bytes per member_to_rep_local entry."""
        return 2  # INT16


# ---------------------------------------------------------------------------
# CRM Module
# ---------------------------------------------------------------------------

class ConfidenceRoutingModule(HardwareModule):
    """
    SCU-1: Confidence Router & Merge Planner.

    Pipeline stages:
      1. Pool: adaptive avg pool full-res confidence → token grid
      2. Sort: bitonic sort of grid confidence values
      3. Select: pick bottom-K as representatives
      4. Assign: nearest-representative assignment via parallel distance units
      5. Score (CAPS only): per-group sensitivity stats
      6. HP-Select (CAPS only): top-K on group scores
    """

    def __init__(self, config: CRMConfig | None = None):
        self.config = config or CRMConfig()
        super().__init__(name="crm")

    def _declare_ports(self) -> dict[str, Any]:
        return {
            "conf_in": MemoryPort(
                name="conf_in",
                data_width_bits=256,
                is_read=True, is_write=False,
            ),
            "lut_out": MemoryPort(
                name="lut_out",
                data_width_bits=256,
                is_read=False, is_write=True,
            ),
            "mode_in": ConfigPort(name="crm_mode", data_width_bits=2),
        }

    # -- Cycle estimation ----------------------------------------------------

    def estimate_pool_cycles(self, image_h: int, image_w: int,
                             grid_h: int, grid_w: int) -> int:
        """Streaming adaptive avg pool: read full-res, accumulate per grid cell."""
        total_pixels = image_h * image_w
        bytes_in = total_pixels * self.config.conf_element_bytes
        read_cycles = max(1, math.ceil(bytes_in / self.ports["conf_in"].bytes_per_beat))
        return read_cycles

    def estimate_sort_cycles(self, num_tokens: int) -> int:
        """Bitonic sort: O(n * log^2(n)) comparisons, pipelined."""
        if num_tokens <= 1:
            return 1
        log_n = math.ceil(math.log2(num_tokens))
        total_comparisons = num_tokens * log_n * (log_n + 1) // 2
        return max(1, math.ceil(total_comparisons / self.config.num_sort_comparators))

    def estimate_assign_cycles(self, num_tokens: int, num_reps: int) -> int:
        """Each token computes distance to all reps, takes argmin."""
        rep_chunks = math.ceil(num_reps / self.config.num_distance_units)
        return num_tokens * rep_chunks

    def estimate_score_cycles(self, num_reps: int) -> int:
        """Per-group streaming stats accumulation."""
        return num_reps

    def estimate_lut_write_cycles(self, num_tokens: int) -> int:
        """Write member_to_rep_local table to L2."""
        bytes_out = num_tokens * self.config.index_entry_bytes
        return max(1, math.ceil(bytes_out / self.ports["lut_out"].bytes_per_beat))

    def estimate_prune_cycles(self, num_tokens: int) -> int:
        """PRUNE mode: threshold + prefix-sum compaction."""
        threshold_cycles = 1  # parallel comparators (one per token, single cycle)
        prefix_sum_cycles = math.ceil(math.log2(max(num_tokens, 2)))
        return threshold_cycles + prefix_sum_cycles

    def estimate_total_cycles(
        self,
        mode: str,
        image_h: int = 518, image_w: int = 518,
        grid_h: int = 37, grid_w: int = 37,
        keep_ratio: float = 0.5,
    ) -> dict[str, Any]:
        """Full CRM cycle breakdown."""
        num_tokens = grid_h * grid_w
        num_reps = max(1, int(round(num_tokens * keep_ratio)))

        pool_cycles = self.estimate_pool_cycles(image_h, image_w, grid_h, grid_w)

        if mode == "PRUNE":
            core_cycles = self.estimate_prune_cycles(num_tokens)
            lut_cycles = self.estimate_lut_write_cycles(num_tokens)
            total = pool_cycles + core_cycles + lut_cycles
            return {
                "mode": mode,
                "pool_cycles": pool_cycles,
                "core_cycles": core_cycles,
                "lut_write_cycles": lut_cycles,
                "total_cycles": total,
                "num_tokens": num_tokens,
                "num_reps": num_reps,
            }

        sort_cycles = self.estimate_sort_cycles(num_tokens)
        assign_cycles = self.estimate_assign_cycles(num_tokens, num_reps)
        lut_cycles = self.estimate_lut_write_cycles(num_tokens)

        breakdown = {
            "mode": mode,
            "pool_cycles": pool_cycles,
            "sort_cycles": sort_cycles,
            "select_cycles": 1,
            "assign_cycles": assign_cycles,
            "lut_write_cycles": lut_cycles,
            "num_tokens": num_tokens,
            "num_reps": num_reps,
        }

        total = pool_cycles + sort_cycles + 1 + assign_cycles + lut_cycles

        if mode == "CAPS_MERGE":
            score_cycles = self.estimate_score_cycles(num_reps)
            hp_select_cycles = self.estimate_sort_cycles(num_reps)
            breakdown["score_cycles"] = score_cycles
            breakdown["hp_select_cycles"] = hp_select_cycles
            total += score_cycles + hp_select_cycles

        breakdown["total_cycles"] = total
        return breakdown

    # -- Event-driven interface (Spec D) ------------------------------------

    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        if not self.is_idle:
            return []

        self.current_op = op
        self.state = ModuleState.FETCH

        mode = op.get("crm_mode", "MERGE")
        est = self.estimate_total_cycles(
            mode=mode,
            image_h=op.get("image_h", 518),
            image_w=op.get("image_w", 518),
            grid_h=op.get("grid_h", 37),
            grid_w=op.get("grid_w", 37),
            keep_ratio=op.get("keep_ratio", 0.5),
        )

        events = []
        c = cycle

        # Stage 1: fetch confidence
        c += est["pool_cycles"]
        events.append(self._emit(c, "crm:fetch_conf_done", {"op_id": op.get("id")}))

        # Stage 2: sort (or threshold for PRUNE)
        if mode == "PRUNE":
            c += est["core_cycles"]
        else:
            c += est["sort_cycles"] + 1  # sort + select
        events.append(self._emit(c, "crm:sort_done", {"op_id": op.get("id")}))

        # Stage 3: assign (MERGE/CAPS only)
        if mode != "PRUNE":
            c += est["assign_cycles"]
        events.append(self._emit(c, "crm:assign_done", {"op_id": op.get("id")}))

        # Stage 4: CAPS scoring (optional)
        if mode == "CAPS_MERGE":
            c += est.get("score_cycles", 0) + est.get("hp_select_cycles", 0)

        # Stage 5: LUT writeback
        c += est["lut_write_cycles"]
        events.append(self._emit(c, "crm:op_complete", {
            "op_id": op.get("id"),
            "total_cycles": c - cycle,
            **est,
        }))

        return events

    def handle_event(self, event: Event, cycle: int) -> list[Event]:
        action = event.action

        if action == "crm:fetch_conf_done":
            self.state = ModuleState.COMPUTE
            return []

        if action == "crm:sort_done":
            return []

        if action == "crm:assign_done":
            self.state = ModuleState.WRITEBACK
            return []

        if action == "crm:op_complete":
            self.state = ModuleState.IDLE
            self.stats.ops_completed += 1
            self.current_op = None
            return [self._emit(cycle, "sched:op_complete", event.data)]

        return []

    # -- Spec / estimation ---------------------------------------------------

    def describe(self) -> dict:
        cfg = self.config
        return {
            "name": "ConfidenceRoutingModule",
            "max_tokens": cfg.max_tokens,
            "max_reps": cfg.max_reps,
            "num_distance_units": cfg.num_distance_units,
            "num_sort_comparators": cfg.num_sort_comparators,
            "l1_size_bytes": cfg.l1_size_bytes,
            "modes": ["PRUNE", "MERGE", "CAPS_MERGE"],
            "confidence_source": "external SGM (DRAM -> L2)",
            "ports": {k: v.describe() for k, v in self.ports.items()},
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        scale = (node_nm / 28) ** 2
        cfg = self.config
        # 32 distance units: each ~800 um^2 (2 muls + adder + comparator)
        distance_area = cfg.num_distance_units * 800e-6 * scale
        # Sort network comparators
        sort_area = cfg.num_sort_comparators * 200e-6 * scale
        # Index SRAM
        sram_area = cfg.l1_size_bytes * 1e-6 * scale  # ~1 um^2/byte at 28nm
        # Control
        control_area = 0.002 * scale

        total = distance_area + sort_area + sram_area + control_area
        return {
            "distance_units_mm2": distance_area,
            "sort_network_mm2": sort_area,
            "l1_sram_mm2": sram_area,
            "control_mm2": control_area,
            "total_mm2": total,
            "node_nm": node_nm,
        }

    def estimate_power_mw(
        self, node_nm: int = 28, freq_mhz: int = 500, utilization: float = 0.1,
    ) -> dict[str, float]:
        scale = (node_nm / 28) ** 1.1 * (freq_mhz / 500)
        cfg = self.config
        distance_power = cfg.num_distance_units * 0.05 * scale * utilization
        sort_power = cfg.num_sort_comparators * 0.02 * scale * utilization
        sram_power = 0.5 * scale * utilization
        total = distance_power + sort_power + sram_power
        return {"total_mw": total, "node_nm": node_nm, "freq_mhz": freq_mhz}
