"""hardware/architecture/weight_streamer.py
============================================
Phase 10 — DRAM WeightStreamer.

Streams INT8 weights from off-chip DRAM into a small on-die double-buffer
(64 KB) so the L2 doesn't have to hold all 4.85 MB of EffViT-B1 weights.

Access pattern: stage-sequential (EffViT runs stage 0 → stage 1 → ... → stage 4),
each stage's weights are streamed once, computed, then evicted.

Bandwidth model: 3 GB/s sustained DRAM (LPDDR4-2400 1-channel @ ~75% BW utilization)
  → 6 bytes/cycle @ 500 MHz.
Latency model: 100-cycle burst overhead + size/BW.

Public API:
  WeightStreamerConfig
  WeightStreamer(HardwareModule):
    fetch_tile(tile_bytes, cycle) -> (cycles_until_ready, tile_id)
    accept_op(op, cycle), handle_event(event, cycle)
    estimate_area_mm2 / estimate_power_mw
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from hardware.base_module import Event, HardwareModule, ModuleState
from hardware.interfaces import MemoryPort


@dataclass
class WeightStreamerConfig:
    """LPDDR4-2400-class single-channel streaming model."""
    dram_bytes_per_cycle: float = 6.0   # 3 GB/s @ 500 MHz
    burst_latency_cycles: int = 100     # one-time per fetch
    double_buffer_bytes: int = 64 * 1024  # 64 KB on-die (2 × 32 KB tiles)
    tile_bytes_default: int = 32 * 1024  # 32 KB per tile

    # Per-fetch power tax (DRAM I/O is dominant when active).
    dram_io_power_mw: float = 70.0  # active streaming
    dram_idle_mw: float = 5.0


class WeightStreamer(HardwareModule):
    """DRAM weight streamer with double-buffer prefetch.

    Each `fetch_tile(tile_bytes, cycle)` returns the cycle when the tile is
    available. Internally we keep a single counter `_next_free_cycle` so that
    consecutive fetches serialize correctly (one DRAM channel).
    """

    def __init__(self, config: WeightStreamerConfig | None = None) -> None:
        self.config = config or WeightStreamerConfig()
        super().__init__(name="weight_streamer")
        self._next_free_cycle = 0
        self._tiles_served = 0
        self._bytes_served = 0
        self._burst_count = 0

    def _declare_ports(self) -> dict[str, Any]:
        return {
            "dram_read": MemoryPort(name="dram_read", data_width_bits=64, is_read=True, is_write=False),
            "weight_out": MemoryPort(name="weight_out", data_width_bits=256, is_read=False, is_write=True),
        }

    # ----- Streaming logic --------------------------------------------------

    def fetch_tile(self, tile_bytes: int, cycle: int) -> tuple[int, int]:
        """Schedule a tile fetch. Returns (ready_cycle, tile_id).

        Includes burst overhead + size/BW. Serializes with previous fetches.
        """
        cfg = self.config
        start = max(cycle, self._next_free_cycle)
        bw_cycles = math.ceil(tile_bytes / max(cfg.dram_bytes_per_cycle, 1.0))
        ready = start + cfg.burst_latency_cycles + bw_cycles
        self._next_free_cycle = ready
        self._tiles_served += 1
        self._bytes_served += tile_bytes
        self._burst_count += 1
        return ready, self._tiles_served - 1

    # ----- Module interface (op-based, for simulator integration) -----------

    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        """Accept a `LOAD_WEIGHT_TILE`-style op.

        op fields:
          - tile_bytes (int)         — bytes to fetch
          - dependent_op_id (str)    — which compute op consumes this tile
        """
        if not self.is_idle:
            return []
        self.current_op = op
        self.state = ModuleState.FETCH

        tile_bytes = op.get("tile_bytes", self.config.tile_bytes_default)
        ready, tile_id = self.fetch_tile(tile_bytes, cycle)

        return [
            self._emit(ready, "weight_streamer:tile_ready", {
                "op_id": op.get("id"),
                "tile_id": tile_id,
                "tile_bytes": tile_bytes,
                "dependent_op_id": op.get("dependent_op_id"),
            }),
        ]

    def handle_event(self, event: Event, cycle: int) -> list[Event]:
        if event.action == "weight_streamer:tile_ready":
            self.state = ModuleState.IDLE
            self.stats.ops_completed += 1
            self.current_op = None
            return [self._emit(cycle, "sched:op_complete", event.data)]
        return []

    # ----- Spec / area / power ----------------------------------------------

    def describe(self) -> dict:
        cfg = self.config
        return {
            "name": "WeightStreamer",
            "phase": 10,
            "dram_bandwidth_GBps": cfg.dram_bytes_per_cycle * 500e6 / 1e9,  # @ 500 MHz
            "burst_latency_cycles": cfg.burst_latency_cycles,
            "double_buffer_bytes": cfg.double_buffer_bytes,
            "tile_bytes_default": cfg.tile_bytes_default,
            "ports": {k: v.describe() for k, v in self.ports.items()},
            "stats": {
                "tiles_served": self._tiles_served,
                "bytes_served": self._bytes_served,
            },
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        scale = (node_nm / 28) ** 2
        cfg = self.config
        # 64 KB SRAM double-buffer @ ~1 um^2 per byte (28 nm).
        sram_mm2 = cfg.double_buffer_bytes * 1e-6 * scale
        # AXI master + control + datapath glue.
        ctrl_mm2 = 0.020 * scale
        total = sram_mm2 + ctrl_mm2
        return {
            "double_buffer_sram_mm2": sram_mm2,
            "control_mm2": ctrl_mm2,
            "total_mm2": total,
            "node_nm": node_nm,
        }

    def estimate_power_mw(
        self, node_nm: int = 28, freq_mhz: int = 500, utilization: float = 0.40,
    ) -> dict[str, float]:
        cfg = self.config
        scale = (freq_mhz / 500)
        active = cfg.dram_io_power_mw * utilization * scale
        idle = cfg.dram_idle_mw * (1 - utilization)
        total = active + idle
        return {
            "active_mw": active,
            "idle_mw": idle,
            "total_mw": total,
            "node_nm": node_nm, "freq_mhz": freq_mhz, "utilization": utilization,
        }
