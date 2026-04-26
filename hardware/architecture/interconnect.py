"""
hardware/architecture/interconnect.py
=====================================
L2 arbiter and on-chip interconnect model.

Models the 32-bank L2 SRAM crossbar with priority-based arbitration.
Used by both the hardware behavioral models and the event-driven simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hardware.base_module import Event, HardwareModule, ModuleState
from hardware.interfaces import MemoryPort


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ArbiterConfig:
    """L2 arbiter configuration."""
    num_banks: int = 32
    bank_size_bytes: int = 16384       # 16 KB per bank -> 512 KB total
    bank_width_bytes: int = 8          # 8 bytes per bank per cycle
    read_latency_cycles: int = 3
    write_latency_cycles: int = 3

    @property
    def total_capacity_bytes(self) -> int:
        return self.num_banks * self.bank_size_bytes

    @property
    def total_capacity_kb(self) -> int:
        return self.total_capacity_bytes // 1024

    @property
    def peak_bandwidth_bytes_per_cycle(self) -> int:
        return self.num_banks * self.bank_width_bytes


# Priority levels (lower = higher priority)
MODULE_PRIORITIES = {
    "systolic_array": 0,
    "crm": 1,
    "gsu": 2,
    "dpc": 3,
    "adcu": 4,
    "fu": 5,
    "dma": 6,
}


# ---------------------------------------------------------------------------
# Memory Request Tracking
# ---------------------------------------------------------------------------

@dataclass
class MemoryRequest:
    """A pending memory access request."""
    requester: str
    request_type: str             # "read" or "write"
    total_bytes: int
    bytes_remaining: int
    priority: int
    buffer_name: str = ""         # logical buffer in L2
    start_cycle: int = 0
    granted_bytes_so_far: int = 0


# ---------------------------------------------------------------------------
# L2 Arbiter
# ---------------------------------------------------------------------------

class L2Arbiter(HardwareModule):
    """
    Round-robin arbiter with priority override for L2 shared SRAM.

    32 independently addressable banks.  Up to 32 non-conflicting
    accesses can proceed in parallel.  Bank conflicts stall lower-
    priority requesters.

    Fast-path: when only one module accesses L2, bypass bank tracking
    and compute transfer_cycles = ceil(bytes / peak_bandwidth).
    """

    def __init__(self, config: ArbiterConfig | None = None):
        self.config = config or ArbiterConfig()

        # Per-bank busy-until cycle (0 = free)
        self.bank_busy_until: list[int] = [0] * self.config.num_banks

        # Pending requests queue (per-priority FIFO)
        self.pending: list[MemoryRequest] = []

        # Allocation tracking for budget validation
        self._allocated_buffers: dict[str, int] = {}  # buffer_name -> bytes

        super().__init__(name="l2_arbiter")

    def _declare_ports(self) -> dict[str, Any]:
        return {}  # arbiter has no external ports; it mediates others' ports

    # -- Bandwidth estimation ------------------------------------------------

    def transfer_cycles(self, total_bytes: int) -> int:
        """Minimum cycles assuming no contention (full bandwidth)."""
        return max(1, math.ceil(
            total_bytes / self.config.peak_bandwidth_bytes_per_cycle
        ))

    def transfer_cycles_at_bandwidth(self, total_bytes: int,
                                      available_bw_bytes_per_cycle: int) -> int:
        """Cycles at a given available bandwidth."""
        if available_bw_bytes_per_cycle <= 0:
            return total_bytes  # degenerate: 1 byte/cycle
        return max(1, math.ceil(total_bytes / available_bw_bytes_per_cycle))

    def free_banks_at(self, cycle: int) -> int:
        """Count banks free at the given cycle."""
        return sum(1 for busy in self.bank_busy_until if busy <= cycle)

    def available_bandwidth_at(self, cycle: int) -> int:
        """Available bandwidth in bytes/cycle at the given cycle."""
        return self.free_banks_at(cycle) * self.config.bank_width_bytes

    # -- Buffer allocation tracking (Spec C budget validation) ---------------

    def allocate_buffer(self, name: str, size_bytes: int) -> bool:
        """Track a logical buffer allocation.  Returns False if would exceed capacity."""
        current_total = sum(self._allocated_buffers.values())
        if current_total + size_bytes > self.config.total_capacity_bytes:
            return False
        self._allocated_buffers[name] = size_bytes
        return True

    def free_buffer(self, name: str) -> None:
        self._allocated_buffers.pop(name, None)

    def current_allocation_bytes(self) -> int:
        return sum(self._allocated_buffers.values())

    def validate_budget(self) -> dict[str, Any]:
        """Check if current allocations exceed L2 capacity."""
        total = self.current_allocation_bytes()
        capacity = self.config.total_capacity_bytes
        return {
            "allocated_bytes": total,
            "capacity_bytes": capacity,
            "utilization": total / capacity if capacity > 0 else 0.0,
            "ok": total <= capacity,
            "buffers": dict(self._allocated_buffers),
        }

    # -- Event-driven interface (pass-through for arbiter) ------------------

    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        return []  # arbiter doesn't accept ops directly

    def handle_event(self, event: Event, cycle: int) -> list[Event]:
        return []  # memory events are handled by the simulator's memory controller

    # -- Spec ----------------------------------------------------------------

    def describe(self) -> dict:
        cfg = self.config
        return {
            "name": "L2Arbiter",
            "num_banks": cfg.num_banks,
            "bank_size_bytes": cfg.bank_size_bytes,
            "total_capacity_kb": cfg.total_capacity_kb,
            "bank_width_bytes": cfg.bank_width_bytes,
            "peak_bandwidth_bytes_per_cycle": cfg.peak_bandwidth_bytes_per_cycle,
            "read_latency_cycles": cfg.read_latency_cycles,
            "write_latency_cycles": cfg.write_latency_cycles,
            "arbitration": "round-robin with priority override",
            "priorities": MODULE_PRIORITIES,
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        scale = (node_nm / 28) ** 2
        # L2 SRAM dominates; crossbar + arbiter logic is small
        sram_area = self.config.total_capacity_bytes / 1024 * 0.001 * scale
        crossbar = 0.02 * scale
        arbiter_logic = 0.005 * scale
        total = sram_area + crossbar + arbiter_logic
        return {
            "l2_sram_mm2": sram_area,
            "crossbar_mm2": crossbar,
            "arbiter_logic_mm2": arbiter_logic,
            "total_mm2": total,
            "node_nm": node_nm,
        }

    def estimate_power_mw(
        self, node_nm: int = 28, freq_mhz: int = 500, utilization: float = 0.5,
    ) -> dict[str, float]:
        scale = (node_nm / 28) ** 1.1 * (freq_mhz / 500)
        sram_power = self.config.total_capacity_bytes / 1024 * 0.3 * scale * utilization
        crossbar_power = 5.0 * scale * utilization
        total = sram_power + crossbar_power
        return {"total_mw": total, "node_nm": node_nm, "freq_mhz": freq_mhz}
