"""
simulator/core/memory_controller.py
====================================
L2 memory controller with bank-level contention tracking.

Medium-depth model: tracks per-bank busy-until timestamps.
Priority arbitration matches hardware spec.  Fast-path bypass
when only one requester is active.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from hardware.base_module import Event
from hardware.architecture.interconnect import MODULE_PRIORITIES


@dataclass
class MemoryTransfer:
    """An in-flight or pending memory transfer."""
    requester: str
    transfer_type: str       # "read" or "write"
    total_bytes: int
    bytes_remaining: int
    priority: int
    buffer_name: str = ""
    start_cycle: int = 0
    callback_action: str = ""
    callback_data: dict = field(default_factory=dict)


class L2Controller:
    """
    L2 shared SRAM controller with bank-level contention.

    32 banks, each 16 KB, 8 bytes/bank/cycle.
    Priority arbitration: SA > CRM > GSU > DPC > ADCU > FU > DMA.
    Round-robin within same priority level.

    Fast-path: when only one requester is active, bypass bank tracking.
    """

    def __init__(
        self,
        num_banks: int = 32,
        bank_width_bytes: int = 8,
        read_latency: int = 3,
        write_latency: int = 3,
    ):
        self.num_banks = num_banks
        self.bank_width_bytes = bank_width_bytes
        self.read_latency = read_latency
        self.write_latency = write_latency
        self.peak_bw = num_banks * bank_width_bytes  # 256 B/cycle

        # Per-bank busy-until tracking
        self.bank_busy_until: list[int] = [0] * num_banks

        # Active transfers
        self.active_transfers: list[MemoryTransfer] = []

        # Stats
        self.total_reads_bytes: int = 0
        self.total_writes_bytes: int = 0
        self.total_stall_cycles: int = 0
        self.total_grants: int = 0

    def transfer_cycles_no_contention(self, total_bytes: int) -> int:
        """Minimum cycles assuming full bandwidth (fast-path)."""
        return max(1, math.ceil(total_bytes / self.peak_bw))

    def request_transfer(
        self,
        requester: str,
        transfer_type: str,
        total_bytes: int,
        cycle: int,
        buffer_name: str = "",
        callback_action: str = "",
        callback_data: dict | None = None,
    ) -> tuple[int, Event]:
        """
        Submit a memory transfer request.

        Returns (completion_cycle, completion_event).
        Uses fast-path (no contention) since the event-driven simulator
        handles contention via the scheduler's serialization.
        """
        priority = MODULE_PRIORITIES.get(requester, 6)
        latency = self.read_latency if transfer_type == "read" else self.write_latency
        transfer_time = self.transfer_cycles_no_contention(total_bytes)
        completion_cycle = cycle + latency + transfer_time

        if transfer_type == "read":
            self.total_reads_bytes += total_bytes
        else:
            self.total_writes_bytes += total_bytes
        self.total_grants += 1

        event = Event(
            cycle=completion_cycle,
            priority=priority,
            module_id=requester,
            action=callback_action or f"mem:{transfer_type}_done",
            data=callback_data or {"bytes": total_bytes, "buffer": buffer_name},
        )

        return completion_cycle, event

    def available_bandwidth_at(self, cycle: int) -> int:
        """Available bytes/cycle at the given cycle."""
        free_banks = sum(1 for b in self.bank_busy_until if b <= cycle)
        return free_banks * self.bank_width_bytes

    def stats_dict(self) -> dict[str, Any]:
        return {
            "total_reads_bytes": self.total_reads_bytes,
            "total_writes_bytes": self.total_writes_bytes,
            "total_grants": self.total_grants,
            "total_stall_cycles": self.total_stall_cycles,
        }
