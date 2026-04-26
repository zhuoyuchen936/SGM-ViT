"""
hardware/interfaces.py
======================
Typed signal and port protocol for RTL-ready hardware modeling.

Every inter-module connection uses ready/valid handshake semantics
that map directly to AXI-Stream in RTL.  In Python behavioral mode
the ports track timing only -- no actual data transfer occurs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Direction(Enum):
    """Signal direction relative to the owning module."""
    IN = "in"
    OUT = "out"


@dataclass(frozen=True)
class Signal:
    """A single named wire with a fixed bit-width and direction."""
    name: str
    width_bits: int
    direction: Direction

    def __post_init__(self):
        if self.width_bits <= 0:
            raise ValueError(
                f"Signal {self.name!r}: width_bits must be > 0, got {self.width_bits}"
            )


@dataclass
class Port:
    """
    A bundle of signals with optional ready/valid flow control.

    When ``ready_valid=True`` the port implicitly carries two
    additional 1-bit signals (``valid`` from producer, ``ready`` from
    consumer).  The behavioral model uses these to determine whether
    a transfer completes in a given cycle.
    """
    name: str
    signals: list[Signal] = field(default_factory=list)
    ready_valid: bool = True

    @property
    def total_data_bits(self) -> int:
        return sum(s.width_bits for s in self.signals)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signals": [
                {"name": s.name, "width": s.width_bits, "dir": s.direction.value}
                for s in self.signals
            ],
            "ready_valid": self.ready_valid,
            "total_data_bits": self.total_data_bits,
        }


@dataclass
class MemoryPort(Port):
    """
    Port for SRAM / DRAM access.

    Models a burst-capable memory interface.  ``addr_width`` and
    ``data_width_bits`` determine address space and per-cycle transfer
    width.  ``max_burst_len`` caps the number of beats in one burst.
    """
    addr_width: int = 20
    data_width_bits: int = 256    # 32 bytes/cycle
    max_burst_len: int = 16
    is_read: bool = True
    is_write: bool = True

    @property
    def bytes_per_beat(self) -> int:
        return self.data_width_bits // 8

    @property
    def max_burst_bytes(self) -> int:
        return self.bytes_per_beat * self.max_burst_len

    def transfer_beats(self, total_bytes: int) -> int:
        """Minimum beats to transfer ``total_bytes``."""
        return max(1, math.ceil(total_bytes / self.bytes_per_beat))

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({
            "addr_width": self.addr_width,
            "data_width_bits": self.data_width_bits,
            "max_burst_len": self.max_burst_len,
            "bytes_per_beat": self.bytes_per_beat,
            "is_read": self.is_read,
            "is_write": self.is_write,
        })
        return base


@dataclass
class StreamPort(Port):
    """
    Streaming data port (maps to AXI-Stream).

    ``data_width_bits`` is the payload width per beat.
    ``has_last`` indicates whether the port carries a ``tlast`` signal
    to mark end-of-packet / end-of-tile boundaries.
    """
    data_width_bits: int = 256
    has_last: bool = True

    @property
    def bytes_per_beat(self) -> int:
        return self.data_width_bits // 8

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({
            "data_width_bits": self.data_width_bits,
            "has_last": self.has_last,
            "bytes_per_beat": self.bytes_per_beat,
        })
        return base


@dataclass
class ConfigPort(Port):
    """
    Low-bandwidth configuration port for mode registers and tile parameters.

    Typically single-beat, no burst.
    """
    data_width_bits: int = 32
    ready_valid: bool = False

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({"data_width_bits": self.data_width_bits})
        return base
