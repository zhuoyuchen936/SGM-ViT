"""
hardware/base_module.py
=======================
Abstract base class for all hardware modules in the EdgeStereoDAv2 accelerator.

Every module (SA, CRM, GSU, DPC, ADCU, FU) inherits from ``HardwareModule``
and implements the tick-based behavioral interface.  The base class enforces:

* Declared ports (validated at elaboration)
* Pipeline-stage tracking with multiple inflight operations
* Per-module statistics collection
* Area / power estimation hooks
* Event emission protocol (per Spec D in the plan)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from hardware.interfaces import Port


# ---------------------------------------------------------------------------
# Module State Machine
# ---------------------------------------------------------------------------

class ModuleState(Enum):
    """Unified state for all hardware modules."""
    IDLE = 0
    FETCH = 1
    COMPUTE = 2
    WRITEBACK = 3
    STALL_MEM = 4       # waiting for L2/DRAM grant
    STALL_DEP = 5       # waiting for predecessor operation
    DRAIN = 6           # pipeline draining
    RECONFIGURE = 7     # mode switch (SA only)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class ModuleStats:
    """Per-module accumulated statistics."""
    idle_cycles: int = 0
    fetch_cycles: int = 0
    compute_cycles: int = 0
    writeback_cycles: int = 0
    stall_mem_cycles: int = 0
    stall_dep_cycles: int = 0
    drain_cycles: int = 0
    reconfigure_cycles: int = 0
    total_macs: int = 0
    total_events_emitted: int = 0
    memory_reads_bytes: int = 0
    memory_writes_bytes: int = 0
    ops_completed: int = 0

    @property
    def busy_cycles(self) -> int:
        return (
            self.fetch_cycles + self.compute_cycles
            + self.writeback_cycles + self.drain_cycles
            + self.reconfigure_cycles
        )

    @property
    def total_cycles(self) -> int:
        return (
            self.busy_cycles + self.idle_cycles
            + self.stall_mem_cycles + self.stall_dep_cycles
        )

    @property
    def utilization(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.compute_cycles / self.total_cycles

    _STATE_FIELD = {
        "IDLE": "idle_cycles",
        "FETCH": "fetch_cycles",
        "COMPUTE": "compute_cycles",
        "WRITEBACK": "writeback_cycles",
        "STALL_MEM": "stall_mem_cycles",
        "STALL_DEP": "stall_dep_cycles",
        "DRAIN": "drain_cycles",
        "RECONFIGURE": "reconfigure_cycles",
    }

    def accumulate_gap(self, gap_cycles: int, state: "ModuleState") -> None:
        """Batch-accumulate cycles for event-skip gaps (Spec D accounting)."""
        if gap_cycles <= 0:
            return
        attr = self._STATE_FIELD.get(state.name)
        if attr is not None:
            setattr(self, attr, getattr(self, attr) + gap_cycles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idle_cycles": self.idle_cycles,
            "fetch_cycles": self.fetch_cycles,
            "compute_cycles": self.compute_cycles,
            "writeback_cycles": self.writeback_cycles,
            "stall_mem_cycles": self.stall_mem_cycles,
            "stall_dep_cycles": self.stall_dep_cycles,
            "drain_cycles": self.drain_cycles,
            "reconfigure_cycles": self.reconfigure_cycles,
            "busy_cycles": self.busy_cycles,
            "total_cycles": self.total_cycles,
            "utilization": round(self.utilization, 4),
            "total_macs": self.total_macs,
            "total_events_emitted": self.total_events_emitted,
            "memory_reads_bytes": self.memory_reads_bytes,
            "memory_writes_bytes": self.memory_writes_bytes,
            "ops_completed": self.ops_completed,
        }


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Event:
    """
    An event in the discrete-event simulation.

    Events are ordered by ``(cycle, priority)`` for the min-heap.
    Lower priority value = higher urgency.
    """
    cycle: int
    priority: int = 0
    module_id: str = field(default="", compare=False)
    action: str = field(default="", compare=False)
    data: dict = field(default_factory=dict, compare=False, repr=False)


# ---------------------------------------------------------------------------
# Pipeline Stage
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    """One stage of a module-internal pipeline."""
    name: str
    remaining_cycles: int = 0
    is_active: bool = False
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract Hardware Module
# ---------------------------------------------------------------------------

class HardwareModule(ABC):
    """
    Abstract base for all hardware modules.

    Subclasses must implement:
    * ``_declare_ports()``     -- return dict of Port objects
    * ``accept_op()``          -- try to start an operation
    * ``handle_event()``       -- react to an incoming event
    * ``describe()``           -- RTL-friendly spec dump
    * ``estimate_area_mm2()``  -- area breakdown
    * ``estimate_power_mw()``  -- power breakdown
    """

    def __init__(self, name: str):
        self.name = name
        self.state = ModuleState.IDLE
        self.stats = ModuleStats()
        self.ports: dict[str, Port] = self._declare_ports()
        self.pipeline: list[PipelineStage] = []
        self.current_op: Optional[dict] = None
        self._last_event_cycle: int = 0

    # -- Port declaration (subclass) -----------------------------------------

    @abstractmethod
    def _declare_ports(self) -> dict[str, Port]:
        """Return the module port map.  Called once at __init__."""
        ...

    # -- Operation interface (subclass) --------------------------------------

    @abstractmethod
    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        """
        Try to start *op* at *cycle*.

        Returns a list of events to push into the event queue (e.g.
        a fetch_start event at the current cycle, or a fetch_done
        event at a future cycle).

        If the module cannot accept (pipeline full), return an empty list
        and set self.state = ModuleState.STALL_DEP.
        """
        ...

    @abstractmethod
    def handle_event(self, event: Event, cycle: int) -> list[Event]:
        """
        React to *event* at *cycle*.

        Returns new events to push (e.g. after fetch_done, compute starts
        and a compute_done event is scheduled).
        """
        ...

    # -- Spec / estimation (subclass) ----------------------------------------

    @abstractmethod
    def describe(self) -> dict:
        """RTL-friendly specification dump."""
        ...

    @abstractmethod
    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        """Area breakdown at the given technology node."""
        ...

    @abstractmethod
    def estimate_power_mw(
        self,
        node_nm: int = 28,
        freq_mhz: int = 500,
        utilization: float = 0.8,
    ) -> dict[str, float]:
        """Power breakdown at the given technology node and frequency."""
        ...

    # -- Convenience ----------------------------------------------------------

    @property
    def is_idle(self) -> bool:
        return self.state == ModuleState.IDLE

    def accumulate_gap(self, gap_cycles: int) -> None:
        """Called by the simulator during event-skip to batch-count cycles."""
        self.stats.accumulate_gap(gap_cycles, self.state)

    def _emit(
        self,
        cycle: int,
        action: str,
        data: Optional[dict] = None,
        priority: int = 0,
    ) -> Event:
        """Helper to build an event originating from this module."""
        self.stats.total_events_emitted += 1
        return Event(
            cycle=cycle,
            priority=priority,
            module_id=self.name,
            action=action,
            data=data or {},
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name!r} state={self.state.name}>"
