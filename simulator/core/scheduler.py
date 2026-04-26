"""
simulator/core/scheduler.py
============================
Operation scheduler for the event-driven simulator.

Step 14 fix:
  C3: dispatch_ready() sorts ready ops by op_id for deterministic,
      reproducible simulation output.
"""
from __future__ import annotations

from typing import Any

from hardware.base_module import HardwareModule
from simulator.core.event_queue import EventQueue
from simulator.core.workload_dag import WorkloadDAG


class OperationScheduler:
    """DAG-aware operation dispatcher."""

    def __init__(self, dag: WorkloadDAG):
        self.dag = dag
        self.completed: set[int] = set()
        self.in_flight: dict[str, int] = {}
        self.dispatched: set[int] = set()

    def dispatch_ready(
        self,
        cycle: int,
        modules: dict[str, HardwareModule],
        event_queue: EventQueue,
    ) -> int:
        """
        Find and dispatch all ready operations to idle modules.

        C3: sort ready ops by op_id to guarantee deterministic order.
        """
        ready = self.dag.ready_ops(self.completed)
        # Deterministic ordering: sort by op_id (which is topologically consistent
        # since add_op assigns IDs in construction order)
        ready.sort()

        dispatched_count = 0

        for op_id in ready:
            if op_id in self.dispatched:
                continue

            op = self.dag.operations[op_id]
            engine = op.engine
            module = modules.get(engine)

            if module is None:
                continue
            if not module.is_idle:
                continue
            if engine in self.in_flight:
                continue

            op_dict = {
                "id": op_id,
                "name": op.name,
                "engine": engine,
                "flops": op.flops,
                "weight_bytes": op.weight_bytes,
                "input_bytes": op.input_bytes,
                "output_bytes": op.output_bytes,
                **op.metadata,
            }

            events = module.accept_op(op_dict, cycle)
            if events:
                event_queue.push_many(events)
                self.in_flight[engine] = op_id
                self.dispatched.add(op_id)
                dispatched_count += 1

        return dispatched_count

    def mark_complete(
        self,
        op_id: int,
        cycle: int,
        modules: dict[str, HardwareModule],
        event_queue: EventQueue,
    ) -> int:
        """Mark op complete and dispatch successors."""
        self.completed.add(op_id)
        op = self.dag.operations.get(op_id)
        if op:
            self.in_flight.pop(op.engine, None)
        return self.dispatch_ready(cycle, modules, event_queue)

    @property
    def all_done(self) -> bool:
        return len(self.completed) == len(self.dag)

    def progress(self) -> dict[str, Any]:
        return {
            "completed": len(self.completed),
            "total": len(self.dag),
            "in_flight": dict(self.in_flight),
            "dispatched": len(self.dispatched),
        }
