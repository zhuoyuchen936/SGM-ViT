"""
simulator/core/workload_dag.py
==============================
Operation DAG for the event-driven simulator.

Replaces the flat operation list from the old batch-mode simulator.
Each operation has typed predecessors and successors; the scheduler
dispatches operations whose predecessors are all complete.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Operation:
    """A single schedulable operation in the workload DAG."""
    id: int
    name: str
    engine: str                        # module that executes this op
    flops: int = 0
    weight_bytes: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    output_buffer: str = ""            # L2 buffer name (for RAW tracking)
    predecessors: list[int] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkloadDAG:
    """
    Directed acyclic graph of operations.

    Provides topological ordering, ready-op queries, and critical-path
    analysis for the event-driven scheduler.
    """

    def __init__(self):
        self.operations: dict[int, Operation] = {}
        self._next_id: int = 0

    def add_op(self, name: str, engine: str, **kwargs) -> int:
        """Add an operation and return its ID."""
        op_id = self._next_id
        self._next_id += 1
        op = Operation(id=op_id, name=name, engine=engine, **kwargs)
        self.operations[op_id] = op
        return op_id

    def add_edge(self, from_id: int, to_id: int) -> None:
        """Add a dependency: to_id cannot start until from_id completes."""
        if from_id not in self.operations or to_id not in self.operations:
            raise ValueError(f"Invalid edge {from_id} -> {to_id}")
        if from_id == to_id:
            raise ValueError(f"Self-loop: {from_id}")
        self.operations[from_id].successors.append(to_id)
        self.operations[to_id].predecessors.append(from_id)

    def add_chain(self, *op_ids: int) -> None:
        """Add sequential dependency edges: op_ids[0] -> op_ids[1] -> ..."""
        for i in range(len(op_ids) - 1):
            self.add_edge(op_ids[i], op_ids[i + 1])

    def ready_ops(self, completed: set[int]) -> list[int]:
        """Return op IDs whose predecessors are all in `completed`."""
        ready = []
        for op_id, op in self.operations.items():
            if op_id in completed:
                continue
            if all(p in completed for p in op.predecessors):
                ready.append(op_id)
        return ready

    def topological_order(self) -> list[int]:
        """Kahn's algorithm for topological sort."""
        in_degree = {op_id: len(op.predecessors) for op_id, op in self.operations.items()}
        queue = deque(op_id for op_id, d in in_degree.items() if d == 0)
        order = []
        while queue:
            op_id = queue.popleft()
            order.append(op_id)
            for succ in self.operations[op_id].successors:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        if len(order) != len(self.operations):
            raise RuntimeError("Cycle detected in workload DAG!")
        return order

    def critical_path(self, cycle_estimates: dict[int, int] | None = None) -> tuple[list[int], int]:
        """
        Find the critical path (longest path) through the DAG.

        Parameters
        ----------
        cycle_estimates : mapping from op_id to estimated cycles.
            If None, uses op.flops // 1024 as a rough proxy.

        Returns
        -------
        (path_ids, total_cycles)
        """
        if cycle_estimates is None:
            cycle_estimates = {
                op_id: max(1, op.flops // 1024)
                for op_id, op in self.operations.items()
            }

        topo = self.topological_order()
        dist: dict[int, int] = {}
        pred: dict[int, int] = {}

        for op_id in topo:
            cost = cycle_estimates.get(op_id, 1)
            if not self.operations[op_id].predecessors:
                dist[op_id] = cost
                pred[op_id] = -1
            else:
                best_parent = max(
                    self.operations[op_id].predecessors,
                    key=lambda p: dist.get(p, 0),
                )
                dist[op_id] = dist.get(best_parent, 0) + cost
                pred[op_id] = best_parent

        if not dist:
            return [], 0

        end = max(dist, key=dist.get)
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = pred.get(cur, -1)
        path.reverse()
        return path, dist[end]

    def ops_by_engine(self) -> dict[str, list[int]]:
        """Group operation IDs by engine."""
        groups: dict[str, list[int]] = defaultdict(list)
        for op_id, op in self.operations.items():
            groups[op.engine].append(op_id)
        return dict(groups)

    def summary(self) -> dict[str, Any]:
        """DAG statistics."""
        by_engine = self.ops_by_engine()
        total_flops = sum(op.flops for op in self.operations.values())
        return {
            "num_ops": len(self.operations),
            "total_flops": total_flops,
            "ops_by_engine": {eng: len(ids) for eng, ids in by_engine.items()},
            "num_edges": sum(len(op.successors) for op in self.operations.values()),
        }

    def __len__(self) -> int:
        return len(self.operations)

    # -- Migration helper ----------------------------------------------------

    @staticmethod
    def from_flat_ops(ops: list[dict]) -> "WorkloadDAG":
        """
        Convert old-style flat operation list to a DAG with
        per-engine sequential dependencies.

        This preserves the batch-mode assumption: operations on the
        same engine execute sequentially.
        """
        dag = WorkloadDAG()
        engine_last: dict[str, int] = {}

        for op_dict in ops:
            op_id = dag.add_op(
                name=op_dict.get("name", "unknown"),
                engine=op_dict.get("engine", "systolic_array"),
                flops=op_dict.get("flops", 0),
                weight_bytes=op_dict.get("weight_bytes", 0),
                input_bytes=op_dict.get("input_bytes", 0),
                output_bytes=op_dict.get("output_bytes", 0),
                metadata=op_dict,
            )
            engine = op_dict.get("engine", "systolic_array")
            if engine in engine_last:
                dag.add_edge(engine_last[engine], op_id)
            engine_last[engine] = op_id

        return dag
