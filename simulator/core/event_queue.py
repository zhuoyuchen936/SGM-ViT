"""
simulator/core/event_queue.py
=============================
Min-heap priority queue for discrete-event simulation.

Events are ordered by (cycle, priority).  The main simulation loop
uses event-skip: it jumps to the next event cycle rather than ticking
every cycle.  Idle/stall gaps are batch-accumulated per module.
"""
from __future__ import annotations

import heapq
from typing import Optional

from hardware.base_module import Event


class EventQueue:
    """Min-heap priority queue of simulation events."""

    def __init__(self):
        self._heap: list[Event] = []
        self._counter: int = 0  # tie-breaker for equal (cycle, priority)

    def push(self, event: Event) -> None:
        # heapq uses tuple comparison: (cycle, priority, counter) for stability
        heapq.heappush(self._heap, (event.cycle, event.priority, self._counter, event))
        self._counter += 1

    def push_many(self, events: list[Event]) -> None:
        for e in events:
            self.push(e)

    def pop(self) -> Event:
        _, _, _, event = heapq.heappop(self._heap)
        return event

    def peek(self) -> Optional[Event]:
        if not self._heap:
            return None
        return self._heap[0][3]

    def peek_cycle(self) -> Optional[int]:
        if not self._heap:
            return None
        return self._heap[0][0]

    def drain_cycle(self, cycle: int) -> list[Event]:
        """Pop all events at the given cycle."""
        events = []
        while self._heap and self._heap[0][0] == cycle:
            events.append(self.pop())
        return events

    def push_at(self, cycle: int, module_id: str, action: str,
                data: Optional[dict] = None, priority: int = 0) -> None:
        self.push(Event(
            cycle=cycle, priority=priority,
            module_id=module_id, action=action,
            data=data or {},
        ))

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return not self.is_empty()
