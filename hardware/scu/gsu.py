"""
hardware/scu/gsu.py
===================
Gather-Scatter Unit (GSU).

Address-generation and data-steering unit between L2 and the SA's L1
input buffer.  No arithmetic -- translates token indices to SRAM
addresses and issues read/write commands.

Streaming rule (Spec C): gathers/scatters in chunks of ``chunk_size``
tokens to bound L2 usage.

Modes:
  GATHER         — index-select M representative tokens from full sequence
  SCATTER_PRUNE  — write M kept token outputs back to original positions
  SCATTER_MERGE  — broadcast each representative output to all group members

Event contract (Spec D):
  Per chunk: chunk_gather_done / chunk_scatter_done
  Per op: gsu:op_complete
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from hardware.base_module import Event, HardwareModule, ModuleState
from hardware.interfaces import MemoryPort


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GSUConfig:
    """Parameterized GSU configuration."""
    max_seq_len: int = 1370          # 1 CLS + 1369 patches
    embed_dim: int = 384             # token dimension
    element_bytes: int = 1           # INT8 activations
    chunk_size: int = 64             # tokens per chunk (bounds L2 at 24KB)
    l1_size_bytes: int = 8192        # 8 KB (index buffer + burst buffer)
    l2_bandwidth_bytes_per_cycle: int = 64  # L2 port width (32 bytes * 2 ports)


# ---------------------------------------------------------------------------
# GSU Module
# ---------------------------------------------------------------------------

class GatherScatterUnit(HardwareModule):
    """
    SCU-2: Gather-Scatter Unit.

    Gather: read index table from CRM, then for each chunk of tokens
    index-select from the full-sequence buffer in L2 into a compact buffer.

    Scatter-Merge: for each of N tokens, read the representative output
    (via member_to_rep_local LUT) and write it to the token's position
    in the full-sequence buffer.

    Scatter-Prune: write M kept-token outputs back to their original
    positions in the full-sequence buffer.
    """

    def __init__(self, config: GSUConfig | None = None):
        self.config = config or GSUConfig()
        super().__init__(name="gsu")

    def _declare_ports(self) -> dict[str, Any]:
        return {
            "index_in": MemoryPort(
                name="index_in",
                data_width_bits=256,
                is_read=True, is_write=False,
            ),
            "src_data": MemoryPort(
                name="src_data",
                data_width_bits=256,
                is_read=True, is_write=False,
            ),
            "dst_data": MemoryPort(
                name="dst_data",
                data_width_bits=256,
                is_read=False, is_write=True,
            ),
        }

    # -- Cycle estimation ----------------------------------------------------

    def _token_transfer_cycles(self, num_tokens: int) -> int:
        """Cycles to transfer num_tokens * embed_dim bytes through L2 port."""
        total_bytes = num_tokens * self.config.embed_dim * self.config.element_bytes
        return max(1, math.ceil(total_bytes / self.config.l2_bandwidth_bytes_per_cycle))

    def _index_load_cycles(self, num_entries: int) -> int:
        """Cycles to load index table entries from L2."""
        total_bytes = num_entries * 2  # INT16 indices
        return max(1, math.ceil(total_bytes / self.config.l2_bandwidth_bytes_per_cycle))

    def estimate_gather_cycles(self, num_reps: int) -> dict[str, Any]:
        """Gather: index-select num_reps tokens from full sequence."""
        index_load = self._index_load_cycles(num_reps)

        num_chunks = math.ceil(num_reps / self.config.chunk_size)
        per_chunk_transfer = self._token_transfer_cycles(
            min(self.config.chunk_size, num_reps)
        )
        # Address generation: 1 cycle per token (pipelined, overlaps with transfer)
        data_cycles = num_chunks * per_chunk_transfer
        total = index_load + data_cycles

        return {
            "mode": "GATHER",
            "num_tokens": num_reps,
            "num_chunks": num_chunks,
            "index_load_cycles": index_load,
            "data_transfer_cycles": data_cycles,
            "total_cycles": total,
        }

    def estimate_scatter_merge_cycles(self, num_tokens: int) -> dict[str, Any]:
        """Scatter-Merge: write all N tokens using LUT (many-to-one read)."""
        index_load = self._index_load_cycles(num_tokens)  # load member_to_rep_local

        num_chunks = math.ceil(num_tokens / self.config.chunk_size)
        per_chunk_transfer = self._token_transfer_cycles(
            min(self.config.chunk_size, num_tokens)
        )
        data_cycles = num_chunks * per_chunk_transfer
        total = index_load + data_cycles

        return {
            "mode": "SCATTER_MERGE",
            "num_tokens": num_tokens,
            "num_chunks": num_chunks,
            "index_load_cycles": index_load,
            "data_transfer_cycles": data_cycles,
            "total_cycles": total,
        }

    def estimate_scatter_prune_cycles(self, num_kept: int) -> dict[str, Any]:
        """Scatter-Prune: write M kept tokens back to original positions."""
        index_load = self._index_load_cycles(num_kept)
        data_cycles = self._token_transfer_cycles(num_kept)
        total = index_load + data_cycles

        return {
            "mode": "SCATTER_PRUNE",
            "num_tokens": num_kept,
            "index_load_cycles": index_load,
            "data_transfer_cycles": data_cycles,
            "total_cycles": total,
        }

    def estimate_total_cycles(
        self,
        mode: str,
        num_tokens: int = 1369,
        num_reps: int = 685,
    ) -> dict[str, Any]:
        if mode == "GATHER":
            return self.estimate_gather_cycles(num_reps)
        elif mode == "SCATTER_MERGE":
            return self.estimate_scatter_merge_cycles(num_tokens)
        elif mode == "SCATTER_PRUNE":
            return self.estimate_scatter_prune_cycles(num_reps)
        else:
            raise ValueError(f"Unknown GSU mode: {mode}")

    # -- Event-driven interface (Spec D) ------------------------------------

    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        if not self.is_idle:
            return []

        self.current_op = op
        self.state = ModuleState.FETCH

        mode = op.get("gsu_mode", "GATHER")
        est = self.estimate_total_cycles(
            mode=mode,
            num_tokens=op.get("num_tokens", 1369),
            num_reps=op.get("num_reps", 685),
        )

        events = []
        c = cycle
        num_chunks = est.get("num_chunks", 1)

        # Index load phase
        c += est["index_load_cycles"]

        # Per-chunk events
        chunk_transfer = max(1, est["data_transfer_cycles"] // max(num_chunks, 1))
        for chunk_idx in range(num_chunks):
            c += chunk_transfer
            if mode == "GATHER":
                events.append(self._emit(c, "gsu:chunk_gather_done", {
                    "op_id": op.get("id"), "chunk": chunk_idx,
                }))
            else:
                events.append(self._emit(c, "gsu:chunk_scatter_done", {
                    "op_id": op.get("id"), "chunk": chunk_idx,
                }))

        events.append(self._emit(c, "gsu:op_complete", {
            "op_id": op.get("id"),
            "total_cycles": c - cycle,
            **est,
        }))

        return events

    def handle_event(self, event: Event, cycle: int) -> list[Event]:
        action = event.action

        if action in ("gsu:chunk_gather_done", "gsu:chunk_scatter_done"):
            self.state = ModuleState.COMPUTE
            return []

        if action == "gsu:op_complete":
            self.state = ModuleState.IDLE
            self.stats.ops_completed += 1
            self.current_op = None
            return [self._emit(cycle, "sched:op_complete", event.data)]

        return []

    # -- Spec / estimation ---------------------------------------------------

    def describe(self) -> dict:
        cfg = self.config
        return {
            "name": "GatherScatterUnit",
            "max_seq_len": cfg.max_seq_len,
            "embed_dim": cfg.embed_dim,
            "chunk_size": cfg.chunk_size,
            "l1_size_bytes": cfg.l1_size_bytes,
            "modes": ["GATHER", "SCATTER_PRUNE", "SCATTER_MERGE"],
            "ports": {k: v.describe() for k, v in self.ports.items()},
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        scale = (node_nm / 28) ** 2
        addr_gen = 0.002 * scale    # 1 multiplier + 1 adder
        index_sram = self.config.l1_size_bytes * 1e-6 * scale
        control = 0.001 * scale
        total = addr_gen + index_sram + control
        return {
            "addr_gen_mm2": addr_gen,
            "l1_sram_mm2": index_sram,
            "control_mm2": control,
            "total_mm2": total,
            "node_nm": node_nm,
        }

    def estimate_power_mw(
        self, node_nm: int = 28, freq_mhz: int = 500, utilization: float = 0.1,
    ) -> dict[str, float]:
        scale = (node_nm / 28) ** 1.1 * (freq_mhz / 500)
        total = 0.3 * scale * utilization
        return {"total_mw": total, "node_nm": node_nm, "freq_mhz": freq_mhz}
