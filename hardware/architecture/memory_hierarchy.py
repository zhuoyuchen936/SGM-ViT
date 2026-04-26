"""
hardware/architecture/memory_hierarchy.py
=========================================
Memory hierarchy specification for EdgeStereoDAv2.

Revised to include per-SCU L1 SRAMs and L2 budget validation
per Spec C (memory residency & streaming contract).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryLevel:
    """Specification for one level of the memory hierarchy."""
    name: str
    capacity_bytes: int
    bandwidth_bytes_per_cycle: int
    read_latency_cycles: int
    write_latency_cycles: int
    energy_per_read_pJ: float    # per 8-byte access
    energy_per_write_pJ: float
    num_banks: int
    bank_width_bytes: int


class MemoryHierarchySpec:
    """Complete memory hierarchy including SCU L1 SRAMs."""

    def __init__(self, process_node_nm: int = 28):
        self.node = process_node_nm
        self.levels = self._build_hierarchy()

    def _build_hierarchy(self) -> dict[str, MemoryLevel]:
        e_scale = (self.node / 28) ** (-0.7) if self.node != 28 else 1.0

        return {
            # -- L0: PE registers --
            "L0_RegFile": MemoryLevel(
                name="L0 Register File",
                capacity_bytes=512,
                bandwidth_bytes_per_cycle=64,
                read_latency_cycles=0, write_latency_cycles=0,
                energy_per_read_pJ=0.05 * e_scale,
                energy_per_write_pJ=0.05 * e_scale,
                num_banks=1, bank_width_bytes=64,
            ),
            # -- L1: SA (unified, replaces VFE+CSFE) --
            "L1_SA": MemoryLevel(
                name="L1 Systolic Array Local SRAM",
                capacity_bytes=64 * 1024,
                bandwidth_bytes_per_cycle=32,
                read_latency_cycles=1, write_latency_cycles=1,
                energy_per_read_pJ=2.0 * e_scale,
                energy_per_write_pJ=2.0 * e_scale,
                num_banks=16, bank_width_bytes=8,
            ),
            # -- L1: CRM --
            "L1_CRM": MemoryLevel(
                name="L1 CRM Local SRAM",
                capacity_bytes=4 * 1024,
                bandwidth_bytes_per_cycle=16,
                read_latency_cycles=1, write_latency_cycles=1,
                energy_per_read_pJ=1.5 * e_scale,
                energy_per_write_pJ=1.5 * e_scale,
                num_banks=4, bank_width_bytes=8,
            ),
            # -- L1: GSU --
            "L1_GSU": MemoryLevel(
                name="L1 GSU Local SRAM",
                capacity_bytes=8 * 1024,
                bandwidth_bytes_per_cycle=16,
                read_latency_cycles=1, write_latency_cycles=1,
                energy_per_read_pJ=1.5 * e_scale,
                energy_per_write_pJ=1.5 * e_scale,
                num_banks=4, bank_width_bytes=8,
            ),
            # -- L1: DPC --
            "L1_DPC": MemoryLevel(
                name="L1 DPC Local SRAM",
                capacity_bytes=2 * 1024,
                bandwidth_bytes_per_cycle=8,
                read_latency_cycles=1, write_latency_cycles=1,
                energy_per_read_pJ=1.2 * e_scale,
                energy_per_write_pJ=1.2 * e_scale,
                num_banks=2, bank_width_bytes=8,
            ),
            # -- L1: ADCU --
            "L1_ADCU": MemoryLevel(
                name="L1 ADCU Local SRAM",
                capacity_bytes=8 * 1024,
                bandwidth_bytes_per_cycle=16,
                read_latency_cycles=1, write_latency_cycles=1,
                energy_per_read_pJ=1.5 * e_scale,
                energy_per_write_pJ=1.5 * e_scale,
                num_banks=4, bank_width_bytes=8,
            ),
            # -- L1: FU --
            "L1_FU": MemoryLevel(
                name="L1 FU Local SRAM",
                capacity_bytes=16 * 1024,
                bandwidth_bytes_per_cycle=32,
                read_latency_cycles=1, write_latency_cycles=1,
                energy_per_read_pJ=2.0 * e_scale,
                energy_per_write_pJ=2.0 * e_scale,
                num_banks=8, bank_width_bytes=8,
            ),
            # -- L2: Global shared SRAM --
            "L2_Global": MemoryLevel(
                name="L2 Global Shared SRAM",
                capacity_bytes=512 * 1024,
                bandwidth_bytes_per_cycle=64,
                read_latency_cycles=3, write_latency_cycles=3,
                energy_per_read_pJ=10.0 * e_scale,
                energy_per_write_pJ=10.0 * e_scale,
                num_banks=32, bank_width_bytes=8,
            ),
            # -- L3: External DRAM --
            "L3_DRAM": MemoryLevel(
                name="L3 External DRAM (LPDDR4x)",
                capacity_bytes=2 * 1024 * 1024 * 1024,
                bandwidth_bytes_per_cycle=4,
                read_latency_cycles=50, write_latency_cycles=50,
                energy_per_read_pJ=200.0,
                energy_per_write_pJ=200.0,
                num_banks=4, bank_width_bytes=16,
            ),
        }

    # -- Spec C: L2 budget validation ----------------------------------------

    def validate_l2_budget(self, allocations: dict[str, int]) -> dict[str, Any]:
        """
        Validate that concurrent L2 buffer allocations do not exceed capacity.

        Parameters
        ----------
        allocations : dict mapping buffer_name -> size_bytes

        Returns
        -------
        dict with 'ok' (bool), 'total_bytes', 'capacity_bytes', 'detail'
        """
        l2 = self.levels["L2_Global"]
        total = sum(allocations.values())
        return {
            "ok": total <= l2.capacity_bytes,
            "total_bytes": total,
            "capacity_bytes": l2.capacity_bytes,
            "utilization_pct": round(100.0 * total / l2.capacity_bytes, 1),
            "headroom_bytes": l2.capacity_bytes - total,
            "allocations": allocations,
        }

    def spec_c_worst_case_budget(self) -> dict[str, Any]:
        """
        Validate the worst-case concurrent L2 budget from Spec C.

        This is the peak L2 usage when SA, GSU, FU, and DMA are all active.
        """
        allocations = {
            "sa_weight_double_buffer": 64 * 1024,
            "sa_activation_tile": 48 * 1024,
            "gsu_compact_chunk": 24 * 1024,
            "decoder_dual_path_buffers": 128 * 1024,
            "fu_strip_buffers": 200 * 1024,
            "dma_staging": 32 * 1024,
        }
        return self.validate_l2_budget(allocations)

    # -- Data placement (updated for SA + SCU architecture) ------------------

    def data_placement_strategy(self) -> dict[str, Any]:
        return {
            "L0_RegFile": {
                "contents": ["MAC operands", "partial sums", "flash attention running stats"],
                "lifetime": "within one tile",
            },
            "L1_SA": {
                "contents": ["weight tile (double-buffered)", "activation tile",
                             "attention score tile (Br x Bc)", "output tile"],
                "lifetime": "within one block/layer",
                "double_buffered": True,
            },
            "L1_CRM": {
                "contents": ["member_to_rep_local LUT (2.7KB)", "rep coordinates"],
                "lifetime": "CRM done -> last scatter (all 12 blocks)",
            },
            "L1_GSU": {
                "contents": ["index table (from CRM)", "burst buffer"],
                "lifetime": "per gather/scatter operation",
            },
            "L1_DPC": {
                "contents": ["histogram SRAM (1KB)", "mask fragment"],
                "lifetime": "per decoder stage",
            },
            "L1_ADCU": {
                "contents": ["reciprocal LUT (8KB, persistent)", "keypoint buffers (per-frame)"],
                "lifetime": "persistent (LUT) / per-frame (keypoints)",
            },
            "L1_FU": {
                "contents": ["strip input buffers", "bilateral LUT", "intermediate strip"],
                "lifetime": "per strip / persistent (LUT)",
            },
            "L2_Global": {
                "contents": [
                    "SA weight prefetch (2x32KB double-buf)",
                    "flash attention Q/KV tiles (24-48KB)",
                    "GSU compact token chunks (24KB)",
                    "decoder dual-path HP/LP buffers (2 x tile_size)",
                    "FU strip buffers (3 x 66KB)",
                    "DMA staging (32KB)",
                ],
                "streaming_rules": [
                    "Token sequence: flash-tiled, never fully materialized",
                    "Encoder taps: spill to DRAM, reload per decoder stage",
                    "Decoder dual-path: HP buf A + LP buf B, blend overwrites A",
                    "FU fusion: 32-row strips, max 3 strips concurrent",
                    "SA weights: only double-buffered L2 allocation",
                ],
                "lifetime": "multi-layer / multi-stage",
            },
            "L3_DRAM": {
                "contents": [
                    "model weights (~25MB INT8)",
                    "input image + SGM disp + confidence (external, ~3MB)",
                    "encoder intermediate features (4 taps, ~2.1MB)",
                    "output disparity frame",
                ],
                "lifetime": "persistent / per-frame",
            },
        }

    # -- Area estimation -----------------------------------------------------

    def total_sram_area_mm2(self) -> dict[str, float]:
        scale = (self.node / 28) ** 2
        sram_density = 0.001 * scale  # mm^2 per KB at 28nm
        areas = {}
        total = 0.0
        for name, level in self.levels.items():
            if "DRAM" not in name:
                kb = level.capacity_bytes / 1024
                area = kb * sram_density
                areas[name] = round(area, 5)
                total += area
        areas["total"] = round(total, 4)
        return areas

    def total_l1_scu_bytes(self) -> int:
        """Total L1 SRAM across all SCUs."""
        scu_keys = ["L1_CRM", "L1_GSU", "L1_DPC", "L1_ADCU", "L1_FU"]
        return sum(self.levels[k].capacity_bytes for k in scu_keys)

    # -- Print ---------------------------------------------------------------

    def print_spec(self) -> None:
        print(f"\n=== Memory Hierarchy ({self.node}nm) ===\n")
        header = f"{'Level':<28} {'Capacity':>10} {'BW(B/cyc)':>10} {'Latency':>8} {'E_rd(pJ)':>9} {'Banks':>6}"
        print(header)
        print("-" * len(header))

        for name, level in self.levels.items():
            cap = level.capacity_bytes
            if cap >= 1024 ** 3:
                cap_str = f"{cap / 1024**3:.0f} GB"
            elif cap >= 1024 ** 2:
                cap_str = f"{cap / 1024**2:.0f} MB"
            elif cap >= 1024:
                cap_str = f"{cap / 1024:.0f} KB"
            else:
                cap_str = f"{cap} B"

            print(f"  {level.name:<26} {cap_str:>10} {level.bandwidth_bytes_per_cycle:>10} "
                  f"{level.read_latency_cycles:>8} {level.energy_per_read_pJ:>9.1f} {level.num_banks:>6}")
