"""
Memory Hierarchy Design for EdgeStereoDAv2
Detailed specification of each memory level
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class MemoryLevel:
    """Specification for one level of the memory hierarchy"""
    name: str
    capacity_bytes: int
    bandwidth_bytes_per_cycle: int
    read_latency_cycles: int
    write_latency_cycles: int
    energy_per_read_pJ: float   # per 64-bit (8-byte) access
    energy_per_write_pJ: float
    num_banks: int
    bank_width_bytes: int


class MemoryHierarchySpec:
    """Complete memory hierarchy specification"""

    def __init__(self, process_node_nm: int = 28):
        self.node = process_node_nm
        self.levels = self._build_hierarchy()

    def _build_hierarchy(self) -> Dict[str, MemoryLevel]:
        # Energy scaling: approximately (node/28)^(-0.7) from CACTI models
        e_scale = (self.node / 28) ** (-0.7) if self.node != 28 else 1.0

        return {
            'L0_RegFile': MemoryLevel(
                name='L0 Register File',
                capacity_bytes=512,  # Per PE
                bandwidth_bytes_per_cycle=64,
                read_latency_cycles=0,
                write_latency_cycles=0,
                energy_per_read_pJ=0.05 * e_scale,
                energy_per_write_pJ=0.05 * e_scale,
                num_banks=1,
                bank_width_bytes=64,
            ),
            'L1_VFE': MemoryLevel(
                name='L1 VFE Local SRAM',
                capacity_bytes=64 * 1024,
                bandwidth_bytes_per_cycle=32,  # 256 bits/cycle
                read_latency_cycles=1,
                write_latency_cycles=1,
                energy_per_read_pJ=2.0 * e_scale,
                energy_per_write_pJ=2.0 * e_scale,
                num_banks=16,
                bank_width_bytes=8,
            ),
            'L1_CSFE': MemoryLevel(
                name='L1 CSFE Local SRAM',
                capacity_bytes=64 * 1024,
                bandwidth_bytes_per_cycle=32,
                read_latency_cycles=1,
                write_latency_cycles=1,
                energy_per_read_pJ=2.0 * e_scale,
                energy_per_write_pJ=2.0 * e_scale,
                num_banks=16,
                bank_width_bytes=8,
            ),
            'L1_ADCU': MemoryLevel(
                name='L1 ADCU Local SRAM',
                capacity_bytes=8 * 1024,
                bandwidth_bytes_per_cycle=16,
                read_latency_cycles=1,
                write_latency_cycles=1,
                energy_per_read_pJ=1.5 * e_scale,
                energy_per_write_pJ=1.5 * e_scale,
                num_banks=4,
                bank_width_bytes=8,
            ),
            'L2_Global': MemoryLevel(
                name='L2 Global Shared SRAM',
                capacity_bytes=512 * 1024,
                bandwidth_bytes_per_cycle=64,  # 512 bits/cycle
                read_latency_cycles=3,
                write_latency_cycles=3,
                energy_per_read_pJ=10.0 * e_scale,
                energy_per_write_pJ=10.0 * e_scale,
                num_banks=32,
                bank_width_bytes=8,
            ),
            'L3_DRAM': MemoryLevel(
                name='L3 External DRAM (LPDDR4x)',
                capacity_bytes=2 * 1024 * 1024 * 1024,
                bandwidth_bytes_per_cycle=4,  # ~16 GB/s at 500MHz = 32 bytes/cycle but shared
                read_latency_cycles=50,
                write_latency_cycles=50,
                energy_per_read_pJ=200.0,  # Relatively constant across nodes
                energy_per_write_pJ=200.0,
                num_banks=4,
                bank_width_bytes=16,
            ),
        }

    def print_spec(self):
        """Print memory hierarchy specification"""
        print(f"\n=== Memory Hierarchy ({self.node}nm) ===\n")
        print(f"{'Level':<20} {'Capacity':>10} {'BW(B/cyc)':>10} {'Latency':>8} {'E_read(pJ)':>10} {'Banks':>6}")
        print("-" * 70)

        for name, level in self.levels.items():
            cap = level.capacity_bytes
            if cap >= 1024 * 1024 * 1024:
                cap_str = f"{cap / (1024**3):.0f} GB"
            elif cap >= 1024 * 1024:
                cap_str = f"{cap / 1024**2:.0f} MB"
            elif cap >= 1024:
                cap_str = f"{cap / 1024:.0f} KB"
            else:
                cap_str = f"{cap} B"

            print(f"  {level.name:<18} {cap_str:>10} {level.bandwidth_bytes_per_cycle:>10} "
                  f"{level.read_latency_cycles:>8} {level.energy_per_read_pJ:>10.1f} {level.num_banks:>6}")

    def data_placement_strategy(self) -> dict:
        """Define what data goes where in the hierarchy"""
        return {
            'L0_RegFile': {
                'contents': [
                    'MAC operands (weight, activation)',
                    'Partial sum accumulators',
                    'Flash attention running max/sum stats',
                ],
                'lifetime': 'Within one tile computation',
            },
            'L1_VFE': {
                'contents': [
                    'Current weight tile (QKV, MLP)',
                    'Input activation tile',
                    'Attention score tile (Br×Bc)',
                    'Output tile before writeback',
                ],
                'lifetime': 'Within one layer/block processing',
                'double_buffered': True,
            },
            'L1_CSFE': {
                'contents': [
                    'Convolution weight kernel (3×3×C)',
                    'Input feature tile + halo',
                    'Output feature tile',
                ],
                'lifetime': 'Within one convolution layer',
                'double_buffered': True,
            },
            'L1_ADCU': {
                'contents': [
                    'Reciprocal LUT (8KB)',
                    'Sparse keypoint patches',
                    'Scale-shift computation buffers',
                ],
                'lifetime': 'Persistent (LUT) / per-frame (patches)',
            },
            'L2_Global': {
                'contents': [
                    'Encoder intermediate features (for DPT extraction)',
                    'Weight prefetch buffer',
                    'Decoder feature maps between fusion stages',
                    'DMA staging area',
                ],
                'lifetime': 'Multi-layer / multi-stage',
                'allocation': {
                    'weight_buffer': '128 KB',
                    'feature_buffer': '256 KB',
                    'dma_staging': '128 KB',
                },
            },
            'L3_DRAM': {
                'contents': [
                    'Full model weights (~25 MB INT8)',
                    'Input image frame buffer',
                    'Output disparity frame buffer',
                    'Intermediate results overflow',
                ],
                'lifetime': 'Persistent / per-frame',
            },
        }

    def total_sram_area_mm2(self) -> dict:
        """Estimate SRAM area"""
        scale = (self.node / 28) ** 2

        # Approximate: 1 KB SRAM ≈ 0.001 mm² at 28nm
        sram_density = 0.001 * scale  # mm² per KB

        areas = {}
        total = 0
        for name, level in self.levels.items():
            if 'DRAM' not in name:
                kb = level.capacity_bytes / 1024
                area = kb * sram_density
                areas[name] = area
                total += area

        areas['total'] = total
        return areas


if __name__ == "__main__":
    for node in [28, 7]:
        spec = MemoryHierarchySpec(process_node_nm=node)
        spec.print_spec()

        areas = spec.total_sram_area_mm2()
        print(f"\n  SRAM Area ({node}nm):")
        for name, area in areas.items():
            print(f"    {name}: {area:.4f} mm²")

    # Data placement
    spec28 = MemoryHierarchySpec(28)
    placement = spec28.data_placement_strategy()
    print(f"\n=== Data Placement Strategy ===")
    for level, info in placement.items():
        print(f"\n  {level}:")
        print(f"    Contents: {info['contents']}")
        print(f"    Lifetime: {info['lifetime']}")
