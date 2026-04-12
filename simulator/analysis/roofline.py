"""
Roofline Model Analysis
Maps each operation onto the roofline to identify bottlenecks
"""

import sys
import numpy as np
sys.path.insert(0, 'G:/workspace')


class RooflineModel:
    """
    Roofline model for the EdgeStereoDAv2 accelerator.

    Two ceilings:
    - Compute ceiling: peak_TOPS
    - Memory bandwidth ceiling: memory_BW_GBps

    Ridge point: peak_TOPS / memory_BW = arithmetic intensity threshold
    """

    def __init__(self, peak_tops: float, memory_bw_gbps: float):
        self.peak_tops = peak_tops
        self.peak_gops = peak_tops * 1000  # GOPS
        self.memory_bw = memory_bw_gbps  # GB/s
        self.ridge_point = self.peak_gops / self.memory_bw  # OPS/Byte

    def attainable_gops(self, arithmetic_intensity: float) -> float:
        """Compute attainable GOPS for given arithmetic intensity"""
        return min(self.peak_gops, arithmetic_intensity * self.memory_bw)

    def is_compute_bound(self, arithmetic_intensity: float) -> bool:
        return arithmetic_intensity >= self.ridge_point

    def analyze_operations(self) -> list:
        """Analyze all operations in the DAv2 pipeline"""
        N = 1370
        D = 384
        H = 6
        d = 64
        M = 1536
        F = 64

        operations = [
            {
                'name': 'QKV Projection',
                'flops': N * D * 3 * D * 2,
                'bytes': N * D + D * 3 * D + N * 3 * D,  # input + weight + output
            },
            {
                'name': 'Attention Q*K^T',
                'flops': H * N * N * d * 2,
                'bytes': N * D + H * N * N * 2,  # Q,K input + attention output (INT16)
            },
            {
                'name': 'Softmax',
                'flops': H * N * N * 5,
                'bytes': H * N * N * 2 * 2,  # Read + write attention scores
            },
            {
                'name': 'Attention * V',
                'flops': H * N * d * N * 2,
                'bytes': H * N * N * 2 + N * D + N * D,
            },
            {
                'name': 'Output Projection',
                'flops': N * D * D * 2,
                'bytes': N * D + D * D + N * D,
            },
            {
                'name': 'MLP Layer 1',
                'flops': N * D * M * 2,
                'bytes': N * D + D * M + N * M,
            },
            {
                'name': 'MLP Layer 2',
                'flops': N * M * D * 2,
                'bytes': N * M + M * D + N * D,
            },
            {
                'name': 'LayerNorm',
                'flops': N * D * 5,
                'bytes': N * D * 2,  # Read + write
            },
            {
                'name': 'GELU',
                'flops': N * M * 10,
                'bytes': N * M * 2,
            },
            {
                'name': 'Decoder 1x1 Conv (37x37)',
                'flops': 37 * 37 * D * F * 2,
                'bytes': 37 * 37 * D + D * F + 37 * 37 * F,
            },
            {
                'name': 'Decoder 3x3 Conv (74x74)',
                'flops': 74 * 74 * F * F * 9 * 2,
                'bytes': 76 * 76 * F + F * F * 9 + 74 * 74 * F,
            },
            {
                'name': 'Decoder 3x3 Conv (296x296)',
                'flops': 296 * 296 * F * F * 9 * 2,
                'bytes': 298 * 298 * F + F * F * 9 + 296 * 296 * F,
            },
            {
                'name': 'Bilinear Upsample',
                'flops': 74 * 74 * F * 4,
                'bytes': 37 * 37 * F + 74 * 74 * F,
            },
            {
                'name': 'ADCU Depth-to-Disp',
                'flops': 518 * 518 * 5,
                'bytes': 518 * 518 * 2 * 2 + 4096 * 2,
            },
        ]

        # Compute arithmetic intensity
        for op in operations:
            op['arithmetic_intensity'] = op['flops'] / op['bytes']
            op['attainable_gops'] = self.attainable_gops(op['arithmetic_intensity'])
            op['bound'] = 'Compute' if self.is_compute_bound(op['arithmetic_intensity']) else 'Memory'
            op['utilization'] = op['attainable_gops'] / self.peak_gops

        return operations


if __name__ == "__main__":
    print("=== Roofline Model Analysis ===\n")

    # 28nm/500MHz config: 1024 MACs, 500MHz → 1.024 TOPS, ~16 GB/s DRAM + on-chip
    # Effective bandwidth considering L1/L2: ~64 GB/s (on-chip dominated)
    model = RooflineModel(peak_tops=1.024, memory_bw_gbps=64)

    print(f"Peak compute: {model.peak_gops:.0f} GOPS")
    print(f"Memory bandwidth: {model.memory_bw:.0f} GB/s")
    print(f"Ridge point: {model.ridge_point:.1f} OPS/Byte")

    ops = model.analyze_operations()

    print(f"\n{'Operation':<30} {'AI(OPS/B)':>10} {'Attain(GOPS)':>12} {'Util':>6} {'Bound':<10}")
    print("-" * 75)
    for op in ops:
        print(f"  {op['name']:<28} {op['arithmetic_intensity']:>10.1f} "
              f"{op['attainable_gops']:>12.0f} {op['utilization']*100:>5.0f}% {op['bound']:<10}")

    # Summary
    compute_ops = [op for op in ops if op['bound'] == 'Compute']
    memory_ops = [op for op in ops if op['bound'] == 'Memory']
    print(f"\nCompute-bound operations: {len(compute_ops)}")
    print(f"Memory-bound operations: {len(memory_ops)}")
