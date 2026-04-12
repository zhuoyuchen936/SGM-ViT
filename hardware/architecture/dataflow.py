"""
Dataflow Analysis for EdgeStereoDAv2
Defines per-engine dataflow strategies and data reuse analysis
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DataflowConfig:
    """Dataflow configuration for an operation"""
    name: str
    dataflow_type: str  # 'WS', 'OS', 'RS'
    tile_m: int  # Output dimension 1
    tile_n: int  # Output dimension 2
    tile_k: int  # Reduction dimension
    pe_rows: int = 32
    pe_cols: int = 32


class DataflowAnalyzer:
    """Analyzes data reuse and efficiency for different dataflow strategies"""

    def __init__(self, pe_rows: int = 32, pe_cols: int = 32):
        self.pe_rows = pe_rows
        self.pe_cols = pe_cols

    def weight_stationary_analysis(self, M: int, N: int, K: int,
                                     tile_m: int, tile_n: int, tile_k: int) -> dict:
        """
        Weight-Stationary dataflow analysis.
        Weights stay in PE registers; inputs and outputs flow through.

        Used for: MHSA projections, MLP layers
        """
        import math

        # Number of tiles
        nm = math.ceil(M / tile_m)
        nn = math.ceil(N / tile_n)
        nk = math.ceil(K / tile_k)

        # Weight reuse: each weight tile is loaded once from L1, used for all M-tiles
        weight_loads = nn * nk  # Each unique weight tile loaded once
        weight_total_bytes = nn * nk * tile_n * tile_k  # Total weight bytes from L1

        # Input reuse: each input tile is loaded once per N-tile
        input_loads = nm * nk * nn  # Worst case
        # With good scheduling (iterate N inner loop): nm * nk
        input_loads_opt = nm * nk
        input_total_bytes = input_loads_opt * tile_m * tile_k

        # Output: partial sums accumulated in output registers, written once
        output_writes = nm * nn
        output_total_bytes = output_writes * tile_m * tile_n

        # Compute
        total_macs = M * N * K
        cycles = nm * nn * nk * (tile_m * tile_n * tile_k) // (self.pe_rows * self.pe_cols)

        # Utilization
        useful_macs = M * N * K
        total_pe_cycles = cycles * self.pe_rows * self.pe_cols
        utilization = useful_macs / total_pe_cycles if total_pe_cycles > 0 else 0

        return {
            'dataflow': 'Weight-Stationary',
            'tiles': {'M': nm, 'N': nn, 'K': nk},
            'weight_bytes_from_l1': weight_total_bytes,
            'input_bytes_from_l1': input_total_bytes,
            'output_bytes_to_l1': output_total_bytes,
            'total_l1_traffic': weight_total_bytes + input_total_bytes + output_total_bytes,
            'total_macs': total_macs,
            'compute_cycles': cycles,
            'utilization': utilization,
            'arithmetic_intensity': total_macs * 2 / (weight_total_bytes + input_total_bytes + output_total_bytes),
        }

    def output_stationary_analysis(self, H: int, W: int, C_in: int,
                                     C_out: int, K: int = 3) -> dict:
        """
        Output-Stationary dataflow analysis for convolution.
        Output feature map stays in PE registers; weights and inputs flow.

        Used for: DPT decoder 3×3 convolutions
        """
        import math

        # Tile the output spatially and across channels
        tile_h = min(H, 8)
        tile_w = min(W, 8)
        tile_co = min(C_out, self.pe_cols)
        tile_ci = min(C_in, self.pe_rows)

        nh = math.ceil(H / tile_h)
        nw = math.ceil(W / tile_w)
        nco = math.ceil(C_out / tile_co)
        nci = math.ceil(C_in / tile_ci)

        # Input tile: (tile_h+K-1) × (tile_w+K-1) × tile_ci
        input_tile_size = (tile_h + K - 1) * (tile_w + K - 1) * tile_ci
        input_loads = nh * nw * nco * nci
        input_total = input_loads * input_tile_size

        # Weight tile: K × K × tile_ci × tile_co
        weight_tile_size = K * K * tile_ci * tile_co
        weight_loads = nco * nci  # Reused across spatial tiles
        # Actually need to reload for each spatial tile if weight doesn't fit
        weight_total = weight_loads * weight_tile_size * nh * nw

        # Output: accumulated across C_in tiles, written once per spatial tile
        output_tile_size = tile_h * tile_w * tile_co
        output_total = nh * nw * nco * output_tile_size

        total_macs = H * W * C_in * C_out * K * K
        cycles = total_macs // (self.pe_rows * self.pe_cols)

        return {
            'dataflow': 'Output-Stationary',
            'tiles': {'H': nh, 'W': nw, 'C_out': nco, 'C_in': nci},
            'tile_size': {'h': tile_h, 'w': tile_w, 'co': tile_co, 'ci': tile_ci},
            'input_bytes': input_total,
            'weight_bytes': weight_total,
            'output_bytes': output_total,
            'total_traffic': input_total + weight_total + output_total,
            'total_macs': total_macs,
            'compute_cycles': cycles,
        }

    def analyze_full_pipeline(self) -> Dict[str, dict]:
        """Analyze dataflow for the complete DAv2 pipeline"""
        N = 1370
        D = 384
        H_heads = 6
        d = 64
        M = 1536

        results = {}

        # Encoder operations (Weight-Stationary)
        # QKV projection: (N, D) × (D, 3D) → (N, 3D)
        results['qkv_projection'] = self.weight_stationary_analysis(
            N, 3*D, D, tile_m=64, tile_n=128, tile_k=64
        )

        # Attention Q×K^T: (N, d) × (d, N) per head → (N, N)
        results['attention_qkt'] = self.weight_stationary_analysis(
            N, N, d, tile_m=64, tile_n=128, tile_k=64
        )

        # Attention × V: (N, N) × (N, d) per head → (N, d)
        results['attention_v'] = self.weight_stationary_analysis(
            N, d, N, tile_m=64, tile_n=64, tile_k=128
        )

        # Out projection: (N, D) × (D, D) → (N, D)
        results['out_projection'] = self.weight_stationary_analysis(
            N, D, D, tile_m=64, tile_n=128, tile_k=64
        )

        # MLP layer 1: (N, D) × (D, M) → (N, M)
        results['mlp_w1'] = self.weight_stationary_analysis(
            N, M, D, tile_m=64, tile_n=128, tile_k=64
        )

        # MLP layer 2: (N, M) × (M, D) → (N, D)
        results['mlp_w2'] = self.weight_stationary_analysis(
            N, D, M, tile_m=64, tile_n=128, tile_k=128
        )

        # Decoder operations (Output-Stationary)
        for size, channels in [(37, 64), (74, 64), (148, 64), (296, 64)]:
            results[f'decoder_conv_{size}'] = self.output_stationary_analysis(
                size, size, channels, channels, 3
            )

        return results


if __name__ == "__main__":
    print("=== Dataflow Analysis ===\n")

    analyzer = DataflowAnalyzer(pe_rows=32, pe_cols=32)
    results = analyzer.analyze_full_pipeline()

    print(f"{'Operation':<25} {'Dataflow':<10} {'MACs':>12} {'Traffic':>12} {'AI':>8}")
    print("-" * 70)

    for name, r in results.items():
        dataflow = r['dataflow'][:2]
        macs = r['total_macs']
        traffic = r.get('total_l1_traffic', r.get('total_traffic', 0))
        ai = r.get('arithmetic_intensity', macs * 2 / (traffic + 1))
        print(f"  {name:<23} {dataflow:<10} {macs:>12,} {traffic:>12,} {ai:>8.1f}")

    # Per-block total for encoder
    encoder_ops = ['qkv_projection', 'attention_qkt', 'attention_v', 'out_projection', 'mlp_w1', 'mlp_w2']
    total_encoder_traffic = sum(results[op].get('total_l1_traffic', 0) for op in encoder_ops)
    total_encoder_macs = sum(results[op]['total_macs'] for op in encoder_ops)
    print(f"\n  Encoder per-block: {total_encoder_macs:,} MACs, {total_encoder_traffic:,} bytes L1 traffic")
    print(f"  Encoder 12 blocks: {total_encoder_macs*12:,} MACs, {total_encoder_traffic*12/1e6:.1f} MB")
