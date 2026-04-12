"""
EdgeStereoDAv2 Cycle-Accurate Simulator
Event-driven simulation engine with cycle-level timing
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import math


class ModuleState(Enum):
    IDLE = 0
    FETCH = 1
    COMPUTE = 2
    WRITEBACK = 3
    STALL = 4


@dataclass
class Event:
    """Simulation event"""
    cycle: int
    module: str
    action: str
    data: dict = field(default_factory=dict)


@dataclass
class SimStats:
    """Per-module simulation statistics"""
    busy_cycles: int = 0
    idle_cycles: int = 0
    stall_cycles: int = 0
    fetch_cycles: int = 0
    compute_cycles: int = 0
    writeback_cycles: int = 0
    total_macs: int = 0
    memory_reads: int = 0
    memory_writes: int = 0

    @property
    def total_cycles(self):
        return self.busy_cycles + self.idle_cycles + self.stall_cycles

    @property
    def utilization(self):
        return self.compute_cycles / max(self.total_cycles, 1)


class HardwareModule:
    """Base class for hardware modules"""

    def __init__(self, name: str, num_macs: int = 0):
        self.name = name
        self.num_macs = num_macs
        self.state = ModuleState.IDLE
        self.stats = SimStats()
        self.current_op = None
        self.remaining_cycles = 0

    def tick(self, cycle: int):
        """Called every clock cycle"""
        if self.state == ModuleState.IDLE:
            self.stats.idle_cycles += 1
        elif self.state == ModuleState.FETCH:
            self.stats.fetch_cycles += 1
            self.stats.busy_cycles += 1
            self.remaining_cycles -= 1
            if self.remaining_cycles <= 0:
                self.state = ModuleState.COMPUTE
        elif self.state == ModuleState.COMPUTE:
            self.stats.compute_cycles += 1
            self.stats.busy_cycles += 1
            self.remaining_cycles -= 1
            if self.remaining_cycles <= 0:
                self.state = ModuleState.WRITEBACK
        elif self.state == ModuleState.WRITEBACK:
            self.stats.writeback_cycles += 1
            self.stats.busy_cycles += 1
            self.remaining_cycles -= 1
            if self.remaining_cycles <= 0:
                self.state = ModuleState.IDLE
                self.current_op = None
        elif self.state == ModuleState.STALL:
            self.stats.stall_cycles += 1

    def start_operation(self, op_name: str, fetch_cycles: int,
                         compute_cycles: int, writeback_cycles: int,
                         macs: int = 0, mem_reads: int = 0, mem_writes: int = 0):
        """Start a new operation"""
        self.current_op = op_name
        self.state = ModuleState.FETCH
        self._fetch_remaining = fetch_cycles
        self._compute_remaining = compute_cycles
        self._writeback_remaining = writeback_cycles
        self.remaining_cycles = fetch_cycles
        self.stats.total_macs += macs
        self.stats.memory_reads += mem_reads
        self.stats.memory_writes += mem_writes

    def tick(self, cycle: int):
        """Cycle-accurate tick with state transitions"""
        if self.state == ModuleState.IDLE:
            self.stats.idle_cycles += 1
        elif self.state == ModuleState.FETCH:
            self.stats.fetch_cycles += 1
            self.stats.busy_cycles += 1
            self._fetch_remaining -= 1
            if self._fetch_remaining <= 0:
                self.state = ModuleState.COMPUTE
                self.remaining_cycles = self._compute_remaining
        elif self.state == ModuleState.COMPUTE:
            self.stats.compute_cycles += 1
            self.stats.busy_cycles += 1
            self.remaining_cycles -= 1
            if self.remaining_cycles <= 0:
                self.state = ModuleState.WRITEBACK
                self.remaining_cycles = self._writeback_remaining
        elif self.state == ModuleState.WRITEBACK:
            self.stats.writeback_cycles += 1
            self.stats.busy_cycles += 1
            self.remaining_cycles -= 1
            if self.remaining_cycles <= 0:
                self.state = ModuleState.IDLE
                self.current_op = None
        elif self.state == ModuleState.STALL:
            self.stats.stall_cycles += 1

    @property
    def is_idle(self):
        return self.state == ModuleState.IDLE


@dataclass
class SparsityConfig:
    """Token sparsity configuration for SGM-guided pruning"""
    prune_ratio: float = 0.0      # Fraction of tokens pruned (0.0 = dense, 0.186 = 18.6%)
    prune_layer: int = 0          # First ViT block that uses GAS sparse attention (0-based)
    gather_scatter_cycles: int = 2  # Overhead cycles per sparse block for index select

    @property
    def keep_ratio(self):
        return 1.0 - self.prune_ratio


@dataclass
class SimConfig:
    """Simulator configuration"""
    clock_freq_mhz: int = 500
    pe_rows: int = 32
    pe_cols: int = 32
    l1_size_bytes: int = 64 * 1024
    l2_size_bytes: int = 512 * 1024
    l2_bandwidth_bytes_per_cycle: int = 64
    dram_bandwidth_bytes_per_cycle: int = 4
    dram_latency_cycles: int = 50
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)

    @property
    def num_macs(self):
        return self.pe_rows * self.pe_cols


class CycleAccurateSimulator:
    """
    Main simulation engine.

    Simulates the EdgeStereoDAv2 accelerator at cycle level:
    - VFE: processes encoder transformer blocks
    - CSFE: processes decoder convolutions
    - ADCU: processes depth-to-disparity conversion
    - Memory: tracks bandwidth utilization
    """

    def __init__(self, config: SimConfig = None):
        self.config = config or SimConfig()
        self.cycle = 0

        # Hardware modules
        self.vfe = HardwareModule('VFE', self.config.num_macs)
        self.csfe = HardwareModule('CSFE', min(self.config.num_macs, 256))
        self.adcu = HardwareModule('ADCU', 32)

        # Memory bandwidth tracking
        self.mem_reads_per_cycle = []
        self.mem_writes_per_cycle = []

        # Operation queue
        self.op_queue: List[dict] = []
        self.completed_ops: List[dict] = []

    def build_workload(self, img_h: int = 518, img_w: int = 518,
                        embed_dim: int = 384, num_heads: int = 6,
                        depth: int = 12, decoder_features: int = 64):
        """Build the operation schedule for one frame"""
        patch_size = 14
        num_patches_h = img_h // patch_size
        num_patches_w = img_w // patch_size
        seq_len = num_patches_h * num_patches_w + 1
        head_dim = embed_dim // num_heads
        mlp_hidden = embed_dim * 4

        ops = []

        # Patch embedding
        pe_flops = num_patches_h * num_patches_w * patch_size**2 * 3 * embed_dim * 2
        ops.append({
            'name': 'patch_embed',
            'engine': 'VFE',
            'flops': pe_flops,
            'weight_bytes': patch_size**2 * 3 * embed_dim,
            'input_bytes': img_h * img_w * 3,
            'output_bytes': seq_len * embed_dim,
        })

        # Sparsity config
        sp = self.config.sparsity
        cru_cycles = 1  # Confidence Router Unit: 1 cycle for threshold compare

        # CRU operation (if sparsity enabled)
        if sp.prune_ratio > 0:
            ops.append({
                'name': 'cru_token_routing',
                'engine': 'VFE',
                'flops': seq_len * 2,  # 1 compare + 1 index per token
                'weight_bytes': 0,
                'input_bytes': seq_len * 4,  # confidence grid (float32)
                'output_bytes': seq_len,      # keep/prune mask (1 byte each)
            })

        # Encoder blocks
        for i in range(depth):
            # Determine effective sequence length for this block
            is_sparse = sp.prune_ratio > 0 and i >= sp.prune_layer
            M = int(seq_len * sp.keep_ratio) if is_sparse else seq_len
            sparse_tag = f' [sparse M={M}]' if is_sparse else ''

            # Gather overhead for sparse blocks
            if is_sparse:
                ops.append({
                    'name': f'block_{i}_gather',
                    'engine': 'VFE',
                    'flops': M * embed_dim,  # index-select
                    'weight_bytes': 0,
                    'input_bytes': M * 4,  # index buffer
                    'output_bytes': M * embed_dim,
                })

            # QKV projection: operates on M tokens (kept tokens only for sparse)
            qkv_flops = M * embed_dim * 3 * embed_dim * 2
            ops.append({
                'name': f'block_{i}_qkv{sparse_tag}',
                'engine': 'VFE',
                'flops': qkv_flops,
                'weight_bytes': embed_dim * 3 * embed_dim,
                'input_bytes': M * embed_dim,
                'output_bytes': M * 3 * embed_dim,
            })

            # Attention (Q×K^T, softmax, attn×V) — M×M instead of N×N for sparse
            attn_flops = num_heads * (
                M * M * head_dim * 2 +  # Q×K^T
                M * M * 5 +              # softmax
                M * head_dim * M * 2     # attn×V
            )
            br = 64
            bc = 128
            num_tiles = math.ceil(M / br) * math.ceil(M / bc) * num_heads

            ops.append({
                'name': f'block_{i}_attention{sparse_tag}',
                'engine': 'VFE',
                'flops': attn_flops,
                'weight_bytes': 0,
                'input_bytes': M * embed_dim * 2,  # Q, K, V from L1
                'output_bytes': M * embed_dim,
            })

            # Output projection: M tokens
            ops.append({
                'name': f'block_{i}_out_proj{sparse_tag}',
                'engine': 'VFE',
                'flops': M * embed_dim * embed_dim * 2,
                'weight_bytes': embed_dim * embed_dim,
                'input_bytes': M * embed_dim,
                'output_bytes': M * embed_dim,
            })

            # Scatter for sparse blocks (write back to full sequence)
            if is_sparse:
                ops.append({
                    'name': f'block_{i}_scatter',
                    'engine': 'VFE',
                    'flops': M * embed_dim,
                    'weight_bytes': 0,
                    'input_bytes': M * embed_dim,
                    'output_bytes': M * embed_dim,
                })

            # MLP: always operates on ALL tokens (per-token, no cross-token dependency)
            ops.append({
                'name': f'block_{i}_mlp',
                'engine': 'VFE',
                'flops': seq_len * (embed_dim * mlp_hidden + mlp_hidden * embed_dim) * 2,
                'weight_bytes': embed_dim * mlp_hidden + mlp_hidden * embed_dim,
                'input_bytes': seq_len * embed_dim,
                'output_bytes': seq_len * embed_dim,
            })

        # Decoder stages
        decoder_sizes = [
            (num_patches_h, num_patches_w),
            (num_patches_h * 2, num_patches_w * 2),
            (num_patches_h * 4, num_patches_w * 4),
            (num_patches_h * 8, num_patches_w * 8),
        ]
        F = decoder_features

        for stage, (dh, dw) in enumerate(decoder_sizes):
            # 1x1 projection
            ops.append({
                'name': f'decoder_proj_{stage}',
                'engine': 'CSFE',
                'flops': dh * dw * embed_dim * F * 2,
                'weight_bytes': embed_dim * F,
                'input_bytes': dh * dw * embed_dim,
                'output_bytes': dh * dw * F,
            })

            # 2 × RCU (each: 2 × 3×3 conv)
            rcu_flops = 2 * 2 * dh * dw * F * F * 9 * 2
            ops.append({
                'name': f'decoder_rcu_{stage}',
                'engine': 'CSFE',
                'flops': rcu_flops,
                'weight_bytes': 2 * 2 * F * F * 9,
                'input_bytes': dh * dw * F,
                'output_bytes': dh * dw * F,
            })

            # Upsample 2x
            ops.append({
                'name': f'decoder_upsample_{stage}',
                'engine': 'CSFE',
                'flops': dh * dw * F * 4,
                'weight_bytes': 0,
                'input_bytes': dh * dw * F,
                'output_bytes': dh * 2 * dw * 2 * F,
            })

        # ADCU
        ops.append({
            'name': 'adcu_sparse_match',
            'engine': 'ADCU',
            'flops': 32 * 192 * 11 * 11 * 5,
            'weight_bytes': 0,
            'input_bytes': 32 * 11 * 11 * 2,
            'output_bytes': 32 * 4,
        })

        ops.append({
            'name': 'adcu_depth_to_disp',
            'engine': 'ADCU',
            'flops': img_h * img_w * 5,
            'weight_bytes': 4096 * 2,
            'input_bytes': img_h * img_w * 2,
            'output_bytes': img_h * img_w * 2,
        })

        self.op_queue = ops
        return ops

    def simulate_frame(self) -> dict:
        """Simulate one complete frame (batch-mode for performance)."""
        total_cycles = 0

        modules = {'VFE': self.vfe, 'CSFE': self.csfe, 'ADCU': self.adcu}
        num_macs = {'VFE': self.config.num_macs, 'CSFE': min(self.config.num_macs, 256), 'ADCU': 32}

        for op in self.op_queue:
            engine = op['engine']
            module = modules[engine]
            n_macs = num_macs[engine]

            # Fetch: load weights/inputs from L2/DRAM
            fetch_cycles = max(
                op['weight_bytes'] // self.config.l2_bandwidth_bytes_per_cycle,
                op['input_bytes'] // self.config.l2_bandwidth_bytes_per_cycle,
                1
            )

            # Compute: FLOPs / (MACs × 2)
            utilization = 0.85 if engine == 'VFE' else 0.75 if engine == 'CSFE' else 0.90
            compute_cycles = max(int(op['flops'] / (n_macs * 2) / utilization), 1)

            # Writeback
            writeback_cycles = max(op['output_bytes'] // self.config.l2_bandwidth_bytes_per_cycle, 1)

            # Pipeline: overlap fetch with compute
            effective_cycles = max(compute_cycles, fetch_cycles) + writeback_cycles

            # Batch-update stats (no per-cycle tick for performance)
            module.stats.fetch_cycles += fetch_cycles
            module.stats.compute_cycles += compute_cycles
            module.stats.writeback_cycles += writeback_cycles
            module.stats.busy_cycles += effective_cycles
            module.stats.total_macs += op['flops'] // 2
            module.stats.memory_reads += op['weight_bytes'] + op['input_bytes']
            module.stats.memory_writes += op['output_bytes']

            total_cycles += effective_cycles
            self.completed_ops.append({
                **op,
                'fetch_cycles': fetch_cycles,
                'compute_cycles': compute_cycles,
                'writeback_cycles': writeback_cycles,
                'effective_cycles': effective_cycles,
                'start_cycle': total_cycles - effective_cycles,
                'end_cycle': total_cycles,
            })

        return self._collect_results(total_cycles)

    def _collect_results(self, total_cycles: int) -> dict:
        """Collect simulation results"""
        freq = self.config.clock_freq_mhz * 1e6
        latency_s = total_cycles / freq
        fps = 1.0 / latency_s if latency_s > 0 else 0

        # Per-engine breakdown
        engine_cycles = {'VFE': 0, 'CSFE': 0, 'ADCU': 0}
        engine_flops = {'VFE': 0, 'CSFE': 0, 'ADCU': 0}

        for op in self.completed_ops:
            engine_cycles[op['engine']] += op['effective_cycles']
            engine_flops[op['engine']] += op['flops']

        total_flops = sum(engine_flops.values())

        sp = self.config.sparsity
        return {
            'total_cycles': total_cycles,
            'latency_ms': latency_s * 1000,
            'fps': fps,
            'total_flops': total_flops,
            'sparsity': {
                'prune_ratio': sp.prune_ratio,
                'prune_layer': sp.prune_layer,
                'keep_ratio': sp.keep_ratio,
                'enabled': sp.prune_ratio > 0,
            },
            'engine_breakdown': {
                eng: {
                    'cycles': engine_cycles[eng],
                    'fraction': engine_cycles[eng] / total_cycles if total_cycles > 0 else 0,
                    'flops': engine_flops[eng],
                }
                for eng in engine_cycles
            },
            'vfe_stats': {
                'compute_cycles': self.vfe.stats.compute_cycles,
                'fetch_cycles': self.vfe.stats.fetch_cycles,
                'total_macs': self.vfe.stats.total_macs,
                'utilization': self.vfe.stats.utilization,
            },
            'operations': [
                {'name': op['name'], 'engine': op['engine'],
                 'cycles': op['effective_cycles'], 'flops': op['flops'],
                 'weight_bytes': op.get('weight_bytes', 0),
                 'input_bytes': op.get('input_bytes', 0),
                 'output_bytes': op.get('output_bytes', 0)}
                for op in self.completed_ops
            ],
        }


if __name__ == "__main__":
    print("=== Cycle-Accurate Simulator Test ===\n")

    for label, prune_ratio, prune_layer in [
        ("Dense (baseline)", 0.0, 0),
        ("Sparse (18.6% prune, layer=0)", 0.186, 0),
        ("Sparse (30% prune, layer=0)", 0.30, 0),
        ("Sparse (18.6% prune, layer=6)", 0.186, 6),
    ]:
        sp = SparsityConfig(prune_ratio=prune_ratio, prune_layer=prune_layer)
        config = SimConfig(clock_freq_mhz=500, pe_rows=32, pe_cols=32, sparsity=sp)
        sim = CycleAccurateSimulator(config)

        ops = sim.build_workload(img_h=518, img_w=518)
        total_flops = sum(op['flops'] for op in ops)
        results = sim.simulate_frame()

        print(f"--- {label} ---")
        print(f"  Ops: {len(ops)},  FLOPs: {total_flops/1e9:.2f}G,  "
              f"Cycles: {results['total_cycles']:,},  "
              f"Latency: {results['latency_ms']:.1f}ms,  FPS: {results['fps']:.2f}")

    # Detailed breakdown for sparse case
    print(f"\n--- Detailed: Sparse 18.6% prune, layer=0 ---")
    sp = SparsityConfig(prune_ratio=0.186, prune_layer=0)
    config = SimConfig(clock_freq_mhz=500, pe_rows=32, pe_cols=32, sparsity=sp)
    sim = CycleAccurateSimulator(config)
    sim.build_workload(img_h=518, img_w=518)
    results = sim.simulate_frame()

    for eng, info in results['engine_breakdown'].items():
        print(f"  {eng}: {info['cycles']:,} cycles ({info['fraction']*100:.1f}%)")

    sorted_ops = sorted(results['operations'], key=lambda x: x['cycles'], reverse=True)
    print(f"\n  Top 10 operations:")
    for op in sorted_ops[:10]:
        print(f"    {op['name']:<40} {op['cycles']:>10,} cycles")
