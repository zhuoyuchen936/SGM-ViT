"""
EdgeStereoDAv2 Top-Level Architecture
Complete hardware accelerator architecture description
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AcceleratorConfig:
    """Top-level accelerator configuration"""
    # Process technology
    process_node_nm: int = 28
    clock_freq_mhz: int = 500

    # PE Array
    pe_rows: int = 32
    pe_cols: int = 32
    mac_bitwidth: int = 8  # INT8 MAC
    num_macs: int = 1024  # pe_rows × pe_cols

    # Memory hierarchy
    l0_reg_bytes_per_pe: int = 512
    l1_sram_bytes_per_engine: int = 64 * 1024  # 64 KB
    l2_shared_sram_bytes: int = 512 * 1024     # 512 KB
    num_l1_banks: int = 16
    num_l2_banks: int = 32

    # Engines
    num_vfe_cores: int = 1  # ViT Feature Engine
    num_csfe_cores: int = 1  # Cross-Scale Fusion Engine
    num_adcu_cores: int = 1  # Absolute Disparity Calibration Unit

    # External memory
    dram_bandwidth_gbps: float = 16.0  # LPDDR4x
    dram_capacity_gb: int = 2

    # DMA
    dma_channels: int = 4
    dma_burst_bytes: int = 256

    @property
    def peak_tops(self) -> float:
        """Peak INT8 throughput in TOPS"""
        return self.num_macs * self.clock_freq_mhz * 1e6 * 2 / 1e12

    @property
    def total_sram_bytes(self) -> int:
        """Total on-chip SRAM"""
        l1_total = self.l1_sram_bytes_per_engine * 3  # VFE, CSFE, ADCU
        return l1_total + self.l2_shared_sram_bytes

    @property
    def total_sram_kb(self) -> float:
        return self.total_sram_bytes / 1024


class ViTFeatureEngine:
    """
    ViT Feature Engine (VFE)
    Processes DINOv2 encoder transformer blocks.

    Components:
    - Reconfigurable MAC array for MHSA and MLP matrix multiplications
    - Hardware LayerNorm unit
    - Piecewise-linear softmax approximation
    - GELU approximation unit
    - Flash Attention tiling controller
    """

    def __init__(self, config: AcceleratorConfig):
        self.config = config

    def describe(self) -> dict:
        return {
            'name': 'ViT Feature Engine (VFE)',
            'function': 'Processes transformer encoder blocks (MHSA + MLP)',
            'components': {
                'mac_array': {
                    'size': f'{self.config.pe_rows}×{self.config.pe_cols}',
                    'type': f'INT{self.config.mac_bitwidth} systolic array',
                    'dataflow': 'Weight-Stationary (WS)',
                    'peak_throughput_gops': self.config.num_macs * self.config.clock_freq_mhz / 1000,
                },
                'layernorm_unit': {
                    'type': 'Streaming LayerNorm',
                    'pipeline_stages': 3,
                    'throughput': f'{self.config.pe_cols} elements/cycle',
                    'operations': 'mean, variance, normalize, scale+shift',
                },
                'softmax_unit': {
                    'type': 'Piecewise-linear approximation',
                    'segments': 16,
                    'precision': '< 0.1% relative error',
                    'pipeline_stages': 4,
                    'operations': 'running-max subtraction, exp-LUT, accumulate, normalize',
                },
                'gelu_unit': {
                    'type': 'Piecewise-linear approximation',
                    'segments': 32,
                    'precision': '< 0.5% relative error',
                    'pipeline_stages': 2,
                },
                'flash_attention_controller': {
                    'tile_br': 64,
                    'tile_bc': 128,
                    'function': 'Coordinates tiled attention without full N×N materialization',
                    'running_stats_regs': '64 entries (max, sum) in FP16',
                },
            },
            'local_sram': f'{self.config.l1_sram_bytes_per_engine // 1024} KB',
            'supported_operations': [
                'Dense matrix multiplication (QKV projection, MLP)',
                'Tiled attention (Q×K^T, softmax, attn×V)',
                'LayerNorm',
                'GELU activation',
                'Residual addition',
            ],
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict:
        """Estimate area breakdown"""
        # Based on published data for similar designs
        scale = (node_nm / 28) ** 2  # Scale factor relative to 28nm

        mac_area = self.config.num_macs * 0.0004 * scale  # ~0.4 um² per INT8 MAC at 28nm
        l1_sram_area = self.config.l1_sram_bytes_per_engine * 0.001 * scale / 1024  # ~1 um²/KB at 28nm
        control_area = 0.05 * scale
        special_units = 0.03 * scale  # LN, softmax, GELU

        return {
            'mac_array_mm2': mac_area,
            'l1_sram_mm2': l1_sram_area,
            'control_mm2': control_area,
            'special_units_mm2': special_units,
            'total_mm2': mac_area + l1_sram_area + control_area + special_units,
        }


class CrossScaleFusionEngine:
    """
    Cross-Scale Fusion Engine (CSFE)
    Processes DPT decoder: multi-scale feature fusion with convolutions.

    Components:
    - Reconfigurable convolution engine (1×1 and 3×3)
    - Bilinear upsampling hardware unit
    - Feature addition/concatenation unit
    """

    def __init__(self, config: AcceleratorConfig):
        self.config = config

    def describe(self) -> dict:
        # CSFE shares the MAC array with VFE but reconfigured for conv
        conv_macs = min(self.config.num_macs, 256)  # Use subset for conv

        return {
            'name': 'Cross-Scale Fusion Engine (CSFE)',
            'function': 'DPT decoder multi-scale feature fusion',
            'components': {
                'conv_engine': {
                    'type': 'Output-Stationary (OS) convolution array',
                    'mac_count': conv_macs,
                    'supported_kernels': ['1×1', '3×3'],
                    'channels': 'Configurable, up to 256 output channels',
                },
                'upsample_unit': {
                    'type': 'Hardware bilinear interpolation',
                    'scale_factors': ['2×', '4×'],
                    'throughput': f'{self.config.pe_cols} pixels/cycle',
                    'pipeline_stages': 2,
                    'implementation': '4 multipliers + 3 adders per pixel',
                },
                'fusion_unit': {
                    'type': 'Element-wise addition with optional ReLU',
                    'throughput': f'{self.config.pe_cols * 4} elements/cycle',
                },
            },
            'local_sram': f'{self.config.l1_sram_bytes_per_engine // 1024} KB',
            'supported_operations': [
                '1×1 pointwise convolution',
                '3×3 depthwise/standard convolution',
                '2× bilinear upsampling',
                'Feature addition/concatenation',
                'ReLU activation',
            ],
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict:
        scale = (node_nm / 28) ** 2
        conv_macs = min(self.config.num_macs, 256)

        return {
            'conv_engine_mm2': conv_macs * 0.0004 * scale,
            'upsample_mm2': 0.01 * scale,
            'l1_sram_mm2': self.config.l1_sram_bytes_per_engine * 0.001 * scale / 1024,
            'control_mm2': 0.02 * scale,
            'total_mm2': conv_macs * 0.0004 * scale + 0.01 * scale +
                        self.config.l1_sram_bytes_per_engine * 0.001 * scale / 1024 + 0.02 * scale,
        }


class AbsoluteDisparityCU:
    """
    Absolute Disparity Calibration Unit (ADCU)
    Converts monocular relative depth to absolute stereo disparity.

    Components:
    - Sparse matching engine (NCC-based)
    - Least-squares scale-shift estimator
    - LUT-based reciprocal unit (1/Z)
    - B×f multiplier
    """

    def __init__(self, config: AcceleratorConfig):
        self.config = config

    def describe(self) -> dict:
        return {
            'name': 'Absolute Disparity Calibration Unit (ADCU)',
            'function': 'Converts relative depth to absolute disparity',
            'components': {
                'sparse_matcher': {
                    'type': 'NCC correlation engine',
                    'keypoints': 32,
                    'patch_size': '11×11',
                    'max_disparity': 192,
                    'throughput': '1 keypoint/16 cycles',
                    'total_cycles': 32 * 16,  # = 512 cycles
                },
                'scale_shift_estimator': {
                    'type': 'Hardware least-squares solver',
                    'problem_size': '32×2 (overdetermined 2×2)',
                    'operations': 'ATA (2×2), ATb (2×1), 2×2 inverse, solve',
                    'total_cycles': 64 + 32 + 10,  # Matrix ops + inverse + solve
                },
                'reciprocal_lut': {
                    'type': 'Dual-port SRAM lookup table',
                    'entries': 4096,
                    'entry_bits': 16,
                    'interpolation': 'Linear (2 reads + 1 multiply + 1 add)',
                    'total_size_bytes': 4096 * 2,
                },
                'depth_to_disparity': {
                    'type': 'Pipelined multiplier',
                    'pipeline_stages': 3,  # scale → LUT → multiply
                    'throughput': f'{self.config.pe_cols} pixels/cycle',
                    'operations': 'Z = α*d_rel + β → 1/Z (LUT) → d = B*f/Z',
                },
            },
            'local_sram': '8 KB (keypoint buffers + LUT)',
            'total_latency_estimate': {
                'sparse_matching': '512 cycles',
                'scale_estimation': '106 cycles',
                'depth_to_disparity_518x518': f'{518*518//self.config.pe_cols} cycles',
            },
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict:
        scale = (node_nm / 28) ** 2

        return {
            'sparse_matcher_mm2': 0.02 * scale,
            'least_squares_mm2': 0.005 * scale,
            'lut_sram_mm2': 4096 * 2 * 0.001 * scale / 1024,  # 8 KB
            'multipliers_mm2': 0.01 * scale,
            'control_mm2': 0.005 * scale,
            'total_mm2': 0.02 + 0.005 + 0.008 + 0.01 + 0.005,
        }


class GlobalMemoryController:
    """
    Global Memory Controller
    Manages L2 shared SRAM and external DRAM access.
    """

    def __init__(self, config: AcceleratorConfig):
        self.config = config

    def describe(self) -> dict:
        return {
            'name': 'Global Memory Controller',
            'components': {
                'l2_sram': {
                    'capacity': f'{self.config.l2_shared_sram_bytes // 1024} KB',
                    'banks': self.config.num_l2_banks,
                    'bank_size': f'{self.config.l2_shared_sram_bytes // self.config.num_l2_banks // 1024} KB',
                    'ports': '2 read + 1 write',
                    'bandwidth': f'{self.config.num_l2_banks * 8} bytes/cycle',
                },
                'dma_engine': {
                    'channels': self.config.dma_channels,
                    'burst_size': f'{self.config.dma_burst_bytes} bytes',
                    'patterns': ['Linear', '2D tiling', 'Strided'],
                    'function': 'Prefetch weights and activations from DRAM to L2',
                },
                'arbiter': {
                    'type': 'Round-robin with priority',
                    'priorities': 'VFE > CSFE > ADCU > DMA',
                },
            },
            'external_interface': {
                'type': 'LPDDR4x',
                'bandwidth': f'{self.config.dram_bandwidth_gbps} GB/s',
                'capacity': f'{self.config.dram_capacity_gb} GB',
            },
        }


class ControlProcessor:
    """
    Control Processor
    Lightweight RISC-V core for task scheduling and configuration.
    """

    def describe(self) -> dict:
        return {
            'name': 'Control Processor',
            'type': 'RISC-V RV32I micro-controller',
            'clock': 'Same as accelerator',
            'functions': [
                'Layer scheduling and sequencing',
                'Engine configuration (dataflow mode, tile sizes)',
                'DMA programming',
                'Interrupt handling',
                'Debug interface',
            ],
            'memory': {
                'instruction_rom': '16 KB',
                'data_ram': '4 KB',
            },
        }


class ConfidenceRouterUnit:
    """
    Confidence Router Unit (CRU)
    Routes tokens based on SGM confidence — the core sparsity-aware component.

    Takes the PKRN confidence map (byproduct of ADCU sparse matching),
    pools it to the token grid (37×37), and generates keep/prune masks.
    This is a single-cycle comparison: conf[i] > threshold → prune.

    Components:
    - 1369 parallel comparators (1 per spatial token)
    - Index generator (compact list of kept token indices)
    - Small index SRAM buffer (~2 KB)
    """

    def __init__(self, config: AcceleratorConfig, num_tokens: int = 1369):
        self.config = config
        self.num_tokens = num_tokens

    def describe(self) -> dict:
        return {
            'name': 'Confidence Router Unit (CRU)',
            'function': 'SGM-guided token pruning mask generation',
            'components': {
                'comparators': {
                    'count': self.num_tokens,
                    'type': 'FP16 threshold comparison',
                    'latency': '1 cycle',
                },
                'index_generator': {
                    'type': 'Parallel prefix-sum compaction',
                    'latency': '1 cycle',
                    'output': 'Compact index list of kept tokens',
                },
                'index_sram': {
                    'size_bytes': self.num_tokens * 2,  # ~2.7 KB (INT16 indices)
                    'purpose': 'Store keep_indices for Gather/Scatter',
                },
            },
            'total_latency': '1 cycle (fully pipelined)',
            'input': 'Confidence grid (37×37 FP16 from ADCU)',
            'output': 'keep_mask (1369 bits), keep_indices (M × INT16)',
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict:
        scale = (node_nm / 28) ** 2
        comparators = self.num_tokens * 0.000002 * scale  # ~2 um² per FP16 comparator
        index_logic = 0.001 * scale  # prefix-sum logic
        index_sram = self.num_tokens * 2 * 0.001 * scale / 1024  # ~2.7 KB

        total = comparators + index_logic + index_sram
        return {
            'comparators_mm2': comparators,
            'index_logic_mm2': index_logic,
            'index_sram_mm2': index_sram,
            'total_mm2': total,
        }


class EdgeStereoDAv2Accelerator:
    """
    Top-level accelerator: integrates all engines.
    """

    def __init__(self, config: AcceleratorConfig = None):
        self.config = config or AcceleratorConfig()
        self.vfe = ViTFeatureEngine(self.config)
        self.csfe = CrossScaleFusionEngine(self.config)
        self.adcu = AbsoluteDisparityCU(self.config)
        self.cru = ConfidenceRouterUnit(self.config)
        self.mem_ctrl = GlobalMemoryController(self.config)
        self.ctrl_proc = ControlProcessor()

    def full_spec(self) -> dict:
        return {
            'accelerator': 'EdgeStereoDAv2',
            'config': {
                'process': f'{self.config.process_node_nm}nm',
                'clock': f'{self.config.clock_freq_mhz} MHz',
                'pe_array': f'{self.config.pe_rows}×{self.config.pe_cols}',
                'peak_tops': f'{self.config.peak_tops:.2f} TOPS (INT8)',
                'total_sram': f'{self.config.total_sram_kb:.0f} KB',
            },
            'engines': {
                'VFE': self.vfe.describe(),
                'CSFE': self.csfe.describe(),
                'ADCU': self.adcu.describe(),
            },
            'memory': self.mem_ctrl.describe(),
            'control': self.ctrl_proc.describe(),
        }

    def area_breakdown(self, node_nm: int = None) -> dict:
        node = node_nm or self.config.process_node_nm
        scale = (node / 28) ** 2

        vfe = self.vfe.estimate_area_mm2(node)
        csfe = self.csfe.estimate_area_mm2(node)
        adcu = self.adcu.estimate_area_mm2(node)
        cru = self.cru.estimate_area_mm2(node)

        l2_sram = self.config.l2_shared_sram_bytes * 0.001 * scale / 1024
        ctrl = 0.05 * scale
        io_pads = 0.3 * scale
        interconnect = 0.1 * scale

        total = (vfe['total_mm2'] + csfe['total_mm2'] + adcu['total_mm2'] +
                 cru['total_mm2'] + l2_sram + ctrl + io_pads + interconnect)

        return {
            'VFE': vfe['total_mm2'],
            'CSFE': csfe['total_mm2'],
            'ADCU': adcu['total_mm2'],
            'CRU': cru['total_mm2'],
            'L2_SRAM': l2_sram,
            'Control': ctrl,
            'IO_Pads': io_pads,
            'Interconnect': interconnect,
            'Total_mm2': total,
            'node_nm': node,
        }

    def power_estimate(self, node_nm: int = None) -> dict:
        """Estimate power consumption"""
        node = node_nm or self.config.process_node_nm
        freq = self.config.clock_freq_mhz

        # Dynamic power scaling with technology node
        # Reference: 28nm at 500MHz
        # Smaller nodes: lower capacitance and voltage → less power per operation
        # But density allows more operations. Net: power per operation decreases.
        # Approximate: dynamic power per op scales as (node/28)^1.1 (Dennard-like)
        node_scale = (node / 28) ** 1.1
        freq_scale = freq / 500

        # Component power estimates (mW at 28nm, 500MHz, 100% utilization)
        mac_power = self.config.num_macs * 0.15  # ~0.15 mW per MAC at 28nm
        l1_sram_power = 3 * self.config.l1_sram_bytes_per_engine / 1024 * 0.2  # ~0.2 mW/KB
        l2_sram_power = self.config.l2_shared_sram_bytes / 1024 * 0.3  # ~0.3 mW/KB
        control_power = 10  # mW
        io_power = 30  # mW

        # Scale by node and frequency
        dynamic = (mac_power + l1_sram_power + l2_sram_power + control_power) * node_scale * freq_scale
        static = dynamic * 0.1  # Leakage ~10% of dynamic
        io = io_power  # IO power is relatively constant

        return {
            'mac_array_mW': mac_power * node_scale * freq_scale,
            'l1_sram_mW': l1_sram_power * node_scale * freq_scale,
            'l2_sram_mW': l2_sram_power * node_scale * freq_scale,
            'control_mW': control_power * node_scale * freq_scale,
            'io_mW': io,
            'dynamic_mW': dynamic,
            'static_mW': static,
            'total_mW': dynamic + static + io,
            'node_nm': node,
            'freq_mhz': freq,
        }

    def performance_estimate(self, total_flops: float = 79e9) -> dict:
        """Estimate performance metrics"""
        peak_ops = self.config.peak_tops * 1e12

        # Effective utilization
        # MHSA: ~85% utilization (good tiling)
        # MLP: ~90% utilization (large matrices)
        # Decoder convs: ~70% utilization (smaller)
        # Overall: ~82% average

        eff_utilization = 0.82
        effective_ops = peak_ops * eff_utilization

        latency_s = total_flops / effective_ops
        fps = 1.0 / latency_s

        return {
            'peak_tops': self.config.peak_tops,
            'effective_utilization': eff_utilization,
            'effective_tops': self.config.peak_tops * eff_utilization,
            'latency_ms': latency_s * 1000,
            'fps': fps,
            'total_flops_G': total_flops / 1e9,
        }


def print_accelerator_spec():
    """Print complete accelerator specification"""
    # 28nm configuration
    config_28nm = AcceleratorConfig(
        process_node_nm=28, clock_freq_mhz=500,
        pe_rows=32, pe_cols=32
    )
    accel_28nm = EdgeStereoDAv2Accelerator(config_28nm)

    # 7nm configuration
    config_7nm = AcceleratorConfig(
        process_node_nm=7, clock_freq_mhz=1000,
        pe_rows=32, pe_cols=32
    )
    accel_7nm = EdgeStereoDAv2Accelerator(config_7nm)

    print("=" * 70)
    print("EdgeStereoDAv2 Accelerator Specification")
    print("=" * 70)

    spec = accel_28nm.full_spec()
    print(f"\nProcess: {spec['config']['process']}")
    print(f"Clock: {spec['config']['clock']}")
    print(f"PE Array: {spec['config']['pe_array']}")
    print(f"Peak Throughput: {spec['config']['peak_tops']}")
    print(f"Total SRAM: {spec['config']['total_sram']}")

    # Area
    print(f"\n--- Area Breakdown ---")
    for node in [28, 7]:
        accel = accel_28nm if node == 28 else accel_7nm
        area = accel.area_breakdown(node)
        print(f"\n  {node}nm:")
        for k, v in area.items():
            if k != 'node_nm' and isinstance(v, float):
                print(f"    {k}: {v:.3f} mm2")

    # Power
    print(f"\n--- Power Estimates ---")
    for label, accel in [('28nm/500MHz', accel_28nm), ('7nm/1GHz', accel_7nm)]:
        pwr = accel.power_estimate()
        print(f"\n  {label}:")
        print(f"    Dynamic: {pwr['dynamic_mW']:.1f} mW")
        print(f"    Static: {pwr['static_mW']:.1f} mW")
        print(f"    IO: {pwr['io_mW']:.1f} mW")
        print(f"    Total: {pwr['total_mW']:.1f} mW")

    # Performance
    print(f"\n--- Performance (79 GFLOPs target model) ---")
    for label, accel in [('28nm/500MHz', accel_28nm), ('7nm/1GHz', accel_7nm)]:
        perf = accel.performance_estimate(79e9)
        print(f"\n  {label}:")
        print(f"    Peak: {perf['peak_tops']:.2f} TOPS")
        print(f"    Effective: {perf['effective_tops']:.2f} TOPS ({perf['effective_utilization']*100:.0f}%)")
        print(f"    Latency: {perf['latency_ms']:.2f} ms")
        print(f"    FPS: {perf['fps']:.1f}")


if __name__ == "__main__":
    print_accelerator_spec()
