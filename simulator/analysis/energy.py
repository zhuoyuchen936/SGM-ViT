"""
Energy Analysis Module
Estimates per-frame energy breakdown by component and operation type
"""

import os
import sys
import numpy as np

_SIM_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SIM_ROOT not in sys.path:
    sys.path.insert(0, _SIM_ROOT)

from simulator.core.simulator import CycleAccurateSimulator, SimConfig, SparsityConfig


class EnergyModel:
    """
    Energy estimation based on technology-calibrated parameters.

    References:
    - Horowitz, "Computing's Energy Problem" (ISSCC 2014)
    - Eyeriss: JSSC 2017
    - CACTI for SRAM
    """

    def __init__(self, process_node_nm: int = 28, clock_freq_mhz: int = 500):
        self.node = process_node_nm
        self.freq = clock_freq_mhz

        # Energy per operation (picojoules) at reference: 45nm
        # Scale to target node
        scale = (process_node_nm / 45) ** 1.2  # Energy scales ~linearly with node

        self.energy_pJ = {
            'int8_mac': 0.2 * scale,           # INT8 multiply-accumulate
            'int8_add': 0.03 * scale,           # INT8 addition
            'int32_add': 0.1 * scale,           # INT32 accumulation
            'fp16_mac': 0.9 * scale,            # FP16 multiply-accumulate
            'reg_read': 0.05 * scale,           # Register file read
            'reg_write': 0.05 * scale,          # Register file write
            'sram_read_64b': 5.0 * scale,       # 64-bit SRAM read (L1)
            'sram_write_64b': 5.0 * scale,      # 64-bit SRAM write (L1)
            'global_sram_read_64b': 20.0 * scale,  # L2 SRAM read
            'global_sram_write_64b': 20.0 * scale,  # L2 SRAM write
            'dram_read_64b': 200.0,             # DRAM read (less dependent on node)
            'dram_write_64b': 200.0,
        }

    def compute_energy(self, total_macs: int, mem_reads_bytes: int,
                        mem_writes_bytes: int, l1_hit_rate: float = 0.80,
                        l2_hit_rate: float = 0.90) -> dict:
        """
        Estimate total energy for an operation.

        Args:
            total_macs: number of MAC operations
            mem_reads_bytes: total memory read bytes
            mem_writes_bytes: total memory write bytes
            l1_hit_rate: fraction of accesses served by L1
            l2_hit_rate: fraction of L1-miss accesses served by L2
        """
        e = self.energy_pJ

        # Compute energy
        mac_energy = total_macs * e['int8_mac']

        # Memory energy (hierarchical)
        total_reads = mem_reads_bytes / 8  # 64-bit accesses
        total_writes = mem_writes_bytes / 8

        l1_reads = total_reads * l1_hit_rate
        l2_reads = total_reads * (1 - l1_hit_rate) * l2_hit_rate
        dram_reads = total_reads * (1 - l1_hit_rate) * (1 - l2_hit_rate)

        l1_writes = total_writes * l1_hit_rate
        l2_writes = total_writes * (1 - l1_hit_rate) * l2_hit_rate
        dram_writes = total_writes * (1 - l1_hit_rate) * (1 - l2_hit_rate)

        mem_read_energy = (
            l1_reads * e['sram_read_64b'] +
            l2_reads * e['global_sram_read_64b'] +
            dram_reads * e['dram_read_64b']
        )

        mem_write_energy = (
            l1_writes * e['sram_write_64b'] +
            l2_writes * e['global_sram_write_64b'] +
            dram_writes * e['dram_write_64b']
        )

        # Register file energy (~2 reads + 1 write per MAC)
        reg_energy = total_macs * (2 * e['reg_read'] + e['reg_write'])

        # Control overhead (~5% of compute)
        control_energy = mac_energy * 0.05

        total = mac_energy + mem_read_energy + mem_write_energy + reg_energy + control_energy

        return {
            'mac_energy_pJ': mac_energy,
            'reg_energy_pJ': reg_energy,
            'l1_energy_pJ': (l1_reads * e['sram_read_64b'] + l1_writes * e['sram_write_64b']),
            'l2_energy_pJ': (l2_reads * e['global_sram_read_64b'] + l2_writes * e['global_sram_write_64b']),
            'dram_energy_pJ': (dram_reads * e['dram_read_64b'] + dram_writes * e['dram_write_64b']),
            'control_energy_pJ': control_energy,
            'total_energy_pJ': total,
            'total_energy_mJ': total / 1e9,
            'breakdown_pct': {
                'mac': mac_energy / total * 100,
                'reg': reg_energy / total * 100,
                'l1': (l1_reads * e['sram_read_64b'] + l1_writes * e['sram_write_64b']) / total * 100,
                'l2': (l2_reads * e['global_sram_read_64b'] + l2_writes * e['global_sram_write_64b']) / total * 100,
                'dram': (dram_reads * e['dram_read_64b'] + dram_writes * e['dram_write_64b']) / total * 100,
                'control': control_energy / total * 100,
            },
        }

    def frame_energy(self, sim_results: dict) -> dict:
        """Compute energy for one complete frame from simulation results"""
        total_energy = {'mac': 0, 'reg': 0, 'l1': 0, 'l2': 0, 'dram': 0, 'control': 0}

        per_engine = {'VFE': 0, 'CSFE': 0, 'ADCU': 0}

        for op in sim_results['operations']:
            macs = op['flops'] // 2
            reads = op.get('weight_bytes', 0) + op.get('input_bytes', 0)
            writes = op.get('output_bytes', 0)

            e = self.compute_energy(macs, reads, writes)

            per_engine[op['engine']] += e['total_energy_pJ']

            total_energy['mac'] += e['mac_energy_pJ']
            total_energy['reg'] += e['reg_energy_pJ']
            total_energy['l1'] += e['l1_energy_pJ']
            total_energy['l2'] += e['l2_energy_pJ']
            total_energy['dram'] += e['dram_energy_pJ']
            total_energy['control'] += e['control_energy_pJ']

        total_pJ = sum(total_energy.values())

        # Static power
        # Leakage: ~10-20% of dynamic at 28nm
        static_fraction = 0.15 if self.node >= 28 else 0.25  # Higher leakage at smaller nodes
        total_cycles = sim_results['total_cycles']
        period_ns = 1000 / self.freq  # ns per cycle
        frame_time_ns = total_cycles * period_ns
        static_energy_pJ = total_pJ * static_fraction

        grand_total = total_pJ + static_energy_pJ

        return {
            'dynamic_energy_mJ': total_pJ / 1e9,
            'static_energy_mJ': static_energy_pJ / 1e9,
            'total_energy_mJ': grand_total / 1e9,
            'average_power_mW': grand_total / frame_time_ns,  # pJ / ns = mW
            'per_engine_mJ': {k: v / 1e9 for k, v in per_engine.items()},
            'breakdown_pct': {k: v / total_pJ * 100 for k, v in total_energy.items()},
            'energy_efficiency_TOPS_W': (
                sum(op['flops'] for op in sim_results['operations']) / 1e12
            ) / (grand_total / 1e12),  # TOPS / W
        }


def energy_vs_fps_analysis():
    """Analyze energy-FPS tradeoff across configurations"""
    results = []

    for node in [28, 7]:
        for freq in [200, 500, 1000]:
            if node == 28 and freq == 1000:
                continue  # Not realistic for 28nm

            config = SimConfig(clock_freq_mhz=freq, pe_rows=32, pe_cols=32)
            sim = CycleAccurateSimulator(config)
            sim.build_workload(img_h=518, img_w=518)
            sim_results = sim.simulate_frame()

            energy = EnergyModel(process_node_nm=node, clock_freq_mhz=freq)
            e_results = energy.frame_energy(sim_results)

            results.append({
                'config': f'{node}nm/{freq}MHz',
                'fps': sim_results['fps'],
                'power_mW': e_results['average_power_mW'],
                'energy_per_frame_mJ': e_results['total_energy_mJ'],
                'tops_per_w': e_results['energy_efficiency_TOPS_W'],
            })

    return results


if __name__ == "__main__":
    print("=== Energy Analysis ===\n")

    # Single frame analysis at 28nm/500MHz
    config = SimConfig(clock_freq_mhz=500, pe_rows=32, pe_cols=32)
    sim = CycleAccurateSimulator(config)
    sim.build_workload(img_h=518, img_w=518)
    sim_results = sim.simulate_frame()

    energy = EnergyModel(process_node_nm=28, clock_freq_mhz=500)
    e = energy.frame_energy(sim_results)

    print(f"--- 28nm / 500MHz ---")
    print(f"Dynamic energy: {e['dynamic_energy_mJ']:.2f} mJ")
    print(f"Static energy: {e['static_energy_mJ']:.2f} mJ")
    print(f"Total energy: {e['total_energy_mJ']:.2f} mJ")
    print(f"Average power: {e['average_power_mW']:.1f} mW")
    print(f"Energy efficiency: {e['energy_efficiency_TOPS_W']:.2f} TOPS/W")

    print(f"\nEnergy breakdown:")
    for k, v in e['breakdown_pct'].items():
        print(f"  {k}: {v:.1f}%")

    print(f"\nPer-engine energy:")
    for k, v in e['per_engine_mJ'].items():
        print(f"  {k}: {v:.2f} mJ")

    # Energy vs FPS sweep
    print(f"\n--- Energy vs FPS Sweep ---")
    results = energy_vs_fps_analysis()
    print(f"{'Config':<15} {'FPS':>8} {'Power(mW)':>10} {'E/frame(mJ)':>12} {'TOPS/W':>8}")
    print("-" * 58)
    for r in results:
        print(f"  {r['config']:<13} {r['fps']:>8.1f} {r['power_mW']:>10.1f} "
              f"{r['energy_per_frame_mJ']:>12.2f} {r['tops_per_w']:>8.2f}")
