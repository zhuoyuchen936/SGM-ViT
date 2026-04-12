"""
Sparsity Sweep — SGM-Guided Token Pruning Hardware Impact Analysis

Sweeps prune_ratio × prune_layer to quantify the hardware benefit of
confidence-guided token sparsity. Generates tables for the paper.
"""

import json
import os
import sys

_SIM_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _SIM_ROOT not in sys.path:
    sys.path.insert(0, _SIM_ROOT)

from simulator.core.simulator import CycleAccurateSimulator, SimConfig, SparsityConfig
from simulator.analysis.energy import EnergyModel


def run_sparsity_sweep():
    prune_ratios = [0.0, 0.10, 0.186, 0.20, 0.30, 0.40, 0.50]
    prune_layers = [0, 3, 6, 9]
    nodes = [(28, 500), (7, 1000)]

    results = []

    # Dense baseline
    for node, freq in nodes:
        config = SimConfig(clock_freq_mhz=freq, pe_rows=32, pe_cols=32,
                           sparsity=SparsityConfig(prune_ratio=0.0))
        sim = CycleAccurateSimulator(config)
        sim.build_workload(img_h=518, img_w=518)
        sim_r = sim.simulate_frame()

        energy = EnergyModel(process_node_nm=node, clock_freq_mhz=freq)
        e_r = energy.frame_energy(sim_r)

        results.append({
            'node_nm': node, 'freq_mhz': freq,
            'prune_ratio': 0.0, 'prune_layer': 'N/A',
            'total_flops_G': sim_r['total_flops'] / 1e9,
            'total_cycles': sim_r['total_cycles'],
            'latency_ms': sim_r['latency_ms'],
            'fps': sim_r['fps'],
            'energy_mJ': e_r['total_energy_mJ'],
            'power_mW': e_r['average_power_mW'],
            'tops_per_w': e_r['energy_efficiency_TOPS_W'],
            'speedup': 1.0,
            'energy_saving_pct': 0.0,
        })

    # Sparse configurations
    for node, freq in nodes:
        # Get baseline for speedup calculation
        baseline_fps = [r for r in results
                        if r['node_nm'] == node and r['prune_ratio'] == 0.0][0]['fps']
        baseline_energy = [r for r in results
                           if r['node_nm'] == node and r['prune_ratio'] == 0.0][0]['energy_mJ']

        for pl in prune_layers:
            for pr in prune_ratios:
                if pr == 0.0:
                    continue  # Already in baseline

                sp = SparsityConfig(prune_ratio=pr, prune_layer=pl)
                config = SimConfig(clock_freq_mhz=freq, pe_rows=32, pe_cols=32,
                                   sparsity=sp)
                sim = CycleAccurateSimulator(config)
                sim.build_workload(img_h=518, img_w=518)
                sim_r = sim.simulate_frame()

                energy = EnergyModel(process_node_nm=node, clock_freq_mhz=freq)
                e_r = energy.frame_energy(sim_r)

                results.append({
                    'node_nm': node, 'freq_mhz': freq,
                    'prune_ratio': pr, 'prune_layer': pl,
                    'total_flops_G': sim_r['total_flops'] / 1e9,
                    'total_cycles': sim_r['total_cycles'],
                    'latency_ms': sim_r['latency_ms'],
                    'fps': sim_r['fps'],
                    'energy_mJ': e_r['total_energy_mJ'],
                    'power_mW': e_r['average_power_mW'],
                    'tops_per_w': e_r['energy_efficiency_TOPS_W'],
                    'speedup': sim_r['fps'] / baseline_fps,
                    'energy_saving_pct': (1 - e_r['total_energy_mJ'] / baseline_energy) * 100,
                })

    return results


def print_results(results):
    print("=" * 90)
    print("  SGM-ViT Sparsity Sweep — Hardware Impact of Token Pruning")
    print("=" * 90)

    for node, freq in [(28, 500), (7, 1000)]:
        subset = [r for r in results if r['node_nm'] == node]
        print(f"\n  --- {node}nm / {freq}MHz ---")
        print(f"  {'Prune%':>7s}  {'Layer':>5s}  {'FLOPs(G)':>9s}  {'Lat(ms)':>8s}  "
              f"{'FPS':>6s}  {'Speedup':>8s}  {'E(mJ)':>7s}  {'E-save%':>8s}  {'TOPS/W':>7s}")
        print(f"  {'-'*80}")

        for r in subset:
            pl = str(r['prune_layer']) if r['prune_layer'] != 'N/A' else '—'
            print(f"  {r['prune_ratio']*100:>6.1f}%  {pl:>5s}  {r['total_flops_G']:>9.1f}  "
                  f"{r['latency_ms']:>8.1f}  {r['fps']:>6.2f}  {r['speedup']:>7.2f}x  "
                  f"{r['energy_mJ']:>7.2f}  {r['energy_saving_pct']:>7.1f}%  "
                  f"{r['tops_per_w']:>7.2f}")

    # Paper-ready summary: our operating point (18.6% prune, layer=0)
    print(f"\n{'='*90}")
    print("  Paper Summary — SGM-Guided Pruning (18.6%, layer=0)")
    print(f"{'='*90}")
    print(f"  {'Config':<20s}  {'Dense FPS':>10s}  {'Sparse FPS':>11s}  {'Speedup':>8s}  "
          f"{'E-Dense(mJ)':>12s}  {'E-Sparse(mJ)':>13s}  {'E-save':>7s}")
    print(f"  {'-'*90}")

    for node, freq in [(28, 500), (7, 1000)]:
        dense = [r for r in results if r['node_nm'] == node and r['prune_ratio'] == 0.0][0]
        sparse = [r for r in results if r['node_nm'] == node
                  and r['prune_ratio'] == 0.186 and r['prune_layer'] == 0]
        if sparse:
            s = sparse[0]
            print(f"  {node}nm/{freq}MHz{'':<8s}  {dense['fps']:>10.2f}  {s['fps']:>11.2f}  "
                  f"{s['speedup']:>7.2f}x  {dense['energy_mJ']:>12.2f}  "
                  f"{s['energy_mJ']:>13.2f}  {s['energy_saving_pct']:>6.1f}%")


def main():
    results = run_sparsity_sweep()
    print_results(results)

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sparsity_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
