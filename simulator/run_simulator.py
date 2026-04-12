"""
Main Simulator Runner
Runs all analyses and collects results for the paper
"""

import sys
import json
import os
_SIM_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _SIM_ROOT not in sys.path:
    sys.path.insert(0, _SIM_ROOT)

from simulator.core.simulator import CycleAccurateSimulator, SimConfig
from simulator.analysis.performance import sweep_configurations, comparison_table
from simulator.analysis.energy import EnergyModel, energy_vs_fps_analysis
from simulator.analysis.area import area_analysis, pe_scaling_analysis
from simulator.analysis.roofline import RooflineModel


def run_all_analyses():
    """Run complete analysis suite"""

    print("=" * 70)
    print("EdgeStereoDAv2 Simulator: Complete Analysis Suite")
    print("=" * 70)

    all_results = {}

    # ============================================
    # 1. Cycle-accurate simulation
    # ============================================
    print("\n[1] Cycle-accurate simulation...")

    for label, node, freq in [('28nm_500MHz', 28, 500), ('7nm_1GHz', 7, 1000)]:
        config = SimConfig(clock_freq_mhz=freq, pe_rows=32, pe_cols=32)
        sim = CycleAccurateSimulator(config)
        sim.build_workload(img_h=518, img_w=518)
        results = sim.simulate_frame()

        print(f"\n  {label}: {results['fps']:.1f} FPS, {results['latency_ms']:.1f} ms, "
              f"{results['total_flops']/1e9:.1f} GFLOPs")

        all_results[f'sim_{label}'] = {
            'fps': results['fps'],
            'latency_ms': results['latency_ms'],
            'total_cycles': results['total_cycles'],
            'total_flops_G': results['total_flops'] / 1e9,
        }

    # ============================================
    # 2. Performance sweeps
    # ============================================
    print("\n[2] Performance sweeps...")
    sweep = sweep_configurations()
    all_results['resolution_sweep'] = sweep['resolution_sweep']
    all_results['pe_sweep'] = sweep['pe_sweep']
    all_results['freq_sweep'] = sweep['freq_sweep']

    print(f"  Resolution sweep: {len(sweep['resolution_sweep'])} configs")
    for r in sweep['resolution_sweep']:
        print(f"    {r['resolution']}: {r['fps']:.1f} FPS")

    # ============================================
    # 3. Energy analysis
    # ============================================
    print("\n[3] Energy analysis...")

    for label, node, freq in [('28nm_500MHz', 28, 500), ('7nm_1GHz', 7, 1000)]:
        config = SimConfig(clock_freq_mhz=freq, pe_rows=32, pe_cols=32)
        sim = CycleAccurateSimulator(config)
        sim.build_workload(img_h=518, img_w=518)
        sim_results = sim.simulate_frame()

        energy_model = EnergyModel(process_node_nm=node, clock_freq_mhz=freq)
        e = energy_model.frame_energy(sim_results)

        print(f"  {label}: {e['total_energy_mJ']:.2f} mJ/frame, {e['average_power_mW']:.0f} mW, "
              f"{e['energy_efficiency_TOPS_W']:.2f} TOPS/W")

        all_results[f'energy_{label}'] = {
            'total_mJ': e['total_energy_mJ'],
            'power_mW': e['average_power_mW'],
            'tops_per_w': e['energy_efficiency_TOPS_W'],
            'breakdown': e['breakdown_pct'],
            'per_engine': e['per_engine_mJ'],
        }

    efps = energy_vs_fps_analysis()
    all_results['energy_fps_sweep'] = efps

    # ============================================
    # 4. Area analysis
    # ============================================
    print("\n[4] Area analysis...")
    area = area_analysis()
    for node_label, data in area.items():
        total = data['top_level']['Total_mm2']
        print(f"  {node_label}: {total:.4f} mm2 total")
        all_results[f'area_{node_label}'] = data['top_level']

    pe_scale = pe_scaling_analysis()
    all_results['pe_scaling'] = pe_scale

    # ============================================
    # 5. Roofline analysis
    # ============================================
    print("\n[5] Roofline analysis...")
    roofline = RooflineModel(peak_tops=1.024, memory_bw_gbps=64)
    ops = roofline.analyze_operations()

    compute_bound = sum(1 for op in ops if op['bound'] == 'Compute')
    memory_bound = sum(1 for op in ops if op['bound'] == 'Memory')
    print(f"  Compute-bound: {compute_bound}, Memory-bound: {memory_bound}")
    print(f"  Ridge point: {roofline.ridge_point:.1f} OPS/Byte")

    all_results['roofline'] = {
        'ridge_point': roofline.ridge_point,
        'operations': [
            {'name': op['name'], 'ai': op['arithmetic_intensity'],
             'bound': op['bound'], 'utilization': op['utilization']}
            for op in ops
        ],
    }

    # ============================================
    # 6. Comparison table
    # ============================================
    print("\n[6] Comparison with baselines...")
    comparison = comparison_table()
    all_results['comparison'] = comparison

    print(f"\n{'Name':<22} {'FPS':>6} {'Power':>7} {'TOPS/W':>7} {'FPS/W':>7}")
    print("-" * 55)
    for row in comparison:
        print(f"  {row['name']:<20} {row['fps']:>6.1f} {row['power_mw']:>6.0f}mW "
              f"{row['tops_per_w']:>7.2f} {row['fps_per_w']:>7.1f}")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)

    # Key metrics for the paper
    sim_28 = all_results['sim_28nm_500MHz']
    sim_7 = all_results['sim_7nm_1GHz']
    e_28 = all_results['energy_28nm_500MHz']
    e_7 = all_results['energy_7nm_1GHz']
    a_28 = all_results['area_28nm']
    a_7 = all_results['area_7nm']

    print(f"\n{'Metric':<30} {'28nm/500MHz':>15} {'7nm/1GHz':>15}")
    print("-" * 62)
    print(f"  {'FPS':<28} {sim_28['fps']:>15.1f} {sim_7['fps']:>15.1f}")
    print(f"  {'Latency (ms)':<28} {sim_28['latency_ms']:>15.1f} {sim_7['latency_ms']:>15.1f}")
    print(f"  {'Power (mW)':<28} {e_28['power_mW']:>15.0f} {e_7['power_mW']:>15.0f}")
    print(f"  {'Energy/frame (mJ)':<28} {e_28['total_mJ']:>15.2f} {e_7['total_mJ']:>15.2f}")
    print(f"  {'TOPS/W':<28} {e_28['tops_per_w']:>15.2f} {e_7['tops_per_w']:>15.2f}")
    print(f"  {'Area (mm2)':<28} {a_28['Total_mm2']:>15.4f} {a_7['Total_mm2']:>15.4f}")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save serializable results
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, (dict, list, int, float, str)):
            serializable[k] = v

    with open(os.path.join(results_dir, 'simulation_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nResults saved to {results_dir}/simulation_results.json")

    return all_results


if __name__ == "__main__":
    run_all_analyses()
