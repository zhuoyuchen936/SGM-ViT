"""
Performance Analysis Module
Sweeps across configurations and collects performance metrics
"""

import sys
import numpy as np
sys.path.insert(0, 'G:/workspace')

from simulator.core.simulator import CycleAccurateSimulator, SimConfig


def sweep_configurations() -> dict:
    """
    Run simulator across multiple configurations:
    - Input resolutions
    - PE array sizes
    - Clock frequencies
    """
    results = {
        'resolution_sweep': [],
        'pe_sweep': [],
        'freq_sweep': [],
    }

    # Resolution sweep (32×32 PE, 500 MHz)
    resolutions = [(480, 640), (518, 518), (720, 1280), (1080, 1920)]
    for h, w in resolutions:
        config = SimConfig(clock_freq_mhz=500, pe_rows=32, pe_cols=32)
        sim = CycleAccurateSimulator(config)
        sim.build_workload(img_h=h, img_w=w)
        r = sim.simulate_frame()

        results['resolution_sweep'].append({
            'resolution': f'{h}x{w}',
            'fps': r['fps'],
            'latency_ms': r['latency_ms'],
            'total_flops_G': r['total_flops'] / 1e9,
            'total_cycles': r['total_cycles'],
        })

    # PE array size sweep (518×518, 500 MHz)
    pe_sizes = [(8, 8), (16, 16), (32, 32)]
    for pr, pc in pe_sizes:
        config = SimConfig(clock_freq_mhz=500, pe_rows=pr, pe_cols=pc)
        sim = CycleAccurateSimulator(config)
        sim.build_workload(img_h=518, img_w=518)
        r = sim.simulate_frame()

        results['pe_sweep'].append({
            'pe_array': f'{pr}x{pc}',
            'num_macs': pr * pc,
            'fps': r['fps'],
            'latency_ms': r['latency_ms'],
            'peak_tops': pr * pc * 500e6 * 2 / 1e12,
        })

    # Frequency sweep (518×518, 32×32 PE)
    freqs = [200, 500, 1000]
    for freq in freqs:
        config = SimConfig(clock_freq_mhz=freq, pe_rows=32, pe_cols=32)
        sim = CycleAccurateSimulator(config)
        sim.build_workload(img_h=518, img_w=518)
        r = sim.simulate_frame()

        results['freq_sweep'].append({
            'freq_mhz': freq,
            'fps': r['fps'],
            'latency_ms': r['latency_ms'],
            'peak_tops': 32 * 32 * freq * 1e6 * 2 / 1e12,
        })

    return results


def comparison_table() -> list:
    """Generate comparison with other accelerators"""
    # Our results at different configs
    configs = [
        ('28nm/500MHz', 28, 500),
        ('7nm/1GHz', 7, 1000),
    ]

    our_results = []
    for label, node, freq in configs:
        config = SimConfig(clock_freq_mhz=freq, pe_rows=32, pe_cols=32)
        sim = CycleAccurateSimulator(config)
        sim.build_workload(img_h=518, img_w=518)
        r = sim.simulate_frame()

        from hardware.architecture.top_level import AcceleratorConfig, EdgeStereoDAv2Accelerator
        accel_config = AcceleratorConfig(process_node_nm=node, clock_freq_mhz=freq, pe_rows=32, pe_cols=32)
        accel = EdgeStereoDAv2Accelerator(accel_config)
        area = accel.area_breakdown(node)
        power = accel.power_estimate(node)

        our_results.append({
            'name': f'Ours ({label})',
            'process_nm': node,
            'freq_mhz': freq,
            'pe_count': 1024,
            'tops': config.num_macs * freq * 1e6 * 2 / 1e12,
            'fps': r['fps'],
            'power_mw': power['total_mW'],
            'area_mm2': area['Total_mm2'],
            'tops_per_w': (config.num_macs * freq * 1e6 * 2 / 1e12) / (power['total_mW'] / 1000),
            'fps_per_w': r['fps'] / (power['total_mW'] / 1000),
        })

    # Baseline comparisons (from literature)
    baselines = [
        {
            'name': 'Eyeriss v2',
            'process_nm': 65,
            'freq_mhz': 200,
            'pe_count': 192,
            'tops': 0.077,
            'fps': 0.2,  # For ViT workload (not natively supported)
            'power_mw': 236,
            'area_mm2': 12.25,
            'tops_per_w': 0.33,
            'fps_per_w': 0.85,
        },
        {
            'name': 'NVDLA (small)',
            'process_nm': 28,
            'freq_mhz': 500,
            'pe_count': 64,
            'tops': 0.064,
            'fps': 0.8,
            'power_mw': 100,
            'area_mm2': 1.0,
            'tops_per_w': 0.64,
            'fps_per_w': 8.0,
        },
        {
            'name': 'ViTA (HPCA23)',
            'process_nm': 28,
            'freq_mhz': 200,
            'pe_count': 256,
            'tops': 0.102,
            'fps': 4.0,  # ViT-S
            'power_mw': 42,
            'area_mm2': 2.1,
            'tops_per_w': 2.4,
            'fps_per_w': 95.2,
        },
        {
            'name': 'FACT (DAC23)',
            'process_nm': 28,
            'freq_mhz': 500,
            'pe_count': 512,
            'tops': 0.512,
            'fps': 7.0,  # Estimated for ViT-S
            'power_mw': 200,
            'area_mm2': 3.5,
            'tops_per_w': 2.56,
            'fps_per_w': 35.0,
        },
        {
            'name': 'Jetson Nano (GPU)',
            'process_nm': 20,
            'freq_mhz': 921,
            'pe_count': 128,
            'tops': 0.472,
            'fps': 5.0,  # ViT-S depth est.
            'power_mw': 5000,
            'area_mm2': 100.0,  # SoC area
            'tops_per_w': 0.094,
            'fps_per_w': 1.0,
        },
    ]

    return our_results + baselines


if __name__ == "__main__":
    print("=== Performance Analysis ===\n")

    # Sweep results
    results = sweep_configurations()

    print("--- Resolution Sweep (32x32 PE, 500MHz) ---")
    print(f"{'Resolution':<15} {'FLOPs(G)':>10} {'Cycles':>12} {'Latency(ms)':>12} {'FPS':>8}")
    print("-" * 60)
    for r in results['resolution_sweep']:
        print(f"  {r['resolution']:<13} {r['total_flops_G']:>10.1f} {r['total_cycles']:>12,} "
              f"{r['latency_ms']:>12.1f} {r['fps']:>8.1f}")

    print(f"\n--- PE Array Size Sweep (518x518, 500MHz) ---")
    print(f"{'PE Array':<12} {'MACs':>8} {'Peak TOPS':>10} {'Latency(ms)':>12} {'FPS':>8}")
    print("-" * 55)
    for r in results['pe_sweep']:
        print(f"  {r['pe_array']:<10} {r['num_macs']:>8} {r['peak_tops']:>10.3f} "
              f"{r['latency_ms']:>12.1f} {r['fps']:>8.1f}")

    print(f"\n--- Frequency Sweep (518x518, 32x32 PE) ---")
    print(f"{'Freq(MHz)':>10} {'Peak TOPS':>10} {'Latency(ms)':>12} {'FPS':>8}")
    print("-" * 45)
    for r in results['freq_sweep']:
        print(f"  {r['freq_mhz']:>8} {r['peak_tops']:>10.3f} {r['latency_ms']:>12.1f} {r['fps']:>8.1f}")

    # Comparison table
    print(f"\n--- Comparison with Other Accelerators ---")
    table = comparison_table()
    print(f"{'Name':<22} {'Node':>5} {'MACs':>6} {'TOPS':>6} {'FPS':>6} {'Power':>7} {'Area':>6} {'TOPS/W':>7} {'FPS/W':>7}")
    print("-" * 85)
    for row in table:
        print(f"  {row['name']:<20} {row['process_nm']:>4}nm {row['pe_count']:>5} {row['tops']:>6.3f} "
              f"{row['fps']:>6.1f} {row['power_mw']:>6.0f}mW {row['area_mm2']:>5.2f} "
              f"{row['tops_per_w']:>7.2f} {row['fps_per_w']:>7.1f}")
