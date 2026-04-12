"""
Area Estimation Module
Detailed area breakdown for the accelerator at different process nodes
"""

import os
import sys

_SIM_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SIM_ROOT not in sys.path:
    sys.path.insert(0, _SIM_ROOT)

from hardware.architecture.top_level import AcceleratorConfig, EdgeStereoDAv2Accelerator


def area_analysis():
    """Comprehensive area analysis at 28nm and 7nm"""
    results = {}

    for node in [28, 7]:
        freq = 500 if node == 28 else 1000
        config = AcceleratorConfig(
            process_node_nm=node,
            clock_freq_mhz=freq,
            pe_rows=32, pe_cols=32
        )
        accel = EdgeStereoDAv2Accelerator(config)
        area = accel.area_breakdown(node)

        # More detailed breakdown
        vfe = accel.vfe.estimate_area_mm2(node)
        csfe = accel.csfe.estimate_area_mm2(node)
        adcu = accel.adcu.estimate_area_mm2(node)

        results[f'{node}nm'] = {
            'top_level': area,
            'vfe_detail': vfe,
            'csfe_detail': csfe,
            'adcu_detail': adcu,
        }

    return results


def pe_scaling_analysis():
    """Area vs PE array size"""
    results = []
    for pe_size in [8, 16, 32, 64]:
        for node in [28, 7]:
            config = AcceleratorConfig(
                process_node_nm=node,
                clock_freq_mhz=500 if node == 28 else 1000,
                pe_rows=pe_size, pe_cols=pe_size
            )
            accel = EdgeStereoDAv2Accelerator(config)
            area = accel.area_breakdown(node)

            results.append({
                'pe_size': f'{pe_size}x{pe_size}',
                'num_macs': pe_size * pe_size,
                'node_nm': node,
                'total_area_mm2': area['Total_mm2'],
                'vfe_area_mm2': area['VFE'],
                'sram_area_mm2': area['L2_SRAM'],
            })

    return results


if __name__ == "__main__":
    print("=== Area Analysis ===\n")

    results = area_analysis()

    for node_label, data in results.items():
        print(f"\n--- {node_label} ---")
        top = data['top_level']
        total = top['Total_mm2']

        print(f"{'Component':<20} {'Area(mm2)':>10} {'Fraction':>10}")
        print("-" * 42)
        for k, v in top.items():
            if k not in ('Total_mm2', 'node_nm') and isinstance(v, float):
                print(f"  {k:<18} {v:>10.4f} {v/total*100:>9.1f}%")
        print(f"  {'TOTAL':<18} {total:>10.4f} {'100.0':>9}%")

    # PE scaling
    print(f"\n--- PE Array Scaling ---")
    pe_results = pe_scaling_analysis()
    print(f"{'PE Array':<12} {'Node':>6} {'MACs':>6} {'Total(mm2)':>10} {'VFE(mm2)':>10} {'SRAM(mm2)':>10}")
    print("-" * 58)
    for r in pe_results:
        print(f"  {r['pe_size']:<10} {r['node_nm']:>5}nm {r['num_macs']:>5} "
              f"{r['total_area_mm2']:>10.4f} {r['vfe_area_mm2']:>10.4f} {r['sram_area_mm2']:>10.4f}")
