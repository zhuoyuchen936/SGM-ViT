"""
simulator/run_simulator.py
==========================
Main simulator runner using the event-driven simulator.

Runs the full analysis suite and produces results/simulation_results.json
and results/sparsity_sweep.json for paper tables and figures.

Step 14 update: uses EventDrivenSimulator (SA + 5 SCUs + event-skip).
Legacy batch CycleAccurateSimulator retained for backward comparison.
"""
import sys
import json
import os

_SIM_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _SIM_ROOT not in sys.path:
    sys.path.insert(0, _SIM_ROOT)

from simulator.core.event_simulator import EventDrivenSimulator, SimConfig
from hardware.architecture.top_level import (
    EdgeStereoDAv2Accelerator, AcceleratorConfig,
)


def run_event_simulation(keep_ratio: float = 1.0, node: int = 28,
                          freq: int = 500, stage_policy: str = "coarse_only",
                          img_h: int = 518, img_w: int = 518) -> dict:
    """Run one frame of event-driven simulation."""
    cfg = SimConfig(
        clock_freq_mhz=freq, process_node_nm=node,
        keep_ratio=keep_ratio, stage_policy=stage_policy,
    )
    sim = EventDrivenSimulator(cfg)
    return sim.simulate_frame(img_h=img_h, img_w=img_w)


def run_area_power_analysis(node: int = 28, freq: int = 500) -> dict:
    """Area + power breakdown via top_level.Accelerator."""
    cfg = AcceleratorConfig(process_node_nm=node, clock_freq_mhz=freq)
    accel = EdgeStereoDAv2Accelerator(cfg)
    return {
        "area": accel.area_breakdown(),
        "power": accel.power_estimate(),
        "l2_budget": accel.mem.spec_c_worst_case_budget(),
    }


def run_sparsity_sweep(
    keep_ratios: list[float] = None,
    stage_policies: list[str] = None,
    node: int = 28,
    freq: int = 500,
) -> list[dict]:
    """Sweep over (keep_ratio, stage_policy) for paper tables."""
    keep_ratios = keep_ratios or [1.0, 0.85, 0.75, 0.6, 0.5]
    stage_policies = stage_policies or ["coarse_only", "all"]

    results = []
    for kr in keep_ratios:
        for sp in stage_policies:
            print(f"  sweep: kr={kr}, policy={sp}")
            sim_result = run_event_simulation(
                keep_ratio=kr, node=node, freq=freq, stage_policy=sp,
            )
            results.append({
                "node_nm": node,
                "freq_mhz": freq,
                "keep_ratio": kr,
                "stage_policy": sp,
                "total_cycles": sim_result["total_cycles"],
                "latency_ms": sim_result["latency_ms"],
                "fps": sim_result["fps"],
                "total_flops_G": sim_result["total_flops"] / 1e9,
                "num_ops": sim_result["dag_summary"]["num_ops"],
                "iterations": sim_result["simulation_iterations"],
                "systolic_array_busy_cycles": (
                    sim_result["detailed_breakdown"]["systolic_array"]["busy_cycles"]
                ),
            })
    return results


def main():
    print("=" * 70)
    print("EdgeStereoDAv2 Event-Driven Simulator: Full Analysis Suite")
    print("=" * 70)

    all_results = {}

    # 1. Per-frame timing at 28nm and 7nm
    print("\n[1] Event-driven simulation (dense baseline)...")
    for label, node, freq in [("28nm_500MHz", 28, 500), ("7nm_1GHz", 7, 1000)]:
        r = run_event_simulation(keep_ratio=1.0, node=node, freq=freq)
        print(f"  {label}: {r['fps']:.2f} FPS, {r['latency_ms']:.1f} ms, "
              f"{r['total_flops']/1e9:.1f} GFLOPs, {r['dag_summary']['num_ops']} ops")
        all_results[f"sim_{label}_dense"] = {
            "fps": r["fps"],
            "latency_ms": r["latency_ms"],
            "total_cycles": r["total_cycles"],
            "total_flops_G": r["total_flops"] / 1e9,
            "num_dag_ops": r["dag_summary"]["num_ops"],
        }

    # 2. Merge mode at 28nm
    print("\n[2] Event-driven simulation (merge kr=0.5)...")
    r = run_event_simulation(keep_ratio=0.5, node=28, freq=500)
    print(f"  28nm merge: {r['fps']:.2f} FPS, {r['latency_ms']:.1f} ms, "
          f"{r['dag_summary']['num_ops']} ops")
    all_results["sim_28nm_500MHz_merge"] = {
        "fps": r["fps"],
        "latency_ms": r["latency_ms"],
        "total_cycles": r["total_cycles"],
        "total_flops_G": r["total_flops"] / 1e9,
        "num_dag_ops": r["dag_summary"]["num_ops"],
    }

    # 3. Area + power
    print("\n[3] Area and power analysis...")
    for label, node, freq in [("28nm", 28, 500), ("7nm", 7, 1000)]:
        ap = run_area_power_analysis(node, freq)
        total_area = ap["area"]["Total"]
        total_power = ap["power"]["total_mw"]
        print(f"  {label}: area = {total_area:.3f} mm², power = {total_power:.1f} mW")
        all_results[f"area_power_{label}"] = {
            "area_mm2": {k: v for k, v in ap["area"].items() if isinstance(v, float)},
            "power_mw": {k: v for k, v in ap["power"].items()
                         if isinstance(v, (int, float))},
            "l2_budget": ap["l2_budget"],
        }

    # 4. Sparsity sweep
    print("\n[4] Sparsity sweep (5 keep_ratios x 2 policies)...")
    sweep = run_sparsity_sweep(node=28, freq=500)
    all_results["sparsity_sweep"] = sweep
    print("  kr  |  policy       |  FPS   | Latency | Ops")
    print("  " + "-" * 55)
    for r in sweep:
        print(f"  {r['keep_ratio']:.2f} | {r['stage_policy']:13s} | "
              f"{r['fps']:5.2f}  | {r['latency_ms']:6.2f} ms | {r['num_ops']:>4d}")

    # 5. Save results
    out_dir = os.path.join(_SIM_ROOT, "simulator", "results")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "simulation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Results written to {out_path}")

    # Separate sparsity file
    sparsity_path = os.path.join(out_dir, "sparsity_sweep.json")
    with open(sparsity_path, "w") as f:
        json.dump(sweep, f, indent=2, default=str)
    print(f"[OK] Sparsity sweep written to {sparsity_path}")


if __name__ == "__main__":
    main()
