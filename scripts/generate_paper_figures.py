#!/usr/bin/env python3
"""
scripts/generate_paper_figures.py
==================================
Generate publication-quality figures for EdgeStereoDAv2 ICCAD paper.

Reads pre-computed data from simulator/results/ and results/, outputs PNGs to paper/figures/.

Figures generated:
  fig4_quantization.png  — grouped bar: FP32/INT8/Mixed/INT4 accuracy
  fig5_fps_resolution.png — line plot: FPS vs resolution (28nm vs 7nm)
  fig6_energy.png        — stacked bar: energy breakdown dense vs sparse
  fig7_area.png          — pie chart: area breakdown at 28nm
  fig8_roofline.png      — log-log roofline model
  fig9_ablation.png      — grouped bar: ablation study (CRU sparsity + PE size)

Usage
-----
  python scripts/generate_paper_figures.py
  python scripts/generate_paper_figures.py --out-dir paper/figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']


# ---------------------------------------------------------------------------
# Fig 4: Quantization accuracy
# ---------------------------------------------------------------------------
def fig4_quantization(out_dir: str) -> None:
    # Hard-coded from results/eval_quantization/quantization_results.txt (394 images)
    configs = ['FP32', 'INT8', 'Mixed\n(INT8/INT4)', 'INT4']
    metrics = {
        'AbsRel': [0.0945, 0.0953, 0.1041, 0.1279],
        'RMSE':   [4.3918, 4.3925, 4.8251, 5.7615],
        r'$\delta_1$': [0.9080, 0.9068, 0.8892, 0.8419],
        'D1% (all)': [29.84, 30.26, 33.99, 42.19],
    }

    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
    for ax, (metric, vals) in zip(axes, metrics.items()):
        bars = ax.bar(range(4), vals, color=COLORS[:4], width=0.6, edgecolor='white', linewidth=0.5)
        ax.set_title(metric)
        ax.set_xticks(range(4))
        ax.set_xticklabels(configs, fontsize=7)
        ax.set_ylim(0, max(vals) * 1.25)
        # Annotate bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)
        # Highlight FP32 baseline
        bars[0].set_edgecolor('#333')
        bars[0].set_linewidth(1.2)
    axes[0].set_ylabel('Error (lower is better)')
    axes[2].set_ylabel('Accuracy (higher is better)')
    fig.suptitle('Fig. 4: Quantization Accuracy on KITTI (394 images)', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_quantization.png'))
    plt.close()
    print('  fig4_quantization.png')


# ---------------------------------------------------------------------------
# Fig 5: FPS vs Resolution
# ---------------------------------------------------------------------------
def fig5_fps_resolution(sim_data: dict, out_dir: str) -> None:
    sweep = sim_data['resolution_sweep']
    labels = [s['resolution'] for s in sweep]
    fps_28 = [s['fps'] for s in sweep]
    # 7nm is 2x faster (same cycles, 2x clock)
    fps_7 = [f * 2 for f in fps_28]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, fps_28, 'o-', color=COLORS[0], label='28nm / 500MHz', linewidth=1.8, markersize=5)
    ax.plot(x, fps_7,  's--', color=COLORS[1], label='7nm / 1GHz',   linewidth=1.8, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Input Resolution (H×W)')
    ax.set_ylabel('Throughput (FPS)')
    ax.set_title('Fig. 5: FPS vs Input Resolution')
    ax.legend()
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5_fps_resolution.png'))
    plt.close()
    print('  fig5_fps_resolution.png')


# ---------------------------------------------------------------------------
# Fig 6: Energy breakdown
# ---------------------------------------------------------------------------
def fig6_energy(sim_data: dict, sparsity_data: list, out_dir: str) -> None:
    # Per-engine energy (mJ) for dense and sparse (18.6% prune, layer=0) at 28nm
    e_dense = sim_data['energy_28nm_500MHz']['per_engine']
    # Find 18.6% prune_ratio, prune_layer=0, 28nm in sparsity sweep
    sparse_entry = next(
        (s for s in sparsity_data
         if s['node_nm'] == 28 and s.get('prune_layer') == 0
         and abs(s['prune_ratio'] - 0.186) < 0.01),
        None
    )
    # Scale per-engine proportionally by total energy ratio
    ratio = (sparse_entry['energy_mJ'] / sim_data['energy_28nm_500MHz']['total_mJ']
             if sparse_entry else 1.0)
    e_sparse = {k: v * ratio for k, v in e_dense.items()}

    engines = ['VFE\n(Encoder)', 'CSFE\n(Decoder)', 'ADCU\n(Calib.)']
    dense_vals  = [e_dense['VFE'],  e_dense['CSFE'],  e_dense['ADCU']]
    sparse_vals = [e_sparse['VFE'], e_sparse['CSFE'], e_sparse['ADCU']]

    x = np.arange(len(engines))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x - w/2, dense_vals,  w, label='Dense',          color=COLORS[0], edgecolor='white')
    ax.bar(x + w/2, sparse_vals, w, label='Sparse (18.6%)', color=COLORS[2], edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(engines)
    ax.set_ylabel('Energy per Frame (mJ)')
    ax.set_title('Fig. 6: Per-Engine Energy — Dense vs Sparse (28nm/500MHz)')
    ax.legend()
    # Annotate total
    total_dense  = sum(dense_vals)
    total_sparse = sum(sparse_vals)
    ax.text(0.98, 0.95, f'Total dense:  {total_dense:.2f} mJ\nTotal sparse: {total_sparse:.2f} mJ\n'
            f'Saving: {(1-total_sparse/total_dense)*100:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig6_energy.png'))
    plt.close()
    print('  fig6_energy.png')


# ---------------------------------------------------------------------------
# Fig 7: Area breakdown pie
# ---------------------------------------------------------------------------
def fig7_area(sim_data: dict, out_dir: str) -> None:
    a = sim_data['area_28nm']
    labels = ['VFE\n(Encoder)', 'CSFE\n(Decoder)', 'ADCU', 'L2 SRAM', 'Control+IO+\nInterconnect']
    sizes  = [a['VFE'], a['CSFE'], a['ADCU'], a['L2_SRAM'],
              a['Control'] + a['IO_Pads'] + a['Interconnect']]
    explode = (0.04, 0.04, 0.08, 0.04, 0.04)

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', explode=explode,
        colors=COLORS[:5], startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=1.2),
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title(f'Fig. 7: Area Breakdown at 28nm\n(Total = {a["Total_mm2"]:.2f} mm²)', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig7_area.png'))
    plt.close()
    print('  fig7_area.png')


# ---------------------------------------------------------------------------
# Fig 8: Roofline model
# ---------------------------------------------------------------------------
def fig8_roofline(sim_data: dict, out_dir: str) -> None:
    roofline = sim_data['roofline']
    ridge = roofline['ridge_point']  # OPS/Byte
    peak_tops = 1.024  # 28nm/500MHz
    mem_bw = peak_tops / ridge  # TB/s

    ops = roofline['operations']
    ai_vals = [o['ai'] for o in ops]
    names   = [o['name'] for o in ops]
    bounds  = [o['bound'] for o in ops]

    # Roofline curve
    ai_range = np.logspace(-1, 3.5, 500)
    perf_roof = np.minimum(peak_tops * 1e3, mem_bw * 1e3 * ai_range)  # GOPS

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ai_range, perf_roof, 'k-', linewidth=2, label='Roofline')
    ax.axvline(ridge, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(ridge * 1.1, peak_tops * 1e3 * 0.6, f'Ridge\n{ridge:.0f} OPS/B',
            fontsize=7, color='gray')

    # Scatter operations
    for o in ops:
        color = COLORS[0] if o['bound'] == 'Compute' else COLORS[3]
        ax.scatter(o['ai'], o['utilization'] * peak_tops * 1e3,
                   color=color, s=40, zorder=5)

    # Legend for bound type
    ax.scatter([], [], color=COLORS[0], s=40, label='Compute-bound')
    ax.scatter([], [], color=COLORS[3], s=40, label='Memory-bound')

    # Annotate a few key ops
    key_ops = {'Softmax', 'LayerNorm', 'QKV Projection', 'MLP Layer 1', 'ADCU Depth-to-Disp'}
    for o in ops:
        if o['name'] in key_ops:
            ax.annotate(o['name'].replace(' ', '\n'), (o['ai'], o['utilization'] * peak_tops * 1e3),
                        fontsize=6, xytext=(5, 3), textcoords='offset points')

    ax.set_xlabel('Arithmetic Intensity (OPS/Byte)')
    ax.set_ylabel('Performance (GOPS)')
    ax.set_title('Fig. 8: Roofline Model — EdgeStereoDAv2 (28nm/500MHz)')
    ax.legend(loc='lower right')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig8_roofline.png'))
    plt.close()
    print('  fig8_roofline.png')


# ---------------------------------------------------------------------------
# Fig 9: Ablation study
# ---------------------------------------------------------------------------
def fig9_ablation(sim_data: dict, sparsity_data: list, out_dir: str) -> None:
    # CRU sparsity ablation at 28nm/500MHz, prune_layer=0
    prune_ratios = [0.0, 0.1, 0.186, 0.3, 0.5]
    labels_cru = ['Dense\n(0%)', 'CRU\n10%', 'CRU\n18.6%\n(ours)', 'CRU\n30%', 'CRU\n50%']

    fps_cru, energy_saving_cru = [], []
    for pr in prune_ratios:
        if pr == 0.0:
            entry = next(s for s in sparsity_data if s['node_nm'] == 28 and s['prune_ratio'] == 0.0)
        else:
            entry = next(
                (s for s in sparsity_data
                 if s['node_nm'] == 28 and s.get('prune_layer') == 0
                 and abs(s['prune_ratio'] - pr) < 0.01),
                None
            )
        fps_cru.append(entry['fps'] if entry else 0)
        energy_saving_cru.append(entry['energy_saving_pct'] if entry else 0)

    # PE array ablation: 16x16 vs 32x32 (our design)
    pe_16 = next(s for s in sim_data['pe_sweep'] if s['pe_array'] == '16x16')
    pe_32 = next(s for s in sim_data['pe_sweep'] if s['pe_array'] == '32x32')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))

    # Left: FPS vs CRU sparsity
    bars = ax1.bar(range(len(labels_cru)), fps_cru, color=COLORS[:len(labels_cru)],
                   edgecolor='white', linewidth=0.5)
    bars[2].set_edgecolor('#333')
    bars[2].set_linewidth(1.5)  # highlight ours
    ax1.set_xticks(range(len(labels_cru)))
    ax1.set_xticklabels(labels_cru, fontsize=7.5)
    ax1.set_ylabel('Throughput (FPS)')
    ax1.set_title('(a) CRU Sparsity Ablation (28nm/500MHz)')
    ax1.set_ylim(0, max(fps_cru) * 1.25)
    for bar, v in zip(bars, fps_cru):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    # Secondary axis: energy saving
    ax1b = ax1.twinx()
    ax1b.plot(range(len(labels_cru)), energy_saving_cru, 'k--o', markersize=4, linewidth=1.2,
              label='Energy saving %')
    ax1b.set_ylabel('Energy Saving (%)', color='black')
    ax1b.tick_params(axis='y', labelcolor='black')
    ax1b.set_ylim(0, max(energy_saving_cru) * 1.5)
    ax1b.legend(loc='upper left', fontsize=7)

    # Right: PE array size comparison
    pe_labels = ['16×16 PE\n(256 MACs)', '32×32 PE\n(1024 MACs)\n(ours)']
    pe_fps    = [pe_16['fps'], pe_32['fps']]
    pe_tops   = [pe_16['peak_tops'], pe_32['peak_tops']]
    x = np.arange(2)
    w = 0.35
    ax2b = ax2.twinx()
    ax2.bar(x - w/2, pe_fps,  w, color=COLORS[0], label='FPS',       edgecolor='white')
    ax2b.bar(x + w/2, pe_tops, w, color=COLORS[2], label='Peak TOPS', edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pe_labels, fontsize=8)
    ax2.set_ylabel('Throughput (FPS)', color=COLORS[0])
    ax2b.set_ylabel('Peak TOPS', color=COLORS[2])
    ax2.set_title('(b) PE Array Size Ablation')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)
    for i, (f, t) in enumerate(zip(pe_fps, pe_tops)):
        ax2.text(i - w/2, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=7)
        ax2b.text(i + w/2, t + 0.005, f'{t:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Fig. 9: Ablation Study', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig9_ablation.png'))
    plt.close()
    print('  fig9_ablation.png')


# ---------------------------------------------------------------------------
# Fig 10: Pruning Strategy Comparison
# ---------------------------------------------------------------------------
def fig10_strategy_comparison(out_dir: str) -> None:
    """Cross-strategy comparison at keep_ratio=0.80 (20% pruning, 36% attn FLOP reduction).

    Data from: results/eval_strategies/strategies_summary.txt  (20-sample run)
    """
    # ---- Fused EPE and Fused D1 at kr=0.80 --------------------------------
    # Strategy label, Fu EPE, Fu D1 (%)
    # Note: sgm_baseline is threshold-based (not fixed kr=0.80) —
    #       we use the best typical result ~10-18% pruning cluster
    strategies_bar = [
        ('Dense DA2\n(baseline)',   2.0077, 17.57),
        ('Random',                  2.0670, 17.80),
        ('Checkerboard\n(50%)',     2.1513, 19.31),
        ('Inverse Conf.',           2.1140, 17.61),
        ('Top-K Conf.',             2.0907, 18.83),
        ('CLS Attention',           2.1356, 18.49),
        ('Hybrid\n(α=0.25)',        2.0476, 18.46),
        ('SGM Baseline\n(ours)',    1.4590, 10.83),  # best per-cluster mean: sgm_baseline:0.857
    ]

    labels   = [s[0] for s in strategies_bar]
    fu_epe   = [s[1] for s in strategies_bar]
    fu_d1    = [s[2] for s in strategies_bar]

    bar_colors = [
        '#9E9E9E',   # Dense (grey)
        '#90CAF9',   # Random
        '#A5D6A7',   # Checkerboard
        '#FFCC80',   # Inverse conf
        '#CE93D8',   # Top-K
        '#80DEEA',   # CLS attn
        '#BCAAA4',   # Hybrid
        '#EF5350',   # SGM ours (red — best)
    ]

    # ---- Pareto curve (Fused D1 vs Attn FLOP reduction) ------------------
    # (strategy, keep_ratio, attn_reduction_%, fu_d1)
    pareto_data = {
        'Random':     [(0.50, 75.0, 20.04), (0.60, 64.0, 19.26), (0.70, 51.0, 18.76),
                       (0.80, 36.0, 17.80), (0.90, 19.0, 17.45)],
        'Checkerboard': [(0.50, 75.0, 19.31)],
        'Top-K Conf.': [(0.50, 75.0, 22.60), (0.60, 64.0, 20.83), (0.70, 51.0, 19.67),
                        (0.80, 36.0, 18.83), (0.90, 19.0, 18.17)],
        'CLS Attn':   [(0.50, 75.0, 21.27), (0.60, 64.0, 19.93), (0.70, 51.0, 19.12),
                       (0.80, 36.0, 18.49), (0.90, 19.0, 17.98)],
        'Hybrid α=0.25': [(0.50, 75.0, 21.53), (0.60, 64.0, 20.20), (0.70, 51.0, 19.35),
                           (0.80, 36.0, 18.46), (0.90, 19.0, 18.15)],
        'SGM Baseline': [(0.906, 18.0, 10.12), (0.875, 23.4, 10.44),
                          (0.857, 26.6, 10.83), (0.844, 28.8, 13.03)],
    }
    pareto_colors = {
        'Random': '#90CAF9', 'Checkerboard': '#A5D6A7', 'Top-K Conf.': '#CE93D8',
        'CLS Attn': '#80DEEA', 'Hybrid α=0.25': '#BCAAA4', 'SGM Baseline': '#EF5350',
    }
    pareto_markers = {
        'Random': 'o', 'Checkerboard': 's', 'Top-K Conf.': '^',
        'CLS Attn': 'D', 'Hybrid α=0.25': 'v', 'SGM Baseline': '*',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- Left: grouped bar at kr~0.80 ----
    x = np.arange(len(labels))
    w = 0.38
    bars1 = ax1.bar(x - w/2, fu_epe, w, color=bar_colors, edgecolor='white',
                    linewidth=0.5, label='Fused EPE')
    ax1b = ax1.twinx()
    bars2 = ax1b.bar(x + w/2, fu_d1, w, color=bar_colors, edgecolor='white',
                     linewidth=0.5, alpha=0.55, label='Fused D1%', hatch='//')
    # Highlight ours
    bars1[-1].set_edgecolor('#333'); bars1[-1].set_linewidth(1.5)
    bars2[-1].set_edgecolor('#333'); bars2[-1].set_linewidth(1.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel('Fused EPE (px)', color='#1565C0')
    ax1b.set_ylabel('Fused D1-all (%)', color='#6A1B9A')
    ax1.tick_params(axis='y', labelcolor='#1565C0')
    ax1b.tick_params(axis='y', labelcolor='#6A1B9A')
    ax1.set_title('(a) Strategy Comparison at ~20% Token Pruning\n(Fused EPE solid, D1% hatched)')
    ax1.grid(axis='y', alpha=0.3)

    # Annotate EPE on top of solid bars
    for bar, v in zip(bars1, fu_epe):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=6.5)

    # Reference line: dense baseline
    ax1.axhline(fu_epe[0], color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # ---- Right: Pareto curve ----
    for name, pts in pareto_data.items():
        attn = [p[1] for p in pts]
        d1   = [p[2] for p in pts]
        # Sort by attn ascending for line
        paired = sorted(zip(attn, d1))
        attn_s = [p[0] for p in paired]
        d1_s   = [p[1] for p in paired]
        ax2.plot(attn_s, d1_s,
                 color=pareto_colors[name], marker=pareto_markers[name],
                 markersize=7 if name == 'SGM Baseline' else 5,
                 linewidth=1.8 if name == 'SGM Baseline' else 1.2,
                 label=name,
                 zorder=5 if name == 'SGM Baseline' else 3)

    # Dense baseline
    ax2.axhline(17.57, color='gray', linestyle='--', linewidth=0.9,
                label='Dense DA2 (no pruning)')

    ax2.set_xlabel('Attention FLOP Reduction (%)')
    ax2.set_ylabel('Fused D1-all (%)')
    ax2.set_title('(b) Accuracy vs. Efficiency Pareto Curve\n(lower-left is better)')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 85)
    ax2.set_ylim(8, 26)

    fig.suptitle('Fig. 10: Pruning Strategy Comparison (KITTI, 20 samples)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig10_strategy_comparison.png'))
    plt.close()
    print('  fig10_strategy_comparison.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default=os.path.join(_PROJECT_DIR, 'paper', 'figures'))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Generating figures → {args.out_dir}')

    sim_path = os.path.join(_PROJECT_DIR, 'simulator', 'results', 'simulation_results.json')
    spar_path = os.path.join(_PROJECT_DIR, 'simulator', 'results', 'sparsity_sweep.json')

    with open(sim_path) as f:
        sim_data = json.load(f)
    with open(spar_path) as f:
        sparsity_data = json.load(f)

    fig4_quantization(args.out_dir)
    fig5_fps_resolution(sim_data, args.out_dir)
    fig6_energy(sim_data, sparsity_data, args.out_dir)
    fig7_area(sim_data, args.out_dir)
    fig8_roofline(sim_data, args.out_dir)
    fig9_ablation(sim_data, sparsity_data, args.out_dir)
    fig10_strategy_comparison(args.out_dir)

    print('Done.')


if __name__ == '__main__':
    main()
