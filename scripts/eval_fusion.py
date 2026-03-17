#!/usr/bin/env python3
"""
scripts/eval_fusion.py
======================
Systematic evaluation of output-level SGM + DA2 fusion strategies on KITTI.

Runs Dense DA2 inference + alignment ONCE per image, then sweeps all fusion
strategies across parameter grids.  Fusion itself is pure NumPy (~1 ms each),
so the sweep adds negligible overhead.

Evaluated strategies
--------------------
  1. soft_blend     — α = clip(conf/θ, 0, 1); current baseline
  2. hard_switch    — binary: SGM if conf ≥ θ AND sgm > 0, else DA2
  3. outlier_aware  — soft_blend attenuated when |SGM − DA2| > outlier_threshold
  4. two_threshold  — dead-zone cascade: DA2 below θ_low, SGM above θ_high, blend between

Usage
-----
  python scripts/eval_fusion.py                          # full 394-image sweep
  python scripts/eval_fusion.py --max-samples 20         # quick sanity check
  python scripts/eval_fusion.py --strategies soft_blend hard_switch
  python scripts/eval_fusion.py --kitti-root /your/path
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_DA2_DIR     = os.path.join(_PROJECT_DIR, "Depth-Anything-V2")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _DA2_DIR)

from demo import (
    load_da2_model, align_depth_to_sgm,
    fuse_dispatch, FUSION_STRATEGIES,
    DA2_MODEL_CONFIGS,
)
from scripts.eval_kitti import (
    build_sample_list, read_pfm, read_kitti_gt,
    compute_metrics, aggregate, load_pkrn_confidence,
)


# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------

def build_sweep_grid(strategies: list[str]) -> dict[str, list[dict]]:
    """
    Build a list of kwarg dicts for each strategy to sweep.

    Returns { strategy_name: [ {param: value, ...}, ... ] }
    """
    ct_grid = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]

    grids: dict[str, list[dict]] = {}

    if 'soft_blend' in strategies:
        grids['soft_blend'] = [
            {'conf_threshold': ct} for ct in ct_grid
        ]

    if 'hard_switch' in strategies:
        grids['hard_switch'] = [
            {'conf_threshold': ct} for ct in ct_grid
        ]

    if 'outlier_aware' in strategies:
        grids['outlier_aware'] = [
            {'conf_threshold': ct, 'outlier_threshold': ot}
            for ct in [0.3, 0.5, 0.7]
            for ot in [5.0, 10.0, 15.0, 20.0, 30.0]
        ]

    if 'two_threshold' in strategies:
        grids['two_threshold'] = [
            {'theta_low': tl, 'theta_high': th}
            for tl in [0.05, 0.1, 0.2, 0.3, 0.4]
            for th in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            if tl < th
        ]

    return grids


def params_label(strategy: str, kwargs: dict) -> str:
    """Human-readable one-line label for a parameter configuration."""
    if strategy in ('soft_blend', 'hard_switch'):
        return f"θ={kwargs['conf_threshold']:.2f}"
    elif strategy == 'outlier_aware':
        return f"θ={kwargs['conf_threshold']:.2f},ot={kwargs['outlier_threshold']:.0f}"
    elif strategy == 'two_threshold':
        return f"θl={kwargs['theta_low']:.2f},θh={kwargs['theta_high']:.2f}"
    return str(kwargs)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_fusion(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, 'eval_fusion.log')),
            logging.StreamHandler(),
        ]
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )

    strategies = args.strategies
    sweep = build_sweep_grid(strategies)
    total_configs = sum(len(v) for v in sweep.values())

    logging.info('=' * 66)
    logging.info('SGM-ViT  Fusion Strategy Sweep')
    logging.info('=' * 66)
    logging.info(f'  KITTI root     : {args.kitti_root}')
    logging.info(f'  Encoder        : {args.encoder}')
    logging.info(f'  Align conf_thr : {args.align_conf_threshold}')
    logging.info(f'  Device         : {device}')
    logging.info(f'  Strategies     : {strategies}')
    logging.info(f'  Total configs  : {total_configs}')
    logging.info(f'  Output         : {args.out_dir}')
    logging.info('=' * 66)

    # ---- Load DA2 model ------------------------------------------------
    model = load_da2_model(args.encoder, args.weights, device)

    # ---- Sample list ---------------------------------------------------
    samples = build_sample_list(args.kitti_root)
    if not samples:
        logging.error('No samples found.'); return
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    logging.info(f'Total samples: {len(samples)}\n')

    # ---- Accumulators --------------------------------------------------
    # Per-(strategy, params_str) -> { 'all': [metrics], 'kitti15': [...], 'kitti12': [...] }
    agg_records: dict[tuple[str, str], dict[str, list]] = {}
    for strat, grid in sweep.items():
        for kw in grid:
            key = (strat, params_label(strat, kw))
            agg_records[key] = {'all': [], 'kitti15': [], 'kitti12': []}

    # Baselines
    sgm_records  = {'all': [], 'kitti15': [], 'kitti12': []}
    da2_records  = {'all': [], 'kitti15': [], 'kitti12': []}
    # sv (SGM-valid) variants
    sgm_records_sv = {'all': [], 'kitti15': [], 'kitti12': []}
    da2_records_sv = {'all': [], 'kitti15': [], 'kitti12': []}
    agg_records_sv: dict[tuple[str, str], dict[str, list]] = {}
    for strat, grid in sweep.items():
        for kw in grid:
            key = (strat, params_label(strat, kw))
            agg_records_sv[key] = {'all': [], 'kitti15': [], 'kitti12': []}

    csv_rows: list[dict] = []
    t_da2_total = 0.0

    for idx, s in enumerate(tqdm(samples, desc='Evaluating', unit='img')):
        # ---- Load inputs -----------------------------------------------
        left_bgr = cv2.imread(s['left'])
        if left_bgr is None:
            logging.warning(f"Skip: {s['left']}"); continue

        gt_disp, valid_gt = read_kitti_gt(s['gt_disp'])
        sgm_disp          = read_pfm(s['sgm_pfm'])

        # Alignment confidence (binary LR-check mask, Gaussian-smoothed)
        mismatch  = np.load(s['mismatch']).astype(bool)
        occlusion = np.load(s['occlusion']).astype(bool)
        conf_align = (~(mismatch | occlusion)).astype(np.float32)
        conf_align = gaussian_filter(conf_align, sigma=args.conf_sigma)

        # Routing / fusion confidence (PKRN)
        if os.path.exists(s.get('pkrn_cache', '')) or os.path.exists(s.get('right', '')):
            conf_fuse = load_pkrn_confidence(
                s, args.disparity_range, args.pkrn_min_dist, args.conf_sigma)
        else:
            conf_fuse = conf_align

        # SGM-valid mask
        sgm_for_eval = sgm_disp
        if sgm_disp.shape != gt_disp.shape:
            sgm_for_eval = cv2.resize(sgm_disp,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        valid_sgm = valid_gt & (sgm_for_eval > 0)

        # ---- SGM baseline -----------------------------------------------
        m_sgm    = compute_metrics(sgm_for_eval, gt_disp, valid_gt)
        m_sgm_sv = compute_metrics(sgm_for_eval, gt_disp, valid_sgm)
        for k in ('all', s['split']):
            sgm_records[k].append(m_sgm)
            sgm_records_sv[k].append(m_sgm_sv)

        # ---- Dense DA2 + alignment (ONCE) --------------------------------
        t0 = time.perf_counter()
        depth_dense = model.infer_image(left_bgr, input_size=518)
        t_da2_total += time.perf_counter() - t0

        disp_da2, _, _ = align_depth_to_sgm(
            depth_mono     = depth_dense,
            disparity_raw  = sgm_disp,
            confidence_map = conf_align,
            conf_threshold = args.align_conf_threshold,
        )

        # Resize to GT shape for evaluation
        if disp_da2.shape != gt_disp.shape:
            disp_da2_eval = cv2.resize(disp_da2,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_LINEAR)
        else:
            disp_da2_eval = disp_da2

        m_da2    = compute_metrics(disp_da2_eval, gt_disp, valid_gt)
        m_da2_sv = compute_metrics(disp_da2_eval, gt_disp, valid_sgm)
        for k in ('all', s['split']):
            da2_records[k].append(m_da2)
            da2_records_sv[k].append(m_da2_sv)

        # Resize confidence for fusion (match da2_eval resolution = gt resolution)
        conf_for_fuse = conf_fuse
        sgm_for_fuse  = sgm_for_eval
        if conf_fuse.shape != gt_disp.shape:
            conf_for_fuse = cv2.resize(conf_fuse,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_LINEAR)

        # ---- Sweep fusion strategies ------------------------------------
        row_base = {
            'idx':   idx,
            'split': s['split'],
            'image': os.path.basename(s['left']),
            'sgm_epe': f"{m_sgm['epe']:.4f}",
            'sgm_d1':  f"{m_sgm['d1']:.4f}",
            'da2_epe': f"{m_da2['epe']:.4f}",
            'da2_d1':  f"{m_da2['d1']:.4f}",
        }

        for strat, grid in sweep.items():
            for kw in grid:
                fused = fuse_dispatch(
                    strategy       = strat,
                    sgm_disp       = sgm_for_fuse,
                    da2_aligned    = disp_da2_eval,
                    confidence_map = conf_for_fuse,
                    **kw,
                )
                m_all = compute_metrics(fused, gt_disp, valid_gt)
                m_sv  = compute_metrics(fused, gt_disp, valid_sgm)

                plabel = params_label(strat, kw)
                key = (strat, plabel)
                for k in ('all', s['split']):
                    agg_records[key][k].append(m_all)
                    agg_records_sv[key][k].append(m_sv)

                csv_rows.append({
                    **row_base,
                    'strategy':  strat,
                    'params':    plabel,
                    'fused_epe_all': f"{m_all['epe']:.4f}",
                    'fused_d1_all':  f"{m_all['d1']:.4f}",
                    'fused_epe_sv':  f"{m_sv['epe']:.4f}",
                    'fused_d1_sv':   f"{m_sv['d1']:.4f}",
                })

        # ---- Periodic log -------------------------------------------------
        if (idx + 1) % max(1, len(samples) // 10) == 0 or idx < 3:
            logging.info(
                f"[{idx+1:3d}/{len(samples)}]  "
                f"SGM EPE={m_sgm['epe']:.2f}  "
                f"DA2 EPE={m_da2['epe']:.2f}"
            )

    # ====================================================================
    # Aggregate and report
    # ====================================================================
    n_samples = len(samples)

    # Helper to format a table for one split
    def _fmt_table(split: str) -> str:
        a_sgm    = aggregate(sgm_records[split])
        a_da2    = aggregate(da2_records[split])
        a_sgm_sv = aggregate(sgm_records_sv[split])
        a_da2_sv = aggregate(da2_records_sv[split])
        if a_sgm['n'] == 0:
            return ''

        tag = {'all': 'COMBINED', 'kitti15': 'KITTI-2015', 'kitti12': 'KITTI-2012'}[split]
        lines = [
            f"\n  {tag}  ({a_sgm['n']} images)",
            f"  {'Method':<30s}  {'Params':<22s}  {'EPE(all)':>9s}  {'D1%(all)':>9s}"
            f"  {'EPE(sv)':>9s}  {'D1%(sv)':>9s}",
            f"  {'-'*110}",
            f"  {'SGM (baseline)':<30s}  {'':22s}"
            f"  {a_sgm['epe']:>9.4f}  {a_sgm['d1']:>9.4f}"
            f"  {a_sgm_sv['epe']:>9.4f}  {a_sgm_sv['d1']:>9.4f}",
            f"  {'Dense DA2 + align':<30s}  {'':22s}"
            f"  {a_da2['epe']:>9.4f}  {a_da2['d1']:>9.4f}"
            f"  {a_da2_sv['epe']:>9.4f}  {a_da2_sv['d1']:>9.4f}",
            f"  {'-'*110}",
        ]

        # Group fusion results by strategy, sorted by EPE(all)
        fusion_rows: list[tuple[str, str, dict, dict]] = []
        for (strat, plabel), rec in agg_records.items():
            a = aggregate(rec[split])
            a_sv = aggregate(agg_records_sv[(strat, plabel)][split])
            if a['n'] > 0:
                fusion_rows.append((strat, plabel, a, a_sv))

        # Sort by EPE(all) ascending
        fusion_rows.sort(key=lambda r: r[2]['epe'])

        for strat, plabel, a, a_sv in fusion_rows:
            lines.append(
                f"  {strat:<30s}  {plabel:<22s}"
                f"  {a['epe']:>9.4f}  {a['d1']:>9.4f}"
                f"  {a_sv['epe']:>9.4f}  {a_sv['d1']:>9.4f}"
            )

        lines.append('')
        return '\n'.join(lines)

    # Best config per strategy
    def _best_per_strategy(split: str = 'all') -> str:
        lines = [
            f"\n  Best configuration per strategy (by EPE-all, {split}):",
            f"  {'Strategy':<20s}  {'Params':<22s}  {'EPE(all)':>9s}  {'D1%(all)':>9s}"
            f"  {'EPE(sv)':>9s}  {'D1%(sv)':>9s}",
            f"  {'-'*90}",
        ]
        seen = set()
        # Collect best per strategy
        for strat in strategies:
            best_epe = float('inf')
            best_row = None
            for (s, pl), rec in agg_records.items():
                if s != strat:
                    continue
                a = aggregate(rec[split])
                if a['n'] > 0 and a['epe'] < best_epe:
                    best_epe = a['epe']
                    a_sv = aggregate(agg_records_sv[(s, pl)][split])
                    best_row = (strat, pl, a, a_sv)
            if best_row and best_row[0] not in seen:
                seen.add(best_row[0])
                strat, pl, a, a_sv = best_row
                lines.append(
                    f"  {strat:<20s}  {pl:<22s}"
                    f"  {a['epe']:>9.4f}  {a['d1']:>9.4f}"
                    f"  {a_sv['epe']:>9.4f}  {a_sv['d1']:>9.4f}"
                )
        lines.append('')
        return '\n'.join(lines)

    header = f"\n{'='*66}\n  SGM-ViT Fusion Strategy Sweep Results\n{'='*66}"
    logging.info(header)

    result_lines = [header]
    for split in ['all', 'kitti15', 'kitti12']:
        tbl = _fmt_table(split)
        if tbl:
            logging.info(tbl)
            result_lines.append(tbl)

    best = _best_per_strategy('all')
    logging.info(best)
    result_lines.append(best)

    timing = (
        f"\n  Timing ({n_samples} images):\n"
        f"    Dense DA2 total: {t_da2_total:.1f}s  avg: {t_da2_total/max(1,n_samples)*1e3:.1f}ms/img\n"
        f"    Fusion configs : {total_configs} per image  (~{total_configs*0.001:.1f}ms overhead)\n"
    )
    logging.info(timing)
    result_lines.append(timing)

    # ---- Save results --------------------------------------------------
    txt_path = os.path.join(args.out_dir, 'fusion_summary.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(result_lines))
    logging.info(f"Summary: {txt_path}")

    csv_path = os.path.join(args.out_dir, 'fusion_sweep.csv')
    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logging.info(f"Per-image CSV: {csv_path}")

    logging.info('=' * 66)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='SGM-ViT fusion strategy sweep on KITTI.'
    )
    p.add_argument('--kitti-root', default='/nfs/usrhome/pdongaa/dataczy/kitti')
    p.add_argument('--weights',
        default=os.path.join(_PROJECT_DIR, 'Depth-Anything-V2', 'checkpoints',
                             'depth_anything_v2_vits.pth'))
    p.add_argument('--encoder', default='vits',
        choices=list(DA2_MODEL_CONFIGS.keys()))
    p.add_argument('--strategies', nargs='+',
        default=list(FUSION_STRATEGIES.keys()),
        help='Which strategies to evaluate.')
    p.add_argument('--align-conf-threshold', type=float, default=0.7,
        help='Confidence threshold for alignment regression (not fusion).')
    p.add_argument('--conf-sigma', type=float, default=5.0,
        help='Gaussian sigma for confidence smoothing.')
    p.add_argument('--disparity-range', type=int, default=128)
    p.add_argument('--pkrn-min-dist', type=int, default=1)
    p.add_argument('--max-samples', type=int, default=0,
        help='Limit to first N samples (0 = all).')
    p.add_argument('--out-dir',
        default=os.path.join(_PROJECT_DIR, 'results', 'eval_fusion'))
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    evaluate_fusion(parse_args())
