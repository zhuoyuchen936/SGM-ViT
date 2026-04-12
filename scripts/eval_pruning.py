#!/usr/bin/env python3
"""
scripts/eval_pruning.py
=======================
Progressive pruning ablation for SGM-ViT on KITTI.

Sweeps ``prune_layer`` (and optionally ``threshold``) to characterise the
tradeoff between attention FLOPs savings and depth accuracy.  The
``prune_layer`` parameter controls which ViT block GAS begins — blocks
``[0, prune_layer)`` run dense, blocks ``[prune_layer, 12)`` run GAS.

Efficiency strategy
-------------------
  1. SGM metrics          — computed ONCE per image  (invariant)
  2. Dense DA2 + align    — computed ONCE per image  (invariant)
  3. Token routing        — computed ONCE per threshold value
  4. GAS sparse DA2       — re-run per (prune_layer, threshold) combination

For the default single-threshold sweep only ``prune_layer`` varies, so
routing is done once per image.

Usage
-----
  python scripts/eval_pruning.py --max-samples 5        # quick sanity check
  python scripts/eval_pruning.py --max-samples 20       # quick ablation
  python scripts/eval_pruning.py                        # full 394-image sweep
  python scripts/eval_pruning.py --sweep-threshold      # cross-sweep
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import core._paths  # noqa: F401  — ensures DA2 is on sys.path
from core.eval_utils import compute_attn_reduction, pareto_frontier
from core.fusion import fuse_sgm_da2
from core.pipeline import (
    DA2_MODEL_CONFIGS,
    EMBED_DIM_MAP,
    TOKEN_GRID_SIZE,
    align_depth_to_sgm,
    load_da2_model,
    run_masked_sparse_da2,
)
from core.token_router import SGMConfidenceTokenRouter
from scripts.eval_kitti import (
    aggregate,
    build_sample_list,
    compute_metrics,
    load_protocol_confidence_maps,
    read_kitti_gt,
    read_pfm,
)
from scripts.common_config import (
    DEFAULT_ALIGN_CONF_THRESHOLD,
    DEFAULT_CONF_SIGMA,
    DEFAULT_DISPARITY_RANGE,
    DEFAULT_ENCODER,
    DEFAULT_KITTI_ROOT,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_PKRN_MIN_DIST,
    DEFAULT_PRUNE_LAYER,
    DEFAULT_PRUNE_THRESHOLD,
    DEFAULT_WEIGHTS,
    default_results_dir,
)

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_pruning(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, 'eval_pruning.log')),
            logging.StreamHandler(),
        ]
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )

    prune_layers = [int(x) for x in args.prune_layers.split(',')]
    if args.sweep_threshold:
        thresholds = [float(x) for x in args.thresholds.split(',')]
    else:
        thresholds = [args.threshold]

    total_configs = len(prune_layers) * len(thresholds)

    logging.info('=' * 66)
    logging.info('SGM-ViT  Progressive Pruning Ablation')
    logging.info('=' * 66)
    logging.info(f'  KITTI root      : {args.kitti_root}')
    logging.info(f'  Encoder         : {args.encoder}')
    logging.info(f'  Prune layers    : {prune_layers}')
    logging.info(f'  Thresholds      : {thresholds}')
    logging.info(f'  Total configs   : {total_configs}')
    logging.info(f'  Re-assembly     : {"on" if not args.no_reassembly else "off"}')
    logging.info(f'  Device          : {device}')
    logging.info(f'  Output          : {args.out_dir}')
    logging.info('=' * 66)

    # ---- Load DA2 model ------------------------------------------------
    logging.info(f'Loading DepthAnythingV2 ({args.encoder}) ...')
    model = load_da2_model(args.encoder, args.weights, device)

    embed_dim = EMBED_DIM_MAP[args.encoder]
    N         = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE

    n_blocks = len(list(model.pretrained.blocks))

    # ---- Sample list ---------------------------------------------------
    samples = build_sample_list(args.kitti_root)
    if not samples:
        logging.error('No samples found — check --kitti-root path.')
        return
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        logging.info(f'Limiting to {args.max_samples} samples (--max-samples)')
    logging.info(f'Total samples: {len(samples)}\n')

    # ---- Accumulators --------------------------------------------------
    # Key: (prune_layer, threshold_str) -> { split: [metrics_dicts] }
    # We track sparse and fused_sparse, both all-pixel and sv (SGM-valid)
    sparse_records:    dict[tuple, dict[str, list]] = {}
    sparse_records_sv: dict[tuple, dict[str, list]] = {}
    fused_records:     dict[tuple, dict[str, list]] = {}
    fused_records_sv:  dict[tuple, dict[str, list]] = {}
    timing_records:    dict[tuple, list[float]] = {}
    prune_ratio_records: dict[tuple, list[float]] = {}

    for pl in prune_layers:
        for th in thresholds:
            key = (pl, f"{th:.2f}")
            for d in (sparse_records, sparse_records_sv,
                      fused_records, fused_records_sv):
                d[key] = {'all': [], 'kitti15': [], 'kitti12': []}
            timing_records[key] = []
            prune_ratio_records[key] = []

    # Baselines (dense DA2 + fused SGM + dense DA2)
    dense_records    = {'all': [], 'kitti15': [], 'kitti12': []}
    dense_records_sv = {'all': [], 'kitti15': [], 'kitti12': []}
    sgm_records      = {'all': [], 'kitti15': [], 'kitti12': []}
    sgm_records_sv   = {'all': [], 'kitti15': [], 'kitti12': []}
    fused_dense_records    = {'all': [], 'kitti15': [], 'kitti12': []}
    fused_dense_records_sv = {'all': [], 'kitti15': [], 'kitti12': []}

    csv_rows: list[dict] = []

    for idx, s in enumerate(tqdm(samples, desc='Evaluating', unit='img')):

        # ---- Load inputs -----------------------------------------------
        left_bgr = cv2.imread(s['left'])
        if left_bgr is None:
            logging.warning(f"Cannot read image: {s['left']}"); continue

        gt_disp, valid_gt = read_kitti_gt(s['gt_disp'])
        sgm_disp          = read_pfm(s['sgm_pfm'])

        # Alignment confidence (binary LR-check, Gaussian-smoothed)
        conf_align, conf_route = load_protocol_confidence_maps(
            s,
            disparity_range=args.disparity_range,
            pkrn_min_dist=args.pkrn_min_dist,
            smooth_sigma=args.conf_sigma,
        )

        # SGM-valid mask
        sgm_for_mask = sgm_disp
        if sgm_disp.shape != gt_disp.shape:
            sgm_for_mask = cv2.resize(sgm_disp,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        valid_sgm = valid_gt & (sgm_for_mask > 0)

        # ---- SGM metrics (ONCE) ----------------------------------------
        sgm_metrics    = compute_metrics(sgm_for_mask, gt_disp, valid_gt)
        sgm_metrics_sv = compute_metrics(sgm_for_mask, gt_disp, valid_sgm)
        for k in ('all', s['split']):
            sgm_records[k].append(sgm_metrics)
            sgm_records_sv[k].append(sgm_metrics_sv)

        # ---- Dense DA2 (ONCE) ------------------------------------------
        depth_dense = model.infer_image(left_bgr, input_size=518)

        disp_dense, _, _ = align_depth_to_sgm(
            depth_mono     = depth_dense,
            disparity_raw  = sgm_disp,
            confidence_map = conf_align,
            conf_threshold = args.conf_threshold,
        )
        disp_dense_eval = disp_dense
        if disp_dense.shape != gt_disp.shape:
            disp_dense_eval = cv2.resize(disp_dense,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_LINEAR)

        dense_metrics    = compute_metrics(disp_dense_eval, gt_disp, valid_gt)
        dense_metrics_sv = compute_metrics(disp_dense_eval, gt_disp, valid_sgm)
        for k in ('all', s['split']):
            dense_records[k].append(dense_metrics)
            dense_records_sv[k].append(dense_metrics_sv)

        # ---- Fused SGM + Dense DA2 (ONCE) — baseline --------------------
        conf_for_fuse = conf_route
        if conf_route.shape != gt_disp.shape:
            conf_for_fuse = cv2.resize(conf_route,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_LINEAR)

        disp_fused_dense = fuse_sgm_da2(
            sgm_disp       = sgm_for_mask,
            da2_aligned    = disp_dense_eval,
            confidence_map = conf_for_fuse,
            conf_threshold = args.conf_threshold,
        )
        fused_d_metrics    = compute_metrics(disp_fused_dense, gt_disp, valid_gt)
        fused_d_metrics_sv = compute_metrics(disp_fused_dense, gt_disp, valid_sgm)
        for k in ('all', s['split']):
            fused_dense_records[k].append(fused_d_metrics)
            fused_dense_records_sv[k].append(fused_d_metrics_sv)

        # ---- Sweep: per threshold → per prune_layer ---------------------
        for th in thresholds:
            # Token routing (ONCE per threshold)
            router = SGMConfidenceTokenRouter(
                token_grid_size      = TOKEN_GRID_SIZE,
                confidence_threshold = th,
                learnable_threshold  = False,
            )
            conf_tensor  = torch.from_numpy(conf_route).unsqueeze(0).unsqueeze(0)
            dummy_tokens = torch.zeros(1, N, embed_dim)
            with torch.no_grad():
                routing = router(conf_tensor, dummy_tokens)

            prune_mask_1d = torch.zeros(N, dtype=torch.bool)
            prune_mask_1d[routing['prune_idx'][0]] = True
            prune_ratio = float(routing['prune_ratio'])
            n_keep = int((~prune_mask_1d).sum().item())

            for pl in prune_layers:
                key = (pl, f"{th:.2f}")

                # GAS sparse DA2 inference
                t0 = time.perf_counter()
                depth_sparse = run_masked_sparse_da2(
                    model         = model,
                    image_bgr     = left_bgr,
                    prune_mask    = prune_mask_1d,
                    input_size    = 518,
                    prune_layer   = pl,
                    do_reassembly = not args.no_reassembly,
                )
                t_sparse_ms = (time.perf_counter() - t0) * 1e3

                # Align sparse depth
                disp_sparse, _, _ = align_depth_to_sgm(
                    depth_mono     = depth_sparse,
                    disparity_raw  = sgm_disp,
                    confidence_map = conf_align,
                    conf_threshold = args.conf_threshold,
                )
                disp_sparse_eval = disp_sparse
                if disp_sparse.shape != gt_disp.shape:
                    disp_sparse_eval = cv2.resize(disp_sparse,
                        (gt_disp.shape[1], gt_disp.shape[0]),
                        interpolation=cv2.INTER_LINEAR)

                # Sparse metrics
                sp_metrics    = compute_metrics(disp_sparse_eval, gt_disp, valid_gt)
                sp_metrics_sv = compute_metrics(disp_sparse_eval, gt_disp, valid_sgm)

                # Fused SGM + Sparse DA2
                disp_fused_sp = fuse_sgm_da2(
                    sgm_disp       = sgm_for_mask,
                    da2_aligned    = disp_sparse_eval,
                    confidence_map = conf_for_fuse,
                    conf_threshold = args.conf_threshold,
                )
                fused_sp_metrics    = compute_metrics(disp_fused_sp, gt_disp, valid_gt)
                fused_sp_metrics_sv = compute_metrics(disp_fused_sp, gt_disp, valid_sgm)

                # Accumulate
                attn_red = compute_attn_reduction(pl, n_keep, N, n_blocks)

                for k in ('all', s['split']):
                    sparse_records[key][k].append(sp_metrics)
                    sparse_records_sv[key][k].append(sp_metrics_sv)
                    fused_records[key][k].append(fused_sp_metrics)
                    fused_records_sv[key][k].append(fused_sp_metrics_sv)

                timing_records[key].append(t_sparse_ms)
                prune_ratio_records[key].append(prune_ratio)

                # CSV row
                csv_rows.append({
                    'idx':              idx,
                    'split':            s['split'],
                    'image':            os.path.basename(s['left']),
                    'prune_layer':      pl,
                    'threshold':        f"{th:.2f}",
                    'prune_pct':        f"{prune_ratio*100:.1f}",
                    'attn_reduction':   f"{attn_red*100:.1f}",
                    'sparse_epe':       f"{sp_metrics['epe']:.4f}",
                    'sparse_d1':        f"{sp_metrics['d1']:.4f}",
                    'sparse_epe_sv':    f"{sp_metrics_sv['epe']:.4f}",
                    'sparse_d1_sv':     f"{sp_metrics_sv['d1']:.4f}",
                    'fused_epe':        f"{fused_sp_metrics['epe']:.4f}",
                    'fused_d1':         f"{fused_sp_metrics['d1']:.4f}",
                    'fused_epe_sv':     f"{fused_sp_metrics_sv['epe']:.4f}",
                    'fused_d1_sv':      f"{fused_sp_metrics_sv['d1']:.4f}",
                    'time_ms':          f"{t_sparse_ms:.1f}",
                })

        # ---- Periodic console update -----------------------------------
        if (idx + 1) % max(1, len(samples) // 10) == 0 or idx < 3:
            logging.info(
                f"[{idx+1:3d}/{len(samples)}]  "
                f"SGM EPE={sgm_metrics['epe']:.2f}  "
                f"Dense EPE={dense_metrics['epe']:.2f}"
            )

    # ====================================================================
    # Aggregate and report
    # ====================================================================
    n_samples = len(samples)
    tag_map = {'all': 'COMBINED', 'kitti15': 'KITTI-2015', 'kitti12': 'KITTI-2012'}

    def _fmt_split_table(split: str) -> str:
        a_sgm    = aggregate(sgm_records[split])
        a_sgm_sv = aggregate(sgm_records_sv[split])
        a_dense  = aggregate(dense_records[split])
        a_dense_sv = aggregate(dense_records_sv[split])
        a_fused_d  = aggregate(fused_dense_records[split])
        a_fused_d_sv = aggregate(fused_dense_records_sv[split])

        if a_sgm['n'] == 0:
            return ''

        tag = tag_map[split]
        sep = '-' * 120
        lines = [
            f"\n  {tag}  ({a_sgm['n']} images)",
            f"  {'Config':<24s}  {'Prune%':>6s}  {'Attn↓%':>6s}"
            f"  {'Sp EPE':>8s}  {'Sp D1%':>8s}  {'Fu EPE':>8s}  {'Fu D1%':>8s}"
            f"  {'Sp EPE(sv)':>10s}  {'Sp D1%(sv)':>10s}  {'Fu EPE(sv)':>10s}  {'Fu D1%(sv)':>10s}"
            f"  {'Time(ms)':>8s}",
            f"  {sep}",
            # Baselines
            f"  {'SGM (baseline)':<24s}  {'':>6s}  {'':>6s}"
            f"  {a_sgm['epe']:>8.4f}  {a_sgm['d1']:>8.4f}  {'':>8s}  {'':>8s}"
            f"  {a_sgm_sv['epe']:>10.4f}  {a_sgm_sv['d1']:>10.4f}  {'':>10s}  {'':>10s}"
            f"  {'':>8s}",
            f"  {'Dense DA2 + align':<24s}  {'':>6s}  {'':>6s}"
            f"  {a_dense['epe']:>8.4f}  {a_dense['d1']:>8.4f}"
            f"  {a_fused_d['epe']:>8.4f}  {a_fused_d['d1']:>8.4f}"
            f"  {a_dense_sv['epe']:>10.4f}  {a_dense_sv['d1']:>10.4f}"
            f"  {a_fused_d_sv['epe']:>10.4f}  {a_fused_d_sv['d1']:>10.4f}"
            f"  {'':>8s}",
            f"  {sep}",
        ]

        # Per-config rows, sorted by (threshold, prune_layer)
        for th in thresholds:
            for pl in prune_layers:
                key = (pl, f"{th:.2f}")
                a_sp    = aggregate(sparse_records[key][split])
                a_sp_sv = aggregate(sparse_records_sv[key][split])
                a_fu    = aggregate(fused_records[key][split])
                a_fu_sv = aggregate(fused_records_sv[key][split])

                if a_sp['n'] == 0:
                    continue

                pr_list = prune_ratio_records[key]
                avg_prune = np.mean(pr_list) * 100 if pr_list else 0.0
                n_keep_avg = N * (1.0 - np.mean(pr_list)) if pr_list else N
                attn_red = compute_attn_reduction(pl, n_keep_avg, N, n_blocks) * 100
                avg_time = np.mean(timing_records[key]) if timing_records[key] else 0.0

                label = f"PL={pl:2d} θ={th:.2f}"
                lines.append(
                    f"  {label:<24s}"
                    f"  {avg_prune:>5.1f}%  {attn_red:>5.1f}%"
                    f"  {a_sp['epe']:>8.4f}  {a_sp['d1']:>8.4f}"
                    f"  {a_fu['epe']:>8.4f}  {a_fu['d1']:>8.4f}"
                    f"  {a_sp_sv['epe']:>10.4f}  {a_sp_sv['d1']:>10.4f}"
                    f"  {a_fu_sv['epe']:>10.4f}  {a_fu_sv['d1']:>10.4f}"
                    f"  {avg_time:>7.1f}"
                )

        lines.append('')
        return '\n'.join(lines)

    def _best_config_table(split: str = 'all') -> str:
        """Best threshold per prune_layer (by fused EPE)."""
        lines = [
            f"\n  Best configuration per prune_layer (by Fused EPE, {tag_map.get(split, split)}):",
            f"  {'PL':>4s}  {'Thresh':>6s}  {'Prune%':>6s}  {'Attn↓%':>6s}"
            f"  {'Sp EPE':>8s}  {'Sp D1%':>8s}  {'Fu EPE':>8s}  {'Fu D1%':>8s}"
            f"  {'Time(ms)':>8s}",
            f"  {'-'*80}",
        ]

        for pl in prune_layers:
            best_epe = float('inf')
            best_row = None
            for th in thresholds:
                key = (pl, f"{th:.2f}")
                a_fu = aggregate(fused_records[key][split])
                if a_fu['n'] > 0 and a_fu['epe'] < best_epe:
                    a_sp = aggregate(sparse_records[key][split])
                    pr_list = prune_ratio_records[key]
                    avg_prune = np.mean(pr_list) * 100 if pr_list else 0.0
                    n_keep_avg = N * (1.0 - np.mean(pr_list)) if pr_list else N
                    attn_red = compute_attn_reduction(pl, n_keep_avg, N, n_blocks) * 100
                    avg_time = np.mean(timing_records[key]) if timing_records[key] else 0.0
                    best_epe = a_fu['epe']
                    best_row = (pl, th, avg_prune, attn_red, a_sp, a_fu, avg_time)

            if best_row:
                pl_, th_, pr_, ar_, a_sp_, a_fu_, t_ = best_row
                lines.append(
                    f"  {pl_:>4d}  {th_:>6.2f}  {pr_:>5.1f}%  {ar_:>5.1f}%"
                    f"  {a_sp_['epe']:>8.4f}  {a_sp_['d1']:>8.4f}"
                    f"  {a_fu_['epe']:>8.4f}  {a_fu_['d1']:>8.4f}"
                    f"  {t_:>7.1f}"
                )

        lines.append('')
        return '\n'.join(lines)

    def _pareto_table(split: str = 'all') -> str:
        """Pareto-optimal operating points (best EPE for each FLOPs budget)."""
        points = []
        for pl in prune_layers:
            for th in thresholds:
                key = (pl, f"{th:.2f}")
                a_fu = aggregate(fused_records[key][split])
                if a_fu['n'] == 0:
                    continue
                a_sp = aggregate(sparse_records[key][split])
                pr_list = prune_ratio_records[key]
                avg_prune = np.mean(pr_list) * 100 if pr_list else 0.0
                n_keep_avg = N * (1.0 - np.mean(pr_list)) if pr_list else N
                attn_red = compute_attn_reduction(pl, n_keep_avg, N, n_blocks)
                avg_time = np.mean(timing_records[key]) if timing_records[key] else 0.0
                points.append({
                    'pl': pl, 'th': th,
                    'prune_pct': avg_prune,
                    'attn_reduction': attn_red,
                    'sparse_epe': a_sp['epe'],
                    'fused_epe': a_fu['epe'],
                    'fused_d1': a_fu['d1'],
                    'time_ms': avg_time,
                })

        frontier = pareto_frontier(points)
        if not frontier:
            return ''

        lines = [
            f"\n  Pareto-optimal operating points ({tag_map.get(split, split)}):",
            f"  {'PL':>4s}  {'Thresh':>6s}  {'Prune%':>6s}  {'Attn↓%':>6s}"
            f"  {'Sp EPE':>8s}  {'Fu EPE':>8s}  {'Fu D1%':>8s}  {'Time(ms)':>8s}",
            f"  {'-'*70}",
        ]
        for p in frontier:
            lines.append(
                f"  {p['pl']:>4d}  {p['th']:>6.2f}  {p['prune_pct']:>5.1f}%"
                f"  {p['attn_reduction']*100:>5.1f}%"
                f"  {p['sparse_epe']:>8.4f}  {p['fused_epe']:>8.4f}"
                f"  {p['fused_d1']:>8.4f}  {p['time_ms']:>7.1f}"
            )
        lines.append('')
        return '\n'.join(lines)

    # ---- Build and emit report ----------------------------------------
    header = f"\n{'='*66}\n  SGM-ViT Progressive Pruning Ablation Results\n{'='*66}"
    logging.info(header)

    result_lines = [header]
    for split in ['all', 'kitti15', 'kitti12']:
        tbl = _fmt_split_table(split)
        if tbl:
            logging.info(tbl)
            result_lines.append(tbl)

    best = _best_config_table('all')
    logging.info(best)
    result_lines.append(best)

    pareto = _pareto_table('all')
    logging.info(pareto)
    result_lines.append(pareto)

    timing_info = f"\n  Samples evaluated: {n_samples}\n"
    logging.info(timing_info)
    result_lines.append(timing_info)

    # ---- Save results --------------------------------------------------
    txt_path = os.path.join(args.out_dir, 'pruning_summary.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(result_lines))
    logging.info(f"Summary: {txt_path}")

    csv_path = os.path.join(args.out_dir, 'pruning_sweep.csv')
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
        description='SGM-ViT progressive pruning ablation on KITTI.'
    )
    p.add_argument('--kitti-root', default=DEFAULT_KITTI_ROOT,
        help='KITTI dataset root (contains training/ and kitti2012/).')
    p.add_argument('--weights',
        default=DEFAULT_WEIGHTS,
        help='Path to DepthAnythingV2 checkpoint.')
    p.add_argument('--encoder', default=DEFAULT_ENCODER,
        choices=list(DA2_MODEL_CONFIGS.keys()))
    p.add_argument('--prune-layers', default='0,2,4,6,8,10',
        help='Comma-separated prune_layer values to sweep (default: 0,2,4,6,8,10).')
    p.add_argument('--threshold', type=float, default=DEFAULT_PRUNE_THRESHOLD,
        help='Token pruning confidence threshold θ (used when not --sweep-threshold).')
    p.add_argument('--sweep-threshold', action='store_true',
        help='Also sweep threshold values (cross-product with prune_layers).')
    p.add_argument('--thresholds', default='0.50,0.55,0.60,0.65,0.70,0.75',
        help='Comma-separated threshold values for --sweep-threshold mode.')
    p.add_argument('--no-reassembly', action='store_true',
        help='Disable token re-assembly before the DPT decoder.')
    p.add_argument('--conf-threshold', type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD,
        help='Minimum confidence to use a pixel for depth alignment.')
    p.add_argument('--conf-sigma', type=float, default=DEFAULT_CONF_SIGMA,
        help='Gaussian σ for smoothing the PKRN confidence map.')
    p.add_argument('--disparity-range', type=int, default=DEFAULT_DISPARITY_RANGE,
        help='SGM disparity search range (for PKRN cache miss).')
    p.add_argument('--pkrn-min-dist', type=int, default=DEFAULT_PKRN_MIN_DIST,
        help='Minimum disparity gap from winner for PKRN second-best search.')
    p.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES,
        help='Evaluate only the first N samples (0 = all).')
    p.add_argument('--out-dir',
        default=default_results_dir('eval_pruning'),
        help='Directory to save results and log.')
    p.add_argument('--cpu', action='store_true',
        help='Force CPU inference.')
    return p.parse_args()


if __name__ == '__main__':
    evaluate_pruning(parse_args())
