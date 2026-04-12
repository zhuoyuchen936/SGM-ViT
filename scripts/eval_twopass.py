#!/usr/bin/env python3
"""
scripts/eval_twopass.py
========================
Compare original GAS vs Two-Pass GAS sparse attention on KITTI.

Two-Pass GAS adds a cross-attention pass where pruned tokens attend to kept
tokens (Q=pruned, KV=kept), restoring global context without full N² cost.

Outputs:
  - results/eval_twopass/twopass_results.txt   — aggregated metrics table
  - results/eval_twopass/twopass_per_image.csv — per-image metrics
  - results/eval_twopass/comparison.png        — visual comparison figure

Usage
-----
  python scripts/eval_twopass.py --max-samples 5    # quick sanity check
  python scripts/eval_twopass.py --max-samples 20   # fast ablation
  python scripts/eval_twopass.py                     # full 394-image benchmark
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
_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

import sys

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import core._paths  # noqa: F401, E402
from core.eval_utils import compute_attn_reduction
from core.fusion import fuse_sgm_da2
from core.pipeline import (
    DA2_MODEL_CONFIGS,
    EMBED_DIM_MAP,
    TOKEN_GRID_SIZE,
    align_depth_to_sgm,
    load_da2_model,
    run_masked_sparse_da2,
    run_twopass_sparse_da2,
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

def evaluate(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, 'eval_twopass.log')),
            logging.StreamHandler(),
        ]
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )
    logging.info('=' * 66)
    logging.info('Two-Pass GAS vs Original GAS Comparison')
    logging.info('=' * 66)
    logging.info(f'  Device        : {device}')
    logging.info(f'  Threshold θ   : {args.threshold}')
    logging.info(f'  Prune layer   : {args.prune_layer}')
    logging.info('=' * 66)

    # ---- Load model ----
    model = load_da2_model(args.encoder, args.weights, device)
    embed_dim = EMBED_DIM_MAP[args.encoder]
    N = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE
    n_blocks = len(list(model.pretrained.blocks))

    router = SGMConfidenceTokenRouter(
        token_grid_size=TOKEN_GRID_SIZE,
        confidence_threshold=args.threshold,
        learnable_threshold=False,
    )

    # ---- Samples ----
    samples = build_sample_list(args.kitti_root)
    if not samples:
        logging.error('No samples found.'); return
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    logging.info(f'Samples: {len(samples)}')

    # ---- Collectors ----
    # Methods: dense, gas_sparse, twopass_sparse, fused_gas, fused_twopass
    methods = ['sgm', 'dense', 'gas_sparse', 'twopass_sparse',
               'fused_dense', 'fused_gas', 'fused_twopass']
    records = {m: {'all': [], 'kitti15': [], 'kitti12': []} for m in methods}
    records_sv = {m: {'all': [], 'kitti15': [], 'kitti12': []} for m in methods}
    csv_rows = []
    t_gas = 0.0
    t_twopass = 0.0

    for idx, s in enumerate(tqdm(samples, desc='Evaluating', unit='img')):
        left_bgr = cv2.imread(s['left'])
        if left_bgr is None:
            continue

        gt_disp, valid_gt = read_kitti_gt(s['gt_disp'])
        sgm_disp = read_pfm(s['sgm_pfm'])

        # Confidence maps
        conf_align, conf_route = load_protocol_confidence_maps(
            s,
            disparity_range=args.disparity_range,
            pkrn_min_dist=args.pkrn_min_dist,
            smooth_sigma=args.conf_sigma,
        )

        sgm_for_mask = sgm_disp
        if sgm_disp.shape != gt_disp.shape:
            sgm_for_mask = cv2.resize(sgm_disp, (gt_disp.shape[1], gt_disp.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        valid_sgm = valid_gt & (sgm_for_mask > 0)

        # ---- SGM ----
        sgm_m = compute_metrics(sgm_for_mask, gt_disp, valid_gt)
        sgm_m_sv = compute_metrics(sgm_for_mask, gt_disp, valid_sgm)

        # ---- Dense DA2 ----
        depth_dense = model.infer_image(left_bgr, input_size=518)
        disp_dense, _, _ = align_depth_to_sgm(
            depth_dense, sgm_disp, conf_align, conf_threshold=args.conf_threshold)
        disp_dense_eval = disp_dense
        if disp_dense.shape != gt_disp.shape:
            disp_dense_eval = cv2.resize(disp_dense, (gt_disp.shape[1], gt_disp.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
        dense_m = compute_metrics(disp_dense_eval, gt_disp, valid_gt)
        dense_m_sv = compute_metrics(disp_dense_eval, gt_disp, valid_sgm)

        # ---- Fused Dense ----
        conf_fuse = conf_route
        if conf_route.shape != gt_disp.shape:
            conf_fuse = cv2.resize(conf_route, (gt_disp.shape[1], gt_disp.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        disp_fused_dense = fuse_sgm_da2(sgm_for_mask, disp_dense_eval, conf_fuse,
                                         conf_threshold=args.conf_threshold)
        fused_dense_m = compute_metrics(disp_fused_dense, gt_disp, valid_gt)
        fused_dense_m_sv = compute_metrics(disp_fused_dense, gt_disp, valid_sgm)

        # ---- Token routing ----
        conf_tensor = torch.from_numpy(conf_route).unsqueeze(0).unsqueeze(0)
        dummy_tokens = torch.zeros(1, N, embed_dim)
        with torch.no_grad():
            routing = router(conf_tensor, dummy_tokens)
        prune_mask_1d = torch.zeros(N, dtype=torch.bool)
        prune_mask_1d[routing['prune_idx'][0]] = True
        n_keep = len(routing['keep_idx'][0])
        prune_ratio = float(routing['prune_ratio'])

        # ---- Original GAS sparse ----
        t0 = time.perf_counter()
        depth_gas = run_masked_sparse_da2(
            model, left_bgr, prune_mask_1d,
            input_size=518, prune_layer=args.prune_layer, do_reassembly=True)
        t_gas += time.perf_counter() - t0

        disp_gas, _, _ = align_depth_to_sgm(
            depth_gas, sgm_disp, conf_align, conf_threshold=args.conf_threshold)
        disp_gas_eval = disp_gas
        if disp_gas.shape != gt_disp.shape:
            disp_gas_eval = cv2.resize(disp_gas, (gt_disp.shape[1], gt_disp.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
        gas_m = compute_metrics(disp_gas_eval, gt_disp, valid_gt)
        gas_m_sv = compute_metrics(disp_gas_eval, gt_disp, valid_sgm)

        disp_fused_gas = fuse_sgm_da2(sgm_for_mask, disp_gas_eval, conf_fuse,
                                       conf_threshold=args.conf_threshold)
        fused_gas_m = compute_metrics(disp_fused_gas, gt_disp, valid_gt)
        fused_gas_m_sv = compute_metrics(disp_fused_gas, gt_disp, valid_sgm)

        # ---- Two-Pass GAS sparse ----
        t0 = time.perf_counter()
        depth_tp = run_twopass_sparse_da2(
            model, left_bgr, prune_mask_1d,
            input_size=518, prune_layer=args.prune_layer, do_reassembly=True)
        t_twopass += time.perf_counter() - t0

        disp_tp, _, _ = align_depth_to_sgm(
            depth_tp, sgm_disp, conf_align, conf_threshold=args.conf_threshold)
        disp_tp_eval = disp_tp
        if disp_tp.shape != gt_disp.shape:
            disp_tp_eval = cv2.resize(disp_tp, (gt_disp.shape[1], gt_disp.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
        tp_m = compute_metrics(disp_tp_eval, gt_disp, valid_gt)
        tp_m_sv = compute_metrics(disp_tp_eval, gt_disp, valid_sgm)

        disp_fused_tp = fuse_sgm_da2(sgm_for_mask, disp_tp_eval, conf_fuse,
                                      conf_threshold=args.conf_threshold)
        fused_tp_m = compute_metrics(disp_fused_tp, gt_disp, valid_gt)
        fused_tp_m_sv = compute_metrics(disp_fused_tp, gt_disp, valid_sgm)

        # ---- Accumulate ----
        all_metrics = {
            'sgm': (sgm_m, sgm_m_sv),
            'dense': (dense_m, dense_m_sv),
            'gas_sparse': (gas_m, gas_m_sv),
            'twopass_sparse': (tp_m, tp_m_sv),
            'fused_dense': (fused_dense_m, fused_dense_m_sv),
            'fused_gas': (fused_gas_m, fused_gas_m_sv),
            'fused_twopass': (fused_tp_m, fused_tp_m_sv),
        }
        for method, (m_all, m_sv) in all_metrics.items():
            for key in ('all', s['split']):
                records[method][key].append(m_all)
                records_sv[method][key].append(m_sv)

        csv_rows.append({
            'idx': idx, 'split': s['split'],
            'image': os.path.basename(s['left']),
            'prune_pct': f"{prune_ratio*100:.1f}",
            'dense_epe': f"{dense_m['epe']:.4f}",
            'dense_d1': f"{dense_m['d1']:.4f}",
            'gas_epe': f"{gas_m['epe']:.4f}",
            'gas_d1': f"{gas_m['d1']:.4f}",
            'twopass_epe': f"{tp_m['epe']:.4f}",
            'twopass_d1': f"{tp_m['d1']:.4f}",
            'fused_dense_epe': f"{fused_dense_m['epe']:.4f}",
            'fused_dense_d1': f"{fused_dense_m['d1']:.4f}",
            'fused_gas_epe': f"{fused_gas_m['epe']:.4f}",
            'fused_gas_d1': f"{fused_gas_m['d1']:.4f}",
            'fused_twopass_epe': f"{fused_tp_m['epe']:.4f}",
            'fused_twopass_d1': f"{fused_tp_m['d1']:.4f}",
        })

        if (idx + 1) % max(1, len(samples) // 10) == 0 or idx < 3:
            logging.info(
                f"[{idx+1}/{len(samples)}]  "
                f"GAS EPE={gas_m['epe']:.2f}  "
                f"TwoPass EPE={tp_m['epe']:.2f}  "
                f"Δ={gas_m['epe']-tp_m['epe']:+.2f}"
            )

    # ---- Aggregate ----
    n_samples = len(samples)
    attn_red = compute_attn_reduction(args.prune_layer, n_keep, N, n_blocks)

    splits = ['all', 'kitti15', 'kitti12']
    tag_map = {'all': 'COMBINED', 'kitti15': 'KITTI-2015', 'kitti12': 'KITTI-2012'}

    result_lines = []
    header = (
        f"\n{'='*76}\n"
        f"  Two-Pass GAS vs Original GAS — Pruning Comparison\n"
        f"  θ={args.threshold}  PL={args.prune_layer}  "
        f"Prune≈{prune_ratio*100:.1f}%  Attn↓≈{attn_red*100:.1f}%\n"
        f"{'='*76}"
    )
    result_lines.append(header)
    logging.info(header)

    for split in splits:
        n_imgs = len(records['sgm'][split])
        if n_imgs == 0:
            continue
        tag = tag_map[split]

        rows = [
            ('SGM',                 'sgm'),
            ('Dense DA2 + align',   'dense'),
            ('GAS Sparse DA2',      'gas_sparse'),
            ('TwoPass Sparse DA2',  'twopass_sparse'),
            ('Fused SGM+Dense',     'fused_dense'),
            ('Fused SGM+GAS',       'fused_gas'),
            ('Fused SGM+TwoPass',   'fused_twopass'),
        ]

        sep = '-' * 72
        block = f"\n  {tag}  ({n_imgs} images)\n"
        block += (f"  {'Method':<24s}  {'EPE(all)':>9s}  {'D1%(all)':>9s}"
                  f"  {'EPE(sv)':>9s}  {'D1%(sv)':>9s}\n")
        block += f"  {sep}\n"

        for label, method in rows:
            r = aggregate(records[method][split])
            r_sv = aggregate(records_sv[method][split])
            block += (f"  {label:<24s}"
                      f"  {r['epe']:>9.4f}  {r['d1']:>9.4f}"
                      f"  {r_sv['epe']:>9.4f}  {r_sv['d1']:>9.4f}\n")

        # Improvement summary
        gas_epe = aggregate(records['gas_sparse'][split])['epe']
        tp_epe = aggregate(records['twopass_sparse'][split])['epe']
        fgas_epe = aggregate(records['fused_gas'][split])['epe']
        ftp_epe = aggregate(records['fused_twopass'][split])['epe']
        block += f"  {sep}\n"
        block += f"  Sparse improvement: EPE {gas_epe:.4f} → {tp_epe:.4f}  (Δ={gas_epe-tp_epe:+.4f}, {(gas_epe-tp_epe)/gas_epe*100:+.1f}%)\n"
        block += f"  Fused  improvement: EPE {fgas_epe:.4f} → {ftp_epe:.4f}  (Δ={fgas_epe-ftp_epe:+.4f}, {(fgas_epe-ftp_epe)/fgas_epe*100:+.1f}%)\n"

        result_lines.append(block)
        logging.info(block)

    timing = (
        f"\n  Timing ({n_samples} images):\n"
        f"    GAS Sparse:     {t_gas:.1f}s  avg: {t_gas/max(1,n_samples)*1e3:.1f}ms/img\n"
        f"    TwoPass Sparse: {t_twopass:.1f}s  avg: {t_twopass/max(1,n_samples)*1e3:.1f}ms/img\n"
        f"    Overhead:       {(t_twopass-t_gas)/max(1,n_samples)*1e3:+.1f}ms/img\n"
    )
    result_lines.append(timing)
    logging.info(timing)

    # ---- Save ----
    txt_path = os.path.join(args.out_dir, 'twopass_results.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(result_lines))
    logging.info(f"Results: {txt_path}")

    csv_path = os.path.join(args.out_dir, 'twopass_per_image.csv')
    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logging.info(f"CSV: {csv_path}")

    logging.info('Done.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Two-Pass GAS vs Original GAS comparison.')
    p.add_argument('--kitti-root', default=DEFAULT_KITTI_ROOT)
    p.add_argument('--weights',
        default=DEFAULT_WEIGHTS)
    p.add_argument('--encoder', default=DEFAULT_ENCODER, choices=list(DA2_MODEL_CONFIGS.keys()))
    p.add_argument('--threshold', type=float, default=DEFAULT_PRUNE_THRESHOLD)
    p.add_argument('--prune-layer', type=int, default=DEFAULT_PRUNE_LAYER)
    p.add_argument('--conf-threshold', type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD)
    p.add_argument('--conf-sigma', type=float, default=DEFAULT_CONF_SIGMA)
    p.add_argument('--disparity-range', type=int, default=DEFAULT_DISPARITY_RANGE)
    p.add_argument('--pkrn-min-dist', type=int, default=DEFAULT_PKRN_MIN_DIST)
    p.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument('--out-dir',
        default=default_results_dir('eval_twopass'))
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
