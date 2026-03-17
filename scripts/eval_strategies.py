#!/usr/bin/env python3
"""
scripts/eval_strategies.py
===========================
Pruning strategy exploration for SGM-ViT on KITTI.

Evaluates 6 alternative token pruning strategies (plus the SGM baseline)
across multiple keep ratios to determine:

  1. Whether SGM confidence provides meaningful guidance (vs random baseline)
  2. Whether content-aware (attention-based) routing outperforms geometry-only
  3. Whether combining geometric + semantic signals yields better tradeoffs
  4. How fixed-ratio selection compares to threshold-based selection

Strategies
----------
  sgm_baseline  — current SGMConfidenceTokenRouter (threshold-based, reference)
  random        — uniform random selection (control)
  topk          — fixed-ratio: keep K tokens with lowest SGM confidence
  inverse_conf  — prune LOW-confidence tokens (hypothesis validation)
  checkerboard  — regular 2x2 spatial downsampling (~50% prune)
  cls_attn      — CLS attention scores from ViT warmup block
  hybrid        — alpha-weighted combination of SGM + CLS attention

Efficiency
----------
  - Dense DA2 baseline:  computed ONCE per image
  - SGM + alignment:     computed ONCE per image
  - CLS attention:       computed ONCE per warmup_block per image
  - Only GAS sparse forward + align + fuse is repeated per strategy x ratio

Usage
-----
  python scripts/eval_strategies.py                          # full sweep
  python scripts/eval_strategies.py --max-samples 20         # quick check
  python scripts/eval_strategies.py --strategies random,topk,cls_attn
  python scripts/eval_strategies.py --keep-ratios 0.6,0.7,0.8,0.9
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
    load_da2_model, run_masked_sparse_da2,
    align_depth_to_sgm, fuse_sgm_da2,
    DA2_MODEL_CONFIGS, TOKEN_GRID_SIZE, EMBED_DIM_MAP,
)
from scripts.eval_kitti import (
    build_sample_list, read_pfm, read_kitti_gt,
    load_pkrn_confidence, compute_metrics, aggregate,
)
from core.token_router import SGMConfidenceTokenRouter
from core.eval_utils import compute_attn_reduction, pareto_frontier, pool_confidence
from core.pruning_strategies import (
    random_prune_mask, topk_confidence_mask, inverse_confidence_mask,
    spatial_checkerboard_mask, cls_attention_mask, hybrid_mask,
)


# ---------------------------------------------------------------------------
# All strategy names
# ---------------------------------------------------------------------------
ALL_STRATEGIES = [
    'sgm_baseline', 'random', 'topk', 'inverse_conf',
    'checkerboard', 'cls_attn', 'hybrid',
]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_strategies(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, 'eval_strategies.log')),
            logging.StreamHandler(),
        ]
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )

    strategies   = [s.strip() for s in args.strategies.split(',')]
    keep_ratios  = [float(x) for x in args.keep_ratios.split(',')]
    hybrid_alphas = [float(x) for x in args.hybrid_alphas.split(',')]

    for s in strategies:
        if s not in ALL_STRATEGIES:
            logging.error(f"Unknown strategy: '{s}'. Valid: {ALL_STRATEGIES}")
            return

    logging.info('=' * 70)
    logging.info('SGM-ViT  Pruning Strategy Exploration')
    logging.info('=' * 70)
    logging.info(f'  KITTI root      : {args.kitti_root}')
    logging.info(f'  Encoder         : {args.encoder}')
    logging.info(f'  Strategies      : {strategies}')
    logging.info(f'  Keep ratios     : {keep_ratios}')
    logging.info(f'  Hybrid alphas   : {hybrid_alphas}')
    logging.info(f'  Warmup block    : {args.warmup_block}')
    logging.info(f'  Prune layer     : {args.prune_layer}')
    logging.info(f'  SGM threshold   : {args.threshold}')
    logging.info(f'  Re-assembly     : {"on" if not args.no_reassembly else "off"}')
    logging.info(f'  Device          : {device}')
    logging.info(f'  Output          : {args.out_dir}')
    logging.info('=' * 70)

    # ---- Load DA2 model ------------------------------------------------
    logging.info(f'Loading DepthAnythingV2 ({args.encoder}) ...')
    model = load_da2_model(args.encoder, args.weights, device)

    embed_dim = EMBED_DIM_MAP[args.encoder]
    N         = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE

    n_blocks = len(list(model.pretrained.blocks))

    # ---- Sample list ---------------------------------------------------
    samples = build_sample_list(args.kitti_root)
    if not samples:
        logging.error('No samples found -- check --kitti-root path.')
        return
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        logging.info(f'Limiting to {args.max_samples} samples (--max-samples)')
    logging.info(f'Total samples: {len(samples)}\n')

    # ---- Accumulators --------------------------------------------------
    # Key format: "strategy:keep_ratio" or "strategy:keep_ratio:alpha" for hybrid
    # Each holds {'all': [], 'kitti15': [], 'kitti12': []} of metric dicts
    sparse_records:    dict[str, dict[str, list]] = {}
    sparse_records_sv: dict[str, dict[str, list]] = {}
    fused_records:     dict[str, dict[str, list]] = {}
    fused_records_sv:  dict[str, dict[str, list]] = {}
    prune_pct_records: dict[str, list[float]] = {}

    def _ensure_key(k: str):
        if k not in sparse_records:
            sparse_records[k]    = {'all': [], 'kitti15': [], 'kitti12': []}
            sparse_records_sv[k] = {'all': [], 'kitti15': [], 'kitti12': []}
            fused_records[k]     = {'all': [], 'kitti15': [], 'kitti12': []}
            fused_records_sv[k]  = {'all': [], 'kitti15': [], 'kitti12': []}
            prune_pct_records[k] = []

    # Baselines (computed ONCE per image, invariant to strategy)
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
        mismatch  = np.load(s['mismatch']).astype(bool)
        occlusion = np.load(s['occlusion']).astype(bool)
        conf_align = (~(mismatch | occlusion)).astype(np.float32)
        conf_align = gaussian_filter(conf_align, sigma=args.conf_sigma)

        # Routing confidence (PKRN)
        if os.path.exists(s.get('pkrn_cache', '')) or os.path.exists(s.get('right', '')):
            conf_route = load_pkrn_confidence(
                s, args.disparity_range, args.pkrn_min_dist, args.conf_sigma)
        else:
            conf_route = conf_align

        # SGM-valid mask
        sgm_for_mask = sgm_disp
        if sgm_disp.shape != gt_disp.shape:
            sgm_for_mask = cv2.resize(sgm_disp,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        valid_sgm = valid_gt & (sgm_for_mask > 0)

        # Confidence for fusion (at gt resolution)
        conf_for_fuse = conf_align
        if conf_align.shape != gt_disp.shape:
            conf_for_fuse = cv2.resize(conf_align,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_LINEAR)

        # ---- SGM metrics (ONCE) ----------------------------------------
        sgm_metrics    = compute_metrics(sgm_for_mask, gt_disp, valid_gt)
        sgm_metrics_sv = compute_metrics(sgm_for_mask, gt_disp, valid_sgm)
        for k in ('all', s['split']):
            sgm_records[k].append(sgm_metrics)
            sgm_records_sv[k].append(sgm_metrics_sv)

        # ---- Dense DA2 baseline (ONCE) ---------------------------------
        depth_dense = model.infer_image(left_bgr, input_size=518)

        disp_dense, _, _ = align_depth_to_sgm(
            depth_mono=depth_dense, disparity_raw=sgm_disp,
            confidence_map=conf_align, conf_threshold=args.conf_threshold,
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

        # ---- Fused SGM + Dense DA2 baseline (ONCE) ---------------------
        disp_fused_dense = fuse_sgm_da2(
            sgm_disp=sgm_for_mask, da2_aligned=disp_dense_eval,
            confidence_map=conf_for_fuse, conf_threshold=args.conf_threshold,
        )
        fused_d_metrics    = compute_metrics(disp_fused_dense, gt_disp, valid_gt)
        fused_d_metrics_sv = compute_metrics(disp_fused_dense, gt_disp, valid_sgm)
        for k in ('all', s['split']):
            fused_dense_records[k].append(fused_d_metrics)
            fused_dense_records_sv[k].append(fused_d_metrics_sv)

        # ---- CLS attention extraction (ONCE per image, if needed) ------
        cls_scores_cache = None
        needs_cls = any(st in strategies for st in ('cls_attn', 'hybrid'))
        if needs_cls:
            _, cls_scores_cache = cls_attention_mask(
                model, left_bgr, keep_ratio=0.5,  # ratio unused for score extraction
                warmup_block=args.warmup_block, input_size=518,
            )
            # cls_scores_cache is (N_actual,) float CPU tensor

        # ---- Per-strategy evaluation -----------------------------------
        def _run_sparse_eval(prune_mask_1d: torch.Tensor, strat_key: str):
            """Run GAS sparse DA2, align, fuse, compute metrics, record."""
            _ensure_key(strat_key)

            n_prune = int(prune_mask_1d.sum().item())
            n_total = prune_mask_1d.shape[0]
            prune_pct = n_prune / n_total * 100.0

            depth_sparse = run_masked_sparse_da2(
                model=model, image_bgr=left_bgr,
                prune_mask=prune_mask_1d, input_size=518,
                prune_layer=args.prune_layer,
                do_reassembly=not args.no_reassembly,
            )

            disp_sparse, _, _ = align_depth_to_sgm(
                depth_mono=depth_sparse, disparity_raw=sgm_disp,
                confidence_map=conf_align, conf_threshold=args.conf_threshold,
            )
            disp_sparse_eval = disp_sparse
            if disp_sparse.shape != gt_disp.shape:
                disp_sparse_eval = cv2.resize(disp_sparse,
                    (gt_disp.shape[1], gt_disp.shape[0]),
                    interpolation=cv2.INTER_LINEAR)

            sp_metrics    = compute_metrics(disp_sparse_eval, gt_disp, valid_gt)
            sp_metrics_sv = compute_metrics(disp_sparse_eval, gt_disp, valid_sgm)

            # Fused SGM + Sparse DA2
            disp_fused_sp = fuse_sgm_da2(
                sgm_disp=sgm_for_mask, da2_aligned=disp_sparse_eval,
                confidence_map=conf_for_fuse, conf_threshold=args.conf_threshold,
            )
            fu_metrics    = compute_metrics(disp_fused_sp, gt_disp, valid_gt)
            fu_metrics_sv = compute_metrics(disp_fused_sp, gt_disp, valid_sgm)

            # Record
            for k in ('all', s['split']):
                sparse_records[strat_key][k].append(sp_metrics)
                sparse_records_sv[strat_key][k].append(sp_metrics_sv)
                fused_records[strat_key][k].append(fu_metrics)
                fused_records_sv[strat_key][k].append(fu_metrics_sv)
            prune_pct_records[strat_key].append(prune_pct)

            # Attention FLOPs reduction
            n_keep = n_total - n_prune
            attn_red = compute_attn_reduction(
                args.prune_layer, n_keep, n_total, n_blocks)

            return {
                'prune_pct': prune_pct,
                'attn_reduction': attn_red * 100.0,
                'sparse_epe': sp_metrics['epe'],
                'sparse_d1': sp_metrics['d1'],
                'fused_epe': fu_metrics['epe'],
                'fused_d1': fu_metrics['d1'],
            }

        # --- SGM baseline (threshold-based, reference) ---
        if 'sgm_baseline' in strategies:
            router = SGMConfidenceTokenRouter(
                token_grid_size=TOKEN_GRID_SIZE,
                confidence_threshold=args.threshold,
                learnable_threshold=False,
            )
            conf_tensor = torch.from_numpy(conf_route).unsqueeze(0).unsqueeze(0)
            dummy_tokens = torch.zeros(1, N, embed_dim)
            with torch.no_grad():
                routing = router(conf_tensor, dummy_tokens)
            prune_mask_1d = torch.zeros(N, dtype=torch.bool)
            prune_mask_1d[routing['prune_idx'][0]] = True

            sgm_keep_ratio = 1.0 - float(routing['prune_ratio'])
            strat_key = f"sgm_baseline:{sgm_keep_ratio:.3f}"
            res = _run_sparse_eval(prune_mask_1d, strat_key)

            csv_rows.append({
                'idx': idx, 'split': s['split'],
                'image': os.path.basename(s['left']),
                'strategy': 'sgm_baseline',
                'keep_ratio': f"{sgm_keep_ratio:.3f}",
                'prune_pct': f"{res['prune_pct']:.1f}",
                'attn_reduction': f"{res['attn_reduction']:.1f}",
                'sparse_epe': f"{res['sparse_epe']:.4f}",
                'sparse_d1': f"{res['sparse_d1']:.4f}",
                'fused_epe': f"{res['fused_epe']:.4f}",
                'fused_d1': f"{res['fused_d1']:.4f}",
            })

        # --- Sweep keep_ratios for fixed-ratio strategies ---
        for kr in keep_ratios:

            # Random
            if 'random' in strategies:
                mask = random_prune_mask(N, kr, seed=idx)  # per-image seed for reproducibility
                strat_key = f"random:{kr:.2f}"
                res = _run_sparse_eval(mask, strat_key)
                csv_rows.append({
                    'idx': idx, 'split': s['split'],
                    'image': os.path.basename(s['left']),
                    'strategy': 'random',
                    'keep_ratio': f"{kr:.2f}",
                    'prune_pct': f"{res['prune_pct']:.1f}",
                    'attn_reduction': f"{res['attn_reduction']:.1f}",
                    'sparse_epe': f"{res['sparse_epe']:.4f}",
                    'sparse_d1': f"{res['sparse_d1']:.4f}",
                    'fused_epe': f"{res['fused_epe']:.4f}",
                    'fused_d1': f"{res['fused_d1']:.4f}",
                })

            # Top-K confidence
            if 'topk' in strategies:
                mask = topk_confidence_mask(conf_route, TOKEN_GRID_SIZE, kr)
                strat_key = f"topk:{kr:.2f}"
                res = _run_sparse_eval(mask, strat_key)
                csv_rows.append({
                    'idx': idx, 'split': s['split'],
                    'image': os.path.basename(s['left']),
                    'strategy': 'topk',
                    'keep_ratio': f"{kr:.2f}",
                    'prune_pct': f"{res['prune_pct']:.1f}",
                    'attn_reduction': f"{res['attn_reduction']:.1f}",
                    'sparse_epe': f"{res['sparse_epe']:.4f}",
                    'sparse_d1': f"{res['sparse_d1']:.4f}",
                    'fused_epe': f"{res['fused_epe']:.4f}",
                    'fused_d1': f"{res['fused_d1']:.4f}",
                })

            # Inverse confidence
            if 'inverse_conf' in strategies:
                # To match the keep_ratio, compute threshold that yields ~(1-kr) pruning
                # Use topk-style selection: prune K tokens with LOWEST confidence
                n_prune = N - max(1, int(round(N * kr)))
                conf_grid_flat = torch.from_numpy(
                    pool_confidence(conf_route, TOKEN_GRID_SIZE).reshape(-1))
                if n_prune > 0 and n_prune < N:
                    # Sort ascending, prune the first n_prune (lowest confidence)
                    _, sorted_idx = conf_grid_flat.sort()
                    mask = torch.zeros(N, dtype=torch.bool)
                    mask[sorted_idx[:n_prune]] = True
                elif n_prune >= N:
                    mask = torch.ones(N, dtype=torch.bool)
                else:
                    mask = torch.zeros(N, dtype=torch.bool)

                strat_key = f"inverse_conf:{kr:.2f}"
                res = _run_sparse_eval(mask, strat_key)
                csv_rows.append({
                    'idx': idx, 'split': s['split'],
                    'image': os.path.basename(s['left']),
                    'strategy': 'inverse_conf',
                    'keep_ratio': f"{kr:.2f}",
                    'prune_pct': f"{res['prune_pct']:.1f}",
                    'attn_reduction': f"{res['attn_reduction']:.1f}",
                    'sparse_epe': f"{res['sparse_epe']:.4f}",
                    'sparse_d1': f"{res['sparse_d1']:.4f}",
                    'fused_epe': f"{res['fused_epe']:.4f}",
                    'fused_d1': f"{res['fused_d1']:.4f}",
                })

            # CLS attention
            if 'cls_attn' in strategies and cls_scores_cache is not None:
                N_actual = cls_scores_cache.shape[0]
                n_keep = max(1, int(round(N_actual * kr)))
                _, top_idx = cls_scores_cache.topk(n_keep, largest=True)
                mask = torch.ones(N_actual, dtype=torch.bool)
                mask[top_idx] = False

                strat_key = f"cls_attn:{kr:.2f}"
                res = _run_sparse_eval(mask, strat_key)
                csv_rows.append({
                    'idx': idx, 'split': s['split'],
                    'image': os.path.basename(s['left']),
                    'strategy': 'cls_attn',
                    'keep_ratio': f"{kr:.2f}",
                    'prune_pct': f"{res['prune_pct']:.1f}",
                    'attn_reduction': f"{res['attn_reduction']:.1f}",
                    'sparse_epe': f"{res['sparse_epe']:.4f}",
                    'sparse_d1': f"{res['sparse_d1']:.4f}",
                    'fused_epe': f"{res['fused_epe']:.4f}",
                    'fused_d1': f"{res['fused_d1']:.4f}",
                })

            # Hybrid (sweep alphas at this keep_ratio)
            if 'hybrid' in strategies and cls_scores_cache is not None:
                for alpha in hybrid_alphas:
                    mask = hybrid_mask(
                        conf_route, cls_scores_cache,
                        TOKEN_GRID_SIZE, kr, alpha=alpha,
                    )
                    strat_key = f"hybrid:{kr:.2f}:a{alpha:.2f}"
                    res = _run_sparse_eval(mask, strat_key)
                    csv_rows.append({
                        'idx': idx, 'split': s['split'],
                        'image': os.path.basename(s['left']),
                        'strategy': f"hybrid_a{alpha:.2f}",
                        'keep_ratio': f"{kr:.2f}",
                        'prune_pct': f"{res['prune_pct']:.1f}",
                        'attn_reduction': f"{res['attn_reduction']:.1f}",
                        'sparse_epe': f"{res['sparse_epe']:.4f}",
                        'sparse_d1': f"{res['sparse_d1']:.4f}",
                        'fused_epe': f"{res['fused_epe']:.4f}",
                        'fused_d1': f"{res['fused_d1']:.4f}",
                    })

        # Checkerboard (fixed ~50% pruning, not ratio-dependent)
        if 'checkerboard' in strategies:
            # Use the actual token grid size from a 518-px image
            image_tensor, _ = model.image2tensor(left_bgr, 518)
            ph = image_tensor.shape[-2] // 14
            pw = image_tensor.shape[-1] // 14
            mask = spatial_checkerboard_mask(ph, pw)

            strat_key = "checkerboard:0.50"
            res = _run_sparse_eval(mask, strat_key)
            csv_rows.append({
                'idx': idx, 'split': s['split'],
                'image': os.path.basename(s['left']),
                'strategy': 'checkerboard',
                'keep_ratio': f"{1.0 - res['prune_pct']/100:.2f}",
                'prune_pct': f"{res['prune_pct']:.1f}",
                'attn_reduction': f"{res['attn_reduction']:.1f}",
                'sparse_epe': f"{res['sparse_epe']:.4f}",
                'sparse_d1': f"{res['sparse_d1']:.4f}",
                'fused_epe': f"{res['fused_epe']:.4f}",
                'fused_d1': f"{res['fused_d1']:.4f}",
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

    def _fmt_summary_table(split: str) -> str:
        a_sgm    = aggregate(sgm_records[split])
        a_dense  = aggregate(dense_records[split])
        a_fused_d = aggregate(fused_dense_records[split])
        if a_sgm['n'] == 0:
            return ''

        tag = tag_map[split]
        sep = '-' * 95
        lines = [
            f"\n  {tag}  ({a_sgm['n']} images)",
            f"  {'Strategy':<28s}  {'KeepR':>6s}  {'Prune%':>6s}  {'Attn↓%':>6s}"
            f"  {'Sp EPE':>8s}  {'Sp D1%':>8s}  {'Fu EPE':>8s}  {'Fu D1%':>8s}",
            f"  {sep}",
            # Baselines
            f"  {'SGM (pre-computed)':<28s}  {'':>6s}  {'':>6s}  {'':>6s}"
            f"  {a_sgm['epe']:>8.4f}  {a_sgm['d1']:>8.4f}  {'':>8s}  {'':>8s}",
            f"  {'Dense DA2 + align':<28s}  {'1.00':>6s}  {'0.0':>6s}  {'0.0':>6s}"
            f"  {a_dense['epe']:>8.4f}  {a_dense['d1']:>8.4f}"
            f"  {a_fused_d['epe']:>8.4f}  {a_fused_d['d1']:>8.4f}",
            f"  {sep}",
        ]

        # Collect all strategy keys, sorted by strategy name then keep_ratio
        sorted_keys = sorted(sparse_records.keys())

        for sk in sorted_keys:
            a_sp = aggregate(sparse_records[sk][split])
            a_fu = aggregate(fused_records[sk][split])
            if a_sp['n'] == 0:
                continue

            pr_list = prune_pct_records[sk]
            avg_prune = np.mean(pr_list) if pr_list else 0.0
            avg_keep = 1.0 - avg_prune / 100.0
            n_keep_avg = N * avg_keep
            attn_red = compute_attn_reduction(
                args.prune_layer, n_keep_avg, N, n_blocks) * 100

            # Parse strategy name from key
            parts = sk.split(':')
            strat_name = parts[0]
            kr_str = parts[1] if len(parts) > 1 else ''
            if len(parts) > 2:
                strat_name = f"{strat_name} ({parts[2]})"

            label = f"{strat_name} kr={kr_str}"
            lines.append(
                f"  {label:<28s}"
                f"  {avg_keep:>5.2f}%  {avg_prune:>5.1f}%  {attn_red:>5.1f}%"
                f"  {a_sp['epe']:>8.4f}  {a_sp['d1']:>8.4f}"
                f"  {a_fu['epe']:>8.4f}  {a_fu['d1']:>8.4f}"
            )

        lines.append('')
        return '\n'.join(lines)

    # ---- Build Pareto data ----
    def _pareto_table(split: str = 'all') -> str:
        points = []
        for sk in sparse_records.keys():
            a_fu = aggregate(fused_records[sk][split])
            a_sp = aggregate(sparse_records[sk][split])
            if a_fu['n'] == 0:
                continue
            pr_list = prune_pct_records[sk]
            avg_prune = np.mean(pr_list) if pr_list else 0.0
            n_keep_avg = N * (1.0 - avg_prune / 100.0)
            attn_red = compute_attn_reduction(
                args.prune_layer, n_keep_avg, N, n_blocks)
            points.append({
                'key': sk,
                'attn_reduction': attn_red,
                'sparse_epe': a_sp['epe'],
                'fused_epe': a_fu['epe'],
                'fused_d1': a_fu['d1'],
                'prune_pct': avg_prune,
            })

        frontier = pareto_frontier(points)
        if not frontier:
            return ''

        lines = [
            f"\n  Pareto-optimal operating points ({tag_map.get(split, split)}):",
            f"  {'Strategy Key':<36s}  {'Prune%':>6s}  {'Attn↓%':>6s}"
            f"  {'Sp EPE':>8s}  {'Fu EPE':>8s}  {'Fu D1%':>8s}",
            f"  {'-'*80}",
        ]
        for p in frontier:
            lines.append(
                f"  {p['key']:<36s}"
                f"  {p['prune_pct']:>5.1f}%"
                f"  {p['attn_reduction']*100:>5.1f}%"
                f"  {p['sparse_epe']:>8.4f}  {p['fused_epe']:>8.4f}"
                f"  {p['fused_d1']:>8.4f}"
            )
        lines.append('')
        return '\n'.join(lines)

    # ---- Emit report ----
    header = f"\n{'='*70}\n  SGM-ViT Pruning Strategy Exploration Results\n{'='*70}"
    logging.info(header)

    result_lines = [header]
    for split in ['all', 'kitti15', 'kitti12']:
        tbl = _fmt_summary_table(split)
        if tbl:
            logging.info(tbl)
            result_lines.append(tbl)

    pareto = _pareto_table('all')
    if pareto:
        logging.info(pareto)
        result_lines.append(pareto)

    info = f"\n  Samples evaluated: {n_samples}\n"
    logging.info(info)
    result_lines.append(info)

    # ---- Save results --------------------------------------------------
    txt_path = os.path.join(args.out_dir, 'strategies_summary.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(result_lines))
    logging.info(f"Summary: {txt_path}")

    csv_path = os.path.join(args.out_dir, 'strategies_sweep.csv')
    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logging.info(f"Per-image CSV: {csv_path}")

    logging.info('=' * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='SGM-ViT pruning strategy exploration on KITTI.'
    )
    p.add_argument('--kitti-root', default='/nfs/usrhome/pdongaa/dataczy/kitti',
        help='KITTI dataset root.')
    p.add_argument('--weights',
        default=os.path.join(_PROJECT_DIR, 'Depth-Anything-V2', 'checkpoints',
                             'depth_anything_v2_vits.pth'),
        help='Path to DepthAnythingV2 checkpoint.')
    p.add_argument('--encoder', default='vits',
        choices=list(DA2_MODEL_CONFIGS.keys()))
    p.add_argument('--strategies',
        default=','.join(ALL_STRATEGIES),
        help=f'Comma-separated strategies (default: all). Options: {ALL_STRATEGIES}')
    p.add_argument('--keep-ratios', default='0.5,0.6,0.7,0.8,0.9',
        help='Comma-separated keep ratios for fixed-ratio strategies.')
    p.add_argument('--hybrid-alphas', default='0.0,0.25,0.5,0.75,1.0',
        help='Comma-separated alpha values for hybrid strategy.')
    p.add_argument('--warmup-block', type=int, default=0,
        help='ViT block from which to extract CLS attention (default: 0).')
    p.add_argument('--prune-layer', type=int, default=0,
        help='ViT block at which GAS pruning starts (0-based).')
    p.add_argument('--threshold', type=float, default=0.65,
        help='SGM confidence threshold for sgm_baseline strategy.')
    p.add_argument('--no-reassembly', action='store_true',
        help='Disable token re-assembly before the DPT decoder.')
    p.add_argument('--conf-threshold', type=float, default=0.7,
        help='Confidence threshold for depth alignment and fusion.')
    p.add_argument('--conf-sigma', type=float, default=5.0,
        help='Gaussian sigma for PKRN confidence smoothing.')
    p.add_argument('--disparity-range', type=int, default=128,
        help='SGM disparity search range (for PKRN cache miss).')
    p.add_argument('--pkrn-min-dist', type=int, default=1,
        help='Minimum disparity gap for PKRN second-best search.')
    p.add_argument('--max-samples', type=int, default=0,
        help='Evaluate only the first N samples (0 = all).')
    p.add_argument('--out-dir',
        default=os.path.join(_PROJECT_DIR, 'results', 'eval_strategies'),
        help='Directory to save results and log.')
    p.add_argument('--cpu', action='store_true',
        help='Force CPU inference.')
    return p.parse_args()


if __name__ == '__main__':
    evaluate_strategies(parse_args())
