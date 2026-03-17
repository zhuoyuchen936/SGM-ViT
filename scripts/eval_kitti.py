#!/usr/bin/env python3
"""
scripts/eval_kitti.py
=====================
KITTI disparity accuracy evaluation for SGM-ViT.

Evaluates three predictors against KITTI ground-truth (annotated) disparity
on the KITTI 2012 + KITTI 2015 combined training sets:

  1. SGM (pre-computed)    — raw stereo baseline from sgm_hole/*.pfm
  2. Dense DA2 + align     — DepthAnythingV2 monocular depth, per-image
                             least-squares aligned to SGM disparity space
  3. Sparse DA2 + align    — SGM-ViT: token-pruned + re-assembled depth,
                             aligned to SGM disparity space

Alignment for (2) and (3):
    argmin_{s,t}  Σ_i || s·d_i + t − disp_sgm_i ||²
at pixels where SGM confidence >= conf_threshold (derived from the
precomputed mismatch + occlusion masks stored in sgm_hole/).
GT disparity is NOT used for alignment — only as the evaluation target.

Metrics (KITTI standard protocol):
  EPE  — mean absolute disparity error at valid GT pixels
  D1   — % of valid pixels where |pred − gt| > max(3.0, 0.05·gt)

Dataset paths (identical to rSGM_Mamba/core/stereo_datasets.py):
  KITTI 2015 root {R}:
    {R}/training/image_2/*_10.png            left images
    {R}/training/disp_occ_0/*_10.png         GT disparity  (16-bit / 256)
    {R}/training/sgm_hole/*_10.pfm           pre-computed SGM disparity
    {R}/training/sgm_hole/*_10_mismatches.npy
    {R}/training/sgm_hole/*_10_occlusion.npy

  KITTI 2012 root {R}/kitti2012:
    .../training/colored_0/*_10.png
    .../training/disp_occ/*_10.png
    .../training/sgm_hole/*_10.pfm / *_mismatches.npy / *_occlusion.npy

Usage
-----
  python scripts/eval_kitti.py
  python scripts/eval_kitti.py --encoder vitb --threshold 0.5
  python scripts/eval_kitti.py --prune-layer 6
  python scripts/eval_kitti.py --max-samples 20   # quick sanity check
  python scripts/eval_kitti.py --no-sparse        # skip sparse DA2
  python scripts/eval_kitti.py --kitti-root /your/path/to/kitti
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from glob import glob

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch

# ---------------------------------------------------------------------------
# Path setup — importable from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_DA2_DIR     = os.path.join(_PROJECT_DIR, "Depth-Anything-V2")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _DA2_DIR)

from core.token_router import SGMConfidenceTokenRouter
from core.sgm_wrapper  import run_sgm_with_confidence
from demo import (
    load_da2_model, run_sparse_da2, run_masked_sparse_da2,
    align_depth_to_sgm, fuse_sgm_da2,
    fuse_dispatch, FUSION_STRATEGIES,
    DA2_MODEL_CONFIGS, TOKEN_GRID_SIZE, EMBED_DIM_MAP,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_pfm(path: str) -> np.ndarray:
    """Read a .pfm disparity file → (H, W) float32."""
    with open(path, 'rb') as f:
        header = f.readline().rstrip()
        assert header in (b'PF', b'Pf'), f"Not a PFM file: {path}"
        w, h  = map(int, f.readline().split())
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        data  = np.fromfile(f, endian + 'f').reshape(h, w)
    return np.ascontiguousarray(np.flipud(data)).astype(np.float32)


def read_kitti_gt(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read KITTI ground-truth disparity (16-bit PNG, value / 256 = disparity).
    Returns (disp_gt, valid_mask).
    """
    raw   = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    disp  = raw.astype(np.float32) / 256.0
    valid = disp > 0.0
    return disp, valid


# ---------------------------------------------------------------------------
# Sample list builder
# ---------------------------------------------------------------------------

def build_sample_list(kitti_root: str) -> list[dict]:
    """
    Build sorted list of sample dicts for KITTI 2015 + KITTI 2012.

    Each dict has keys:
        left      : path to left colour image
        right     : path to right colour image (for live PKRN computation)
        gt_disp   : path to GT disparity PNG
        sgm_pfm   : path to pre-computed SGM disparity PFM
        mismatch  : path to mismatch mask .npy
        occlusion : path to occlusion mask .npy
        pkrn_cache: path where PKRN confidence will be cached (*_pkrn.npy)
        split     : 'kitti15' or 'kitti12'
    """
    def sg(pat): return sorted(glob(pat))

    root_15 = kitti_root
    root_12 = os.path.join(kitti_root, 'kitti2012')

    samples: list[dict] = []

    # --- KITTI 2015 ---
    lefts  = sg(os.path.join(root_15, 'training', 'image_2',    '*_10.png'))
    rights = sg(os.path.join(root_15, 'training', 'image_3',    '*_10.png'))
    gts    = sg(os.path.join(root_15, 'training', 'disp_occ_0', '*_10.png'))
    sgms   = sg(os.path.join(root_15, 'training', 'sgm_hole',   '*_10.pfm'))
    mis    = sg(os.path.join(root_15, 'training', 'sgm_hole',   '*_10_mismatches.npy'))
    occ    = sg(os.path.join(root_15, 'training', 'sgm_hole',   '*_10_occlusion.npy'))
    for l, r, g, s, m, o in zip(lefts, rights, gts, sgms, mis, occ):
        pkrn = s.replace('.pfm', '_pkrn.npy')
        samples.append(dict(left=l, right=r, gt_disp=g, sgm_pfm=s,
                            mismatch=m, occlusion=o, pkrn_cache=pkrn,
                            split='kitti15'))
    logging.info(f"KITTI 2015: {len(lefts)} samples found")

    # --- KITTI 2012 ---
    lefts  = sg(os.path.join(root_12, 'training', 'colored_0', '*_10.png'))
    rights = sg(os.path.join(root_12, 'training', 'colored_1', '*_10.png'))
    gts    = sg(os.path.join(root_12, 'training', 'disp_occ',  '*_10.png'))
    sgms   = sg(os.path.join(root_12, 'training', 'sgm_hole',  '*_10.pfm'))
    mis    = sg(os.path.join(root_12, 'training', 'sgm_hole',  '*_10_mismatches.npy'))
    occ    = sg(os.path.join(root_12, 'training', 'sgm_hole',  '*_10_occlusion.npy'))
    for l, r, g, s, m, o in zip(lefts, rights, gts, sgms, mis, occ):
        pkrn = s.replace('.pfm', '_pkrn.npy')
        samples.append(dict(left=l, right=r, gt_disp=g, sgm_pfm=s,
                            mismatch=m, occlusion=o, pkrn_cache=pkrn,
                            split='kitti12'))
    logging.info(f"KITTI 2012: {len(lefts)} samples found")

    return samples


def load_pkrn_confidence(
    s: dict,
    disparity_range: int,
    pkrn_min_dist: int,
    smooth_sigma: float,
) -> np.ndarray:
    """
    Return the PKRN confidence map for sample ``s``.

    Strategy (lazy cache):
    1. If ``s['pkrn_cache']`` exists on disk → load and return it.
    2. Otherwise run ``run_sgm_with_confidence()`` on the stereo pair,
       save the raw (unsmoothed) PKRN map to ``s['pkrn_cache']``, then
       return the Gaussian-smoothed version for this call.

    The cache stores the **unsmoothed** PKRN map so that ``smooth_sigma``
    can be changed without invalidating the cache.
    """
    cache_path = s['pkrn_cache']

    if os.path.exists(cache_path):
        pkrn_raw = np.load(cache_path)                         # (H, W) float32
    else:
        # Run SGM live with smooth_sigma=0 so conf_map == raw PKRN × ~hole_mask
        _, pkrn_raw, _ = run_sgm_with_confidence(
            left_path       = s['left'],
            right_path      = s['right'],
            disparity_range = disparity_range,
            smooth_sigma    = 0.0,
            pkrn_min_dist   = pkrn_min_dist,
            verbose         = False,
        )
        np.save(cache_path, pkrn_raw)
        logging.info(f"Cached PKRN: {cache_path}")

    # Apply Gaussian smoothing (caller's sigma)
    if smooth_sigma > 0:
        return gaussian_filter(pkrn_raw, sigma=smooth_sigma).clip(0.0, 1.0)
    return pkrn_raw


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_gt: np.ndarray,
    max_disp: float = 192.0,
) -> dict:
    """
    EPE and D1 at valid GT pixels with gt < max_disp.

    D1 criterion (KITTI standard):
        |pred − gt| > max(3.0, 0.05·gt)
    """
    mask = valid_gt & (gt < max_disp) & np.isfinite(pred) & (pred >= 0)
    n    = int(mask.sum())
    if n == 0:
        return {'epe': float('nan'), 'd1': float('nan'), 'n': 0}
    err  = np.abs(pred[mask] - gt[mask])
    epe  = float(err.mean())
    d1   = float((err > np.maximum(3.0, 0.05 * gt[mask])).mean() * 100.0)
    return {'epe': epe, 'd1': d1, 'n': n}


def aggregate(records: list[dict]) -> dict:
    """Image-level macro mean for EPE and D1.

    Returns
    -------
    epe      : float  — macro mean EPE across images
    d1       : float  — macro mean D1 across images
    n        : int    — number of images with valid metrics
    pixel_n  : int    — total valid pixels summed across all images
    """
    epes    = [r['epe'] for r in records if not np.isnan(r['epe'])]
    d1s     = [r['d1']  for r in records if not np.isnan(r['d1'])]
    px_ns   = [r['n']   for r in records if not np.isnan(r['epe'])]
    return {
        'epe':     float(np.mean(epes)) if epes else float('nan'),
        'd1':      float(np.mean(d1s))  if d1s  else float('nan'),
        'n':       len(epes),
        'pixel_n': int(sum(px_ns)),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, 'eval_kitti.log')),
            logging.StreamHandler(),
        ]
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )
    logging.info('=' * 66)
    logging.info('SGM-ViT  KITTI Disparity Evaluation')
    logging.info('=' * 66)
    logging.info(f'  KITTI root    : {args.kitti_root}')
    logging.info(f'  Encoder       : {args.encoder}')
    logging.info(f'  Prune layer   : {args.prune_layer}')
    logging.info(f'  Sparse mode   : {args.sparse_mode}')
    logging.info(f'  Threshold θ   : {args.threshold}')
    logging.info(f'  Re-assembly   : {"on" if not args.no_reassembly else "off"}')
    logging.info(f'  Device        : {device}')
    logging.info(f'  Output        : {args.out_dir}')
    logging.info('=' * 66)

    # ---- Load DA2 model ------------------------------------------------
    logging.info(f'Loading DepthAnythingV2 ({args.encoder}) ...')
    model = load_da2_model(args.encoder, args.weights, device)

    # ---- Token router --------------------------------------------------
    embed_dim = EMBED_DIM_MAP[args.encoder]
    N         = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE
    router    = SGMConfidenceTokenRouter(
        token_grid_size      = TOKEN_GRID_SIZE,
        confidence_threshold = args.threshold,
        learnable_threshold  = False,
    )

    # ---- Sample list ---------------------------------------------------
    samples = build_sample_list(args.kitti_root)
    if not samples:
        logging.error('No samples found — check --kitti-root path.')
        return
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        logging.info(f'Limiting to {args.max_samples} samples (--max-samples)')
    logging.info(f'Total samples: {len(samples)}\n')

    # ---- Per-method record collectors ----------------------------------
    # Two evaluation masks per sample:
    #   (a) standard  — all GT-valid pixels  (valid_gt)
    #   (b) sgm_valid — GT-valid pixels where SGM also predicts disp > 0
    #                   This excludes SGM holes and sky/background, giving a
    #                   fairer comparison within SGM's own coverage region.
    records: dict[str, dict[str, list]] = {
        method: {'all': [], 'kitti15': [], 'kitti12': []}
        for method in ('sgm', 'dense', 'sparse', 'fused', 'fused_sparse')
    }
    records_sv: dict[str, dict[str, list]] = {
        method: {'all': [], 'kitti15': [], 'kitti12': []}
        for method in ('sgm', 'dense', 'sparse', 'fused', 'fused_sparse')
    }

    csv_rows: list[dict] = []
    t_dense_total  = 0.0
    t_sparse_total = 0.0

    for idx, s in enumerate(tqdm(samples, desc='Evaluating', unit='img')):

        # ---- Load inputs -----------------------------------------------
        left_bgr = cv2.imread(s['left'])
        if left_bgr is None:
            logging.warning(f"Cannot read image: {s['left']}"); continue

        gt_disp, valid_gt = read_kitti_gt(s['gt_disp'])
        sgm_disp          = read_pfm(s['sgm_pfm'])

        # Alignment confidence: original binary LR-check mask.
        # This is the anchor for scale+shift fitting; binary values ensure only
        # geometrically reliable SGM pixels constrain the regression.
        mismatch  = np.load(s['mismatch']).astype(bool)
        occlusion = np.load(s['occlusion']).astype(bool)
        conf_align = (~(mismatch | occlusion)).astype(np.float32)
        conf_align = gaussian_filter(conf_align, sigma=args.conf_sigma)

        # Routing confidence: PKRN from aggregated cost volume (hardware-friendly).
        # Used only for token pruning decisions — not for alignment.
        if os.path.exists(s.get('pkrn_cache', '')) or os.path.exists(s.get('right', '')):
            conf_route = load_pkrn_confidence(
                s, args.disparity_range, args.pkrn_min_dist, args.conf_sigma)
        else:
            conf_route = conf_align  # fallback when right image unavailable

        # SGM-valid mask: GT-valid pixels where SGM also has a non-zero prediction.
        # Resizes sgm_disp to gt_disp shape if needed (should match for KITTI).
        sgm_for_mask = sgm_disp
        if sgm_disp.shape != gt_disp.shape:
            sgm_for_mask = cv2.resize(sgm_disp, (gt_disp.shape[1], gt_disp.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        valid_sgm = valid_gt & (sgm_for_mask > 0)

        # ---- SGM metrics -----------------------------------------------
        sgm_metrics    = compute_metrics(sgm_disp,     gt_disp, valid_gt)
        sgm_metrics_sv = compute_metrics(sgm_for_mask, gt_disp, valid_sgm)
        for key in ('all', s['split']):
            records['sgm'][key].append(sgm_metrics)
            records_sv['sgm'][key].append(sgm_metrics_sv)

        # ---- Dense DA2 -------------------------------------------------
        t0          = time.perf_counter()
        depth_dense = model.infer_image(left_bgr, input_size=518)
        t_dense_total += time.perf_counter() - t0

        disp_dense, scale_d, shift_d = align_depth_to_sgm(
            depth_mono     = depth_dense,
            disparity_raw  = sgm_disp,
            confidence_map = conf_align,
            conf_threshold = args.conf_threshold,
        )
        # Resize disp_dense to gt_disp shape for metric computation
        disp_dense_eval = disp_dense
        if disp_dense.shape != gt_disp.shape:
            disp_dense_eval = cv2.resize(disp_dense, (gt_disp.shape[1], gt_disp.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
        dense_metrics    = compute_metrics(disp_dense_eval, gt_disp, valid_gt)
        dense_metrics_sv = compute_metrics(disp_dense_eval, gt_disp, valid_sgm)
        for key in ('all', s['split']):
            records['dense'][key].append(dense_metrics)
            records_sv['dense'][key].append(dense_metrics_sv)

        # ---- Fused SGM + Dense DA2 -------------------------------------
        # Use SGM where confident, DA2 elsewhere.
        # conf_align (binary LR-check, Gaussian-smoothed) serves as the
        # confidence signal; conf_threshold controls the blend boundary.
        disp_fused = fuse_sgm_da2(
            sgm_disp       = sgm_for_mask,   # already at gt_disp resolution
            da2_aligned    = disp_dense_eval,
            confidence_map = cv2.resize(conf_align,
                                        (gt_disp.shape[1], gt_disp.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
                             if conf_align.shape != gt_disp.shape else conf_align,
            conf_threshold = args.conf_threshold,
        )
        fused_metrics    = compute_metrics(disp_fused, gt_disp, valid_gt)
        fused_metrics_sv = compute_metrics(disp_fused, gt_disp, valid_sgm)
        for key in ('all', s['split']):
            records['fused'][key].append(fused_metrics)
            records_sv['fused'][key].append(fused_metrics_sv)

        # ---- Sparse DA2 (SGM-ViT) --------------------------------------
        if not args.no_sparse:
            # Build prune mask from PKRN routing confidence
            conf_tensor  = torch.from_numpy(conf_route).unsqueeze(0).unsqueeze(0)
            dummy_tokens = torch.zeros(1, N, embed_dim)
            with torch.no_grad():
                routing = router(conf_tensor, dummy_tokens)

            prune_mask_1d = torch.zeros(N, dtype=torch.bool)
            prune_mask_1d[routing['prune_idx'][0]] = True

            t0 = time.perf_counter()
            if args.sparse_mode == 'mask':
                depth_sparse = run_masked_sparse_da2(
                    model         = model,
                    image_bgr     = left_bgr,
                    prune_mask    = prune_mask_1d,
                    input_size    = 518,
                    prune_layer   = args.prune_layer,
                    do_reassembly = not args.no_reassembly,
                )
            else:
                depth_sparse = run_sparse_da2(
                    model         = model,
                    image_bgr     = left_bgr,
                    prune_mask    = prune_mask_1d,
                    input_size    = 518,
                    prune_layer   = args.prune_layer,
                    do_reassembly = not args.no_reassembly,
                )
            t_sparse_total += time.perf_counter() - t0

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
            sparse_metrics    = compute_metrics(disp_sparse_eval, gt_disp, valid_gt)
            sparse_metrics_sv = compute_metrics(disp_sparse_eval, gt_disp, valid_sgm)
            prune_ratio       = float(routing['prune_ratio'])

            # ---- Fused SGM + Sparse DA2 -----------------------------------
            conf_for_fuse_sp = conf_align
            if conf_align.shape != gt_disp.shape:
                conf_for_fuse_sp = cv2.resize(conf_align,
                                              (gt_disp.shape[1], gt_disp.shape[0]),
                                              interpolation=cv2.INTER_LINEAR)
            disp_fused_sparse = fuse_sgm_da2(
                sgm_disp       = sgm_for_mask,
                da2_aligned    = disp_sparse_eval,
                confidence_map = conf_for_fuse_sp,
                conf_threshold = args.conf_threshold,
            )
            fused_sp_metrics    = compute_metrics(disp_fused_sparse, gt_disp, valid_gt)
            fused_sp_metrics_sv = compute_metrics(disp_fused_sparse, gt_disp, valid_sgm)
        else:
            nan_m = {'epe': float('nan'), 'd1': float('nan'), 'n': 0}
            sparse_metrics = sparse_metrics_sv = nan_m
            fused_sp_metrics = fused_sp_metrics_sv = nan_m
            prune_ratio = float('nan')

        for key in ('all', s['split']):
            records['sparse'][key].append(sparse_metrics)
            records_sv['sparse'][key].append(sparse_metrics_sv)
            records['fused_sparse'][key].append(fused_sp_metrics)
            records_sv['fused_sparse'][key].append(fused_sp_metrics_sv)

        # ---- CSV row ---------------------------------------------------
        csv_rows.append({
            'idx':             idx,
            'split':           s['split'],
            'image':           os.path.basename(s['left']),
            'sgm_epe':         f"{sgm_metrics['epe']:.4f}",
            'sgm_d1':          f"{sgm_metrics['d1']:.4f}",
            'sgm_epe_sv':      f"{sgm_metrics_sv['epe']:.4f}",
            'sgm_d1_sv':       f"{sgm_metrics_sv['d1']:.4f}",
            'dense_epe':       f"{dense_metrics['epe']:.4f}",
            'dense_d1':        f"{dense_metrics['d1']:.4f}",
            'dense_epe_sv':    f"{dense_metrics_sv['epe']:.4f}",
            'dense_d1_sv':     f"{dense_metrics_sv['d1']:.4f}",
            'fused_epe':       f"{fused_metrics['epe']:.4f}",
            'fused_d1':        f"{fused_metrics['d1']:.4f}",
            'fused_epe_sv':    f"{fused_metrics_sv['epe']:.4f}",
            'fused_d1_sv':     f"{fused_metrics_sv['d1']:.4f}",
            'sparse_epe':      f"{sparse_metrics['epe']:.4f}",
            'sparse_d1':       f"{sparse_metrics['d1']:.4f}",
            'sparse_epe_sv':   f"{sparse_metrics_sv['epe']:.4f}",
            'sparse_d1_sv':    f"{sparse_metrics_sv['d1']:.4f}",
            'fused_sp_epe':    f"{fused_sp_metrics['epe']:.4f}",
            'fused_sp_d1':     f"{fused_sp_metrics['d1']:.4f}",
            'fused_sp_epe_sv': f"{fused_sp_metrics_sv['epe']:.4f}",
            'fused_sp_d1_sv':  f"{fused_sp_metrics_sv['d1']:.4f}",
            'prune_pct':       f"{prune_ratio*100:.1f}" if not np.isnan(prune_ratio) else 'n/a',
            'scale_d':         f"{scale_d:.4f}",
            'shift_d':         f"{shift_d:.4f}",
        })

        # ---- Periodic console update -----------------------------------
        if (idx + 1) % max(1, len(samples) // 10) == 0 or idx < 5:
            logging.info(
                f"[{idx+1:3d}/{len(samples)}]  "
                f"SGM EPE={sgm_metrics['epe']:.2f} D1={sgm_metrics['d1']:.2f}%  "
                f"Dense EPE={dense_metrics['epe']:.2f} D1={dense_metrics['d1']:.2f}%"
                + (f"  Sparse EPE={sparse_metrics['epe']:.2f} D1={sparse_metrics['d1']:.2f}%"
                   if not args.no_sparse else '')
            )

    # ---- Aggregate and print results -----------------------------------
    splits_to_report = ['all', 'kitti15', 'kitti12']
    tag_map = {'all': 'COMBINED', 'kitti15': 'KITTI-2015', 'kitti12': 'KITTI-2012'}

    def _format_table(rec: dict, rec_sv: dict, split: str, no_sparse: bool) -> str:
        r_sgm    = aggregate(rec['sgm'][split])
        r_dense  = aggregate(rec['dense'][split])
        r_fused  = aggregate(rec['fused'][split])
        r_sparse = aggregate(rec['sparse'][split])
        r_fused_sp = aggregate(rec['fused_sparse'][split])
        r_sgm_sv    = aggregate(rec_sv['sgm'][split])
        r_dense_sv  = aggregate(rec_sv['dense'][split])
        r_fused_sv  = aggregate(rec_sv['fused'][split])
        r_sparse_sv = aggregate(rec_sv['sparse'][split])
        r_fused_sp_sv = aggregate(rec_sv['fused_sparse'][split])

        n_imgs   = r_sgm['n']
        n_px_all = r_sgm['pixel_n']
        n_px_sv  = r_sgm_sv['pixel_n']
        if n_imgs == 0:
            return ''

        tag = tag_map[split]
        sep = '-' * 60
        out = (
            f"\n  {tag}  ({n_imgs} images"
            f"  |  all: {n_px_all:,} px  |  sv: {n_px_sv:,} px"
            f"  [{100*n_px_sv/max(n_px_all,1):.1f}% SGM coverage])\n"
            f"  {'Method':<24s}  {'EPE (all)':>9s}  {'D1% (all)':>9s}"
            f"  {'EPE (sv)':>9s}  {'D1% (sv)':>9s}\n"
            f"  {sep}\n"
            f"  {'SGM (pre-computed)':<24s}"
            f"  {r_sgm['epe']:>9.4f}  {r_sgm['d1']:>9.4f}"
            f"  {r_sgm_sv['epe']:>9.4f}  {r_sgm_sv['d1']:>9.4f}\n"
            f"  {'Dense DA2 + align':<24s}"
            f"  {r_dense['epe']:>9.4f}  {r_dense['d1']:>9.4f}"
            f"  {r_dense_sv['epe']:>9.4f}  {r_dense_sv['d1']:>9.4f}\n"
            f"  {'Fused SGM + Dense DA2':<24s}"
            f"  {r_fused['epe']:>9.4f}  {r_fused['d1']:>9.4f}"
            f"  {r_fused_sv['epe']:>9.4f}  {r_fused_sv['d1']:>9.4f}\n"
        )
        if not no_sparse:
            sparse_label = f"Sparse DA2 ({args.sparse_mode})"
            fused_sp_label = f"Fused SGM + Sparse DA2"
            out += (
                f"  {sparse_label:<24s}"
                f"  {r_sparse['epe']:>9.4f}  {r_sparse['d1']:>9.4f}"
                f"  {r_sparse_sv['epe']:>9.4f}  {r_sparse_sv['d1']:>9.4f}\n"
                f"  {fused_sp_label:<24s}"
                f"  {r_fused_sp['epe']:>9.4f}  {r_fused_sp['d1']:>9.4f}"
                f"  {r_fused_sp_sv['epe']:>9.4f}  {r_fused_sp_sv['d1']:>9.4f}\n"
            )
        out += (
            f"  (all = all GT-valid pixels;  "
            f"sv = GT-valid ∩ SGM-predicted pixels only)\n"
        )
        return out

    header = f"\n{'='*66}\n  SGM-ViT Evaluation Results\n{'='*66}"
    logging.info(header)

    result_lines: list[str] = [header]
    for split in splits_to_report:
        line = _format_table(records, records_sv, split, args.no_sparse)
        if line:
            logging.info(line)
            result_lines.append(line)

    n_samples = len(samples)
    timing = (
        f"\n  Timing ({n_samples} images):\n"
        f"    Dense DA2  total: {t_dense_total:.1f}s  avg: {t_dense_total/max(1,n_samples)*1e3:.1f}ms/img\n"
    )
    if not args.no_sparse:
        timing += (
            f"    Sparse DA2 total: {t_sparse_total:.1f}s  avg: {t_sparse_total/max(1,n_samples)*1e3:.1f}ms/img\n"
        )
    logging.info(timing)

    # ---- Save results --------------------------------------------------
    txt_path = os.path.join(args.out_dir, 'eval_results.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(result_lines))
        f.write(timing)
    logging.info(f"Results saved: {txt_path}")

    csv_path = os.path.join(args.out_dir, 'eval_per_image.csv')
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
        description='SGM-ViT KITTI disparity accuracy evaluation.'
    )
    p.add_argument('--kitti-root', default='/nfs/usrhome/pdongaa/dataczy/kitti',
        help='KITTI dataset root (contains training/ and kitti2012/).')
    p.add_argument('--weights',
        default=os.path.join(_PROJECT_DIR, 'Depth-Anything-V2', 'checkpoints',
                             'depth_anything_v2_vits.pth'),
        help='Path to DepthAnythingV2 checkpoint.')
    p.add_argument('--encoder', default='vits',
        choices=list(DA2_MODEL_CONFIGS.keys()))
    p.add_argument('--threshold', type=float, default=0.65,
        help='Token pruning confidence threshold θ.')
    p.add_argument('--prune-layer', type=int, default=0,
        help='ViT block at which pruning starts (0-based).')
    p.add_argument('--sparse-mode', default='mask',
        choices=['zero', 'mask'],
        help="Sparse attention mode: 'mask' = GAS (Gather-Attend-Scatter), "
             "'zero' = legacy token zeroing. (default: mask)")
    p.add_argument('--no-reassembly', action='store_true',
        help='Disable token re-assembly before the DPT decoder.')
    p.add_argument('--no-sparse', action='store_true',
        help='Skip sparse DA2 evaluation (only SGM and dense DA2).')
    p.add_argument('--conf-threshold', type=float, default=0.7,
        help='Minimum confidence to use a pixel for depth alignment.')
    p.add_argument('--conf-sigma', type=float, default=5.0,
        help='Gaussian σ for smoothing the PKRN confidence map.')
    p.add_argument('--disparity-range', type=int, default=128,
        help='SGM disparity search range (must be multiple of 4). '
             'Used when computing PKRN live (cache miss).')
    p.add_argument('--pkrn-min-dist', type=int, default=1,
        help='Minimum disparity gap from winner for PKRN second-best search.')
    p.add_argument('--max-samples', type=int, default=0,
        help='Evaluate only the first N samples (0 = all).')
    p.add_argument('--out-dir',
        default=os.path.join(_PROJECT_DIR, 'results', 'eval_kitti'),
        help='Directory to save results and log.')
    p.add_argument('--cpu', action='store_true',
        help='Force CPU inference.')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
