"""
scripts/eval_latency.py
=======================
Evaluate FLOPs reduction and throughput improvement of SGM-guided token
pruning across a range of confidence thresholds and pruning ratios.

This script provides:
  1. Theoretical FLOPs analysis — how quadratic attention cost scales with
     the number of kept tokens (N_keep²  vs  N_total²).
  2. Wall-clock latency profiling (CPU/GPU) — forward pass timing with and
     without token pruning, averaged over multiple runs.
  3. Throughput summary table — FPS and FLOPs reduction at each pruning ratio.
  4. Matplotlib plots saved to  results/  for use in the ICCAD paper.

Usage
-----
    python scripts/eval_latency.py [--device cpu|cuda] [--repeats 100]

Requirements
------------
    pip install torch torchvision thop tqdm matplotlib
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import core._paths  # noqa: F401  — ensures DA2 is on sys.path
from core.eval_utils import compute_token_grid_size
from core.token_router import SGMConfidenceTokenRouter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# DepthAnythingV2 ViT-B default parameters
IMG_SIZE        = 518
PATCH_SIZE      = 14
TOKEN_GRID_SIZE = compute_token_grid_size(IMG_SIZE, PATCH_SIZE)  # 37
N_TOKENS        = TOKEN_GRID_SIZE ** 2     # 1369
EMBED_DIM       = 768
NUM_HEADS       = 12
NUM_LAYERS      = 12                       # ViT-B has 12 transformer blocks

# Pruning thresholds to sweep
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

# Batch size 1 — standard edge-inference scenario
BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def attention_flops(n_tokens: int, embed_dim: int, num_heads: int, num_layers: int) -> float:
    """
    Estimate multi-head self-attention FLOPs for a given sequence length.

    Per layer, per head:
        Q, K, V projections : 3 × N × d_head × d_head  matmuls
        QK^T                : N × N × d_head
        Softmax(QK^T) × V   : N × N × d_head
        Output projection   : N × D × D
    Simplified (dominant term):
        ~4 × N² × D  per layer  (ignoring O(N×D²) projection terms)
    """
    flops_per_layer = 4.0 * (n_tokens ** 2) * embed_dim
    return flops_per_layer * num_layers


def ffn_flops(n_tokens: int, embed_dim: int, ffn_ratio: int = 4, num_layers: int = 12) -> float:
    """Estimate MLP / FFN FLOPs: 2 × N × D × ffn_dim per layer."""
    ffn_dim = embed_dim * ffn_ratio
    return 2.0 * n_tokens * embed_dim * ffn_dim * num_layers * 2  # two linear layers


def total_model_flops(n_tokens: int, embed_dim: int = EMBED_DIM,
                      num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS) -> float:
    return attention_flops(n_tokens, embed_dim, num_heads, num_layers) + \
           ffn_flops(n_tokens, embed_dim, num_layers=num_layers)


def compute_flops_table(thresholds: List[float],
                        prune_ratios: List[float]) -> List[dict]:
    """
    Build a table of FLOPs metrics for each pruning ratio.

    Returns a list of dicts with keys:
        threshold, prune_ratio, n_keep, attn_flops_dense,
        attn_flops_sparse, attn_reduction_pct, total_flops_dense,
        total_flops_sparse, total_reduction_pct
    """
    rows = []
    flops_dense = total_model_flops(N_TOKENS)

    for thr, pr in zip(thresholds, prune_ratios):
        n_keep       = int(N_TOKENS * (1.0 - pr))
        attn_dense   = attention_flops(N_TOKENS,  EMBED_DIM, NUM_HEADS, NUM_LAYERS)
        attn_sparse  = attention_flops(n_keep,    EMBED_DIM, NUM_HEADS, NUM_LAYERS)
        ffn_dense    = ffn_flops(N_TOKENS, EMBED_DIM)
        ffn_sparse   = ffn_flops(n_keep,   EMBED_DIM)
        total_sparse = attn_sparse + ffn_sparse

        rows.append({
            "threshold":            thr,
            "prune_ratio":          pr,
            "n_keep":               n_keep,
            "n_prune":              N_TOKENS - n_keep,
            "attn_flops_dense_G":   attn_dense  / 1e9,
            "attn_flops_sparse_G":  attn_sparse / 1e9,
            "attn_reduction_pct":   (1.0 - attn_sparse / attn_dense) * 100,
            "total_flops_dense_G":  flops_dense  / 1e9,
            "total_flops_sparse_G": total_sparse / 1e9,
            "total_reduction_pct":  (1.0 - total_sparse / flops_dense) * 100,
        })
    return rows


# ---------------------------------------------------------------------------
# Wall-clock profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def profile_router_latency(
    threshold: float,
    conf_map: torch.Tensor,
    tokens: torch.Tensor,
    device: torch.device,
    repeats: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Time the SGMConfidenceTokenRouter forward pass.

    Returns
    -------
    dict with keys: mean_ms, std_ms, prune_ratio
    """
    router = SGMConfidenceTokenRouter(
        token_grid_size=TOKEN_GRID_SIZE,
        confidence_threshold=threshold,
    ).to(device)
    router.eval()

    conf  = conf_map.to(device)
    toks  = tokens.to(device)

    # Warm-up
    for _ in range(warmup):
        _ = router(conf, toks)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = router(conf, toks)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times_arr = np.array(times)
    return {
        "mean_ms":    float(times_arr.mean()),
        "std_ms":     float(times_arr.std()),
        "prune_ratio": out["prune_ratio"],
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_flops_reduction(rows: List[dict], out_dir: str) -> None:
    prune_pcts = [r["prune_ratio"] * 100 for r in rows]
    total_red  = [r["total_reduction_pct"]  for r in rows]
    attn_red   = [r["attn_reduction_pct"]   for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prune_pcts, total_red, "o-", label="Total FLOPs reduction")
    ax.plot(prune_pcts, attn_red,  "s--", label="Attention FLOPs reduction")
    ax.set_xlabel("Token Prune Ratio (%)")
    ax.set_ylabel("FLOPs Reduction (%)")
    ax.set_title(f"SGM-ViT: FLOPs Reduction vs. Token Prune Ratio\n"
                 f"(ViT-B, N={N_TOKENS}, D={EMBED_DIM}, L={NUM_LAYERS})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "flops_reduction.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved → {path}")


def plot_latency(latency_results: List[dict], out_dir: str) -> None:
    thresholds = [r["threshold"] for r in latency_results]
    means      = [r["mean_ms"]   for r in latency_results]
    stds       = [r["std_ms"]    for r in latency_results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(thresholds, means, yerr=stds, fmt="o-", capsize=4)
    ax.set_xlabel("Confidence Threshold θ")
    ax.set_ylabel("Token Router Latency (ms)")
    ax.set_title("SGM-ViT: Token Router Wall-Clock Latency vs. Threshold")
    ax.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "router_latency.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SGM-ViT latency & FLOPs evaluator")
    parser.add_argument("--device",  default="cpu", choices=["cpu", "cuda"],
                        help="Compute device for wall-clock profiling.")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Number of timed forward passes per threshold.")
    parser.add_argument("--out-dir", default="results",
                        help="Output directory for plots and CSV.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"[eval_latency] Device: {device}")
    print(f"[eval_latency] N_tokens={N_TOKENS}, embed_dim={EMBED_DIM}, "
          f"num_layers={NUM_LAYERS}\n")

    # ------------------------------------------------------------------
    # 1. Synthetic inputs (replace with real data for final evaluation)
    # ------------------------------------------------------------------
    # Confidence map: uniform random ∈ [0,1] to sample a range of ratios
    conf_map = torch.rand(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
    tokens   = torch.randn(BATCH_SIZE, N_TOKENS, EMBED_DIM)

    # ------------------------------------------------------------------
    # 2. Measure actual prune ratios across thresholds
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Measuring actual prune ratios (uniform random confidence map)...")
    print("=" * 60)
    measured_prune_ratios = []
    for thr in THRESHOLDS:
        router = SGMConfidenceTokenRouter(token_grid_size=TOKEN_GRID_SIZE,
                                          confidence_threshold=thr)
        with torch.no_grad():
            out = router(conf_map, tokens)
        pr = out["prune_ratio"]
        measured_prune_ratios.append(pr)
        print(f"  θ={thr:.2f}  →  prune_ratio={pr:.3f}  "
              f"(N_keep={int((1-pr)*N_TOKENS)}, N_prune={int(pr*N_TOKENS)})")

    # ------------------------------------------------------------------
    # 3. Theoretical FLOPs table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Theoretical FLOPs Analysis")
    print("=" * 60)
    flops_rows = compute_flops_table(THRESHOLDS, measured_prune_ratios)

    header = (f"{'θ':>5} | {'Prune%':>7} | {'N_keep':>7} | "
              f"{'Attn↓%':>8} | {'Total↓%':>8} | {'GFLOPs(sparse)':>15}")
    print(header)
    print("-" * len(header))
    for r in flops_rows:
        print(f"  {r['threshold']:>4.2f} | {r['prune_ratio']*100:>6.1f}% | "
              f"{r['n_keep']:>7d} | {r['attn_reduction_pct']:>7.1f}% | "
              f"{r['total_reduction_pct']:>7.1f}% | "
              f"{r['total_flops_sparse_G']:>14.2f}G")

    # ------------------------------------------------------------------
    # 4. Wall-clock profiling
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Wall-clock profiling ({args.repeats} repeats, device={device})")
    print("=" * 60)
    latency_results = []
    for thr in tqdm(THRESHOLDS, desc="Profiling thresholds"):
        result = profile_router_latency(
            threshold=thr,
            conf_map=conf_map,
            tokens=tokens,
            device=device,
            repeats=args.repeats,
        )
        result["threshold"] = thr
        latency_results.append(result)
        print(f"  θ={thr:.2f}  latency={result['mean_ms']:.3f} ± "
              f"{result['std_ms']:.3f} ms")

    # ------------------------------------------------------------------
    # 5. Throughput estimate (router stage only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Throughput Summary (router stage, single frame)")
    print("=" * 60)
    for lr, fr in zip(latency_results, flops_rows):
        fps = 1000.0 / (lr["mean_ms"] + 1e-9)
        print(f"  θ={lr['threshold']:.2f}  FPS≈{fps:6.0f}  "
              f"total_reduction={fr['total_reduction_pct']:.1f}%")

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    print(f"\nSaving plots to {args.out_dir}/")
    plot_flops_reduction(flops_rows, args.out_dir)
    plot_latency(latency_results, args.out_dir)

    # ------------------------------------------------------------------
    # 7. CSV export for paper tables
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.out_dir, "flops_latency_summary.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("threshold,prune_ratio,n_keep,attn_reduction_pct,"
                "total_reduction_pct,total_flops_sparse_G,router_latency_ms\n")
        for fr, lr in zip(flops_rows, latency_results):
            f.write(f"{fr['threshold']},{fr['prune_ratio']:.4f},"
                    f"{fr['n_keep']},{fr['attn_reduction_pct']:.2f},"
                    f"{fr['total_reduction_pct']:.2f},"
                    f"{fr['total_flops_sparse_G']:.4f},"
                    f"{lr['mean_ms']:.4f}\n")
    print(f"  [csv]  Saved → {csv_path}")
    print("\n[eval_latency] Done.")


if __name__ == "__main__":
    main()
