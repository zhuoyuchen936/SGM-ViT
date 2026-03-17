"""
core/pruning_strategies.py
===========================
Unified module implementing 6 alternative token pruning mask generators
for SGM-ViT ablation studies.

All functions return a 1-D ``BoolTensor`` of shape ``(N,)`` where
``True = pruned`` and ``False = kept``, matching the convention used by
``demo.run_masked_sparse_da2``.

Strategies
----------
1. random          — uniform random token selection (control baseline)
2. topk_confidence — fixed-ratio: keep K tokens with lowest SGM confidence
3. inverse_conf    — prune LOW-confidence tokens (hypothesis validation)
4. checkerboard    — regular 2x2 spatial downsampling pattern
5. cls_attention   — prune tokens with lowest CLS attention score (content-aware)
6. hybrid          — combine SGM confidence + CLS attention scores
"""

from __future__ import annotations

import torch
import numpy as np
import cv2

from .eval_utils import pool_confidence as _pool_confidence


# ---------------------------------------------------------------------------
# 1. Random Pruning (control baseline)
# ---------------------------------------------------------------------------

def random_prune_mask(
    N: int,
    keep_ratio: float,
    seed: int | None = None,
) -> torch.BoolTensor:
    """
    Randomly select tokens to prune at a fixed keep ratio.

    Parameters
    ----------
    N          : total number of spatial tokens
    keep_ratio : fraction of tokens to keep (e.g. 0.8 keeps 80%)
    seed       : optional RNG seed for reproducibility

    Returns
    -------
    mask : (N,) bool Tensor, True = pruned
    """
    n_keep = max(1, int(round(N * keep_ratio)))
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    perm = torch.randperm(N, generator=gen)
    mask = torch.ones(N, dtype=torch.bool)
    mask[perm[:n_keep]] = False
    return mask


# ---------------------------------------------------------------------------
# 2. Top-K Confidence (fixed-ratio variant)
# ---------------------------------------------------------------------------

def topk_confidence_mask(
    conf_map: np.ndarray,
    token_grid_size: int,
    keep_ratio: float,
) -> torch.BoolTensor:
    """
    Keep the K tokens with LOWEST SGM confidence (fixed-ratio selection).

    Unlike threshold-based routing (variable count), this always keeps
    exactly ``N * keep_ratio`` tokens.

    Parameters
    ----------
    conf_map        : (H, W) float32 confidence map [0, 1]
    token_grid_size : side length of square token grid (e.g. 37)
    keep_ratio      : fraction of tokens to keep

    Returns
    -------
    mask : (N,) bool Tensor, True = pruned
    """
    N = token_grid_size * token_grid_size
    n_keep = max(1, int(round(N * keep_ratio)))

    # Pool confidence to token grid
    conf_grid = _pool_confidence(conf_map, token_grid_size)  # (G, G)
    conf_flat = torch.from_numpy(conf_grid.reshape(-1))      # (N,)

    # Keep tokens with LOWEST confidence (ascending sort, take first n_keep)
    _, indices = conf_flat.sort()
    mask = torch.ones(N, dtype=torch.bool)
    mask[indices[:n_keep]] = False
    return mask


# ---------------------------------------------------------------------------
# 3. Inverse Confidence (hypothesis validation)
# ---------------------------------------------------------------------------

def inverse_confidence_mask(
    conf_map: np.ndarray,
    token_grid_size: int,
    threshold: float = 0.65,
) -> torch.BoolTensor:
    """
    Prune LOW-confidence tokens (opposite of SGMConfidenceTokenRouter).

    The standard SGM router prunes HIGH-confidence tokens. This inverts
    the logic: tokens with confidence BELOW the threshold are pruned.

    Parameters
    ----------
    conf_map        : (H, W) float32 confidence map [0, 1]
    token_grid_size : side length of square token grid
    threshold       : confidence value; tokens with conf < threshold are pruned

    Returns
    -------
    mask : (N,) bool Tensor, True = pruned
    """
    N = token_grid_size * token_grid_size
    conf_grid = _pool_confidence(conf_map, token_grid_size)
    conf_flat = torch.from_numpy(conf_grid.reshape(-1))

    # Prune tokens with confidence BELOW the threshold (inverse of normal)
    mask = conf_flat < threshold
    return mask


# ---------------------------------------------------------------------------
# 4. Spatial Checkerboard (structured baseline)
# ---------------------------------------------------------------------------

def spatial_checkerboard_mask(
    patch_h: int,
    patch_w: int,
) -> torch.BoolTensor:
    """
    Prune tokens in a regular 2x2 checkerboard pattern.

    Every other token (in a row+col parity pattern) is pruned,
    giving ~50% pruning. No runtime decisions needed.

    Parameters
    ----------
    patch_h, patch_w : spatial dimensions of the token grid

    Returns
    -------
    mask : (patch_h * patch_w,) bool Tensor, True = pruned
    """
    rows = torch.arange(patch_h).unsqueeze(1).expand(patch_h, patch_w)
    cols = torch.arange(patch_w).unsqueeze(0).expand(patch_h, patch_w)
    # Prune where (row + col) is even → keeps ~50%
    checker = ((rows + cols) % 2 == 0)
    return checker.reshape(-1)


# ---------------------------------------------------------------------------
# 5. CLS Attention Score (content-aware)
# ---------------------------------------------------------------------------

def cls_attention_mask(
    model,
    image_bgr: np.ndarray,
    keep_ratio: float,
    warmup_block: int = 0,
    input_size: int = 518,
) -> tuple[torch.BoolTensor, torch.Tensor]:
    """
    Extract CLS token attention weights from a ViT block and prune
    tokens with the lowest attention scores.

    After running all blocks up to (and including) ``warmup_block``,
    manually compute Q*K^T at that block's attention layer to extract
    the CLS-to-patch attention distribution.

    Parameters
    ----------
    model        : DepthAnythingV2 model (eval mode, on device)
    image_bgr    : (H, W, 3) uint8 BGR input image
    keep_ratio   : fraction of tokens to keep
    warmup_block : which block to extract attention from (0-based)
    input_size   : model input resolution

    Returns
    -------
    mask       : (N,) bool Tensor (CPU), True = pruned
    cls_scores : (N,) float Tensor (CPU), CLS attention per patch token
    """
    device = next(model.parameters()).device
    backbone = model.pretrained

    # Prepare input
    image_tensor, (h, w) = model.image2tensor(image_bgr, input_size)
    patch_h = image_tensor.shape[-2] // 14
    patch_w = image_tensor.shape[-1] // 14
    N = patch_h * patch_w

    nr = getattr(backbone, "num_register_tokens", 0)
    num_prefix = 1 + nr

    # Run through the backbone up to the target block to get the input state
    with torch.no_grad():
        x = backbone.prepare_tokens_with_masks(image_tensor)  # (B, 1+nr+N, D)

        if getattr(backbone, "chunked_blocks", False):
            blocks = [b for chunk in backbone.blocks for b in chunk]
        else:
            blocks = list(backbone.blocks)

        # Run dense forward through blocks [0, warmup_block)
        for i in range(warmup_block):
            x = blocks[i](x)

        # At warmup_block: manually compute attention scores
        blk = blocks[warmup_block]
        x_normed = blk.norm1(x)  # (B, L, D)

        B, L, D = x_normed.shape
        qkv = blk.attn.qkv(x_normed)  # (B, L, 3*D)
        num_heads = blk.attn.num_heads
        head_dim = D // num_heads

        qkv = qkv.reshape(B, L, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, L, head_dim)

        scale = head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, L, L)
        attn_weights = attn_weights.softmax(dim=-1)

        # Extract CLS row (row 0) attention to patch tokens
        # attn_weights[:, :, 0, :] is CLS attending to all tokens
        cls_attn = attn_weights[:, :, 0, num_prefix:]  # (B, num_heads, N_actual)
        cls_attn = cls_attn.mean(dim=1)  # (B, N_actual) — average across heads
        cls_scores = cls_attn[0].cpu()  # (N_actual,)

    # Adapt to standard N if needed (non-square images)
    N_actual = cls_scores.shape[0]

    # Build mask: prune tokens with lowest CLS attention
    n_keep = max(1, int(round(N_actual * keep_ratio)))
    _, top_indices = cls_scores.topk(n_keep, largest=True)
    mask = torch.ones(N_actual, dtype=torch.bool)
    mask[top_indices] = False

    return mask, cls_scores


# ---------------------------------------------------------------------------
# 6. Hybrid SGM + CLS Attention
# ---------------------------------------------------------------------------

def hybrid_mask(
    conf_map: np.ndarray,
    cls_attn_scores: torch.Tensor,
    token_grid_size: int,
    keep_ratio: float,
    alpha: float = 0.5,
) -> torch.BoolTensor:
    """
    Combine SGM confidence and CLS attention into a single importance score.

    importance = alpha * (1 - conf_norm) + (1 - alpha) * cls_attn_norm

    High importance = should be KEPT (low SGM confidence = uncertain geometry,
    high CLS attention = ViT considers token important).

    Parameters
    ----------
    conf_map         : (H, W) float32 confidence map [0, 1]
    cls_attn_scores  : (N,) float Tensor, CLS attention per patch token
    token_grid_size  : side length of square token grid
    keep_ratio       : fraction of tokens to keep
    alpha            : weight for SGM signal (1.0 = pure SGM, 0.0 = pure CLS)

    Returns
    -------
    mask : (N,) bool Tensor, True = pruned
    """
    N = token_grid_size * token_grid_size

    # Pool and normalise SGM confidence
    conf_grid = _pool_confidence(conf_map, token_grid_size)
    conf_flat = torch.from_numpy(conf_grid.reshape(-1)).float()  # (N,)

    # Normalise confidence to [0, 1]
    c_min, c_max = conf_flat.min(), conf_flat.max()
    if c_max - c_min > 1e-8:
        conf_norm = (conf_flat - c_min) / (c_max - c_min)
    else:
        conf_norm = torch.zeros_like(conf_flat)

    # Handle shape mismatch: cls_attn_scores may differ from N (non-square)
    cls_scores = cls_attn_scores.float()
    if cls_scores.shape[0] != N:
        G = int(round(cls_scores.shape[0] ** 0.5))
        cls_2d = cls_scores.reshape(G, G).numpy()
        cls_2d = cv2.resize(cls_2d, (token_grid_size, token_grid_size),
                            interpolation=cv2.INTER_LINEAR)
        cls_scores = torch.from_numpy(cls_2d.reshape(-1))

    # Normalise CLS attention to [0, 1]
    a_min, a_max = cls_scores.min(), cls_scores.max()
    if a_max - a_min > 1e-8:
        attn_norm = (cls_scores - a_min) / (a_max - a_min)
    else:
        attn_norm = torch.zeros_like(cls_scores)

    # Combined importance: high = should keep
    # (1 - conf_norm): low confidence → high importance (uncertain → need ViT)
    # attn_norm: high attention → high importance (ViT considers it important)
    importance = alpha * (1.0 - conf_norm) + (1.0 - alpha) * attn_norm

    # Keep top-K by importance
    n_keep = max(1, int(round(N * keep_ratio)))
    _, top_indices = importance.topk(n_keep, largest=True)
    mask = torch.ones(N, dtype=torch.bool)
    mask[top_indices] = False

    return mask


