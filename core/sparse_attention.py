"""
core/sparse_attention.py
========================
Gather-Attend-Scatter (GAS) sparse attention for SGM-ViT.

Instead of zeroing pruned tokens (which causes attention pollution via
``softmax(Q·0/√d) = exp(0)`` distributing uniform attention to zero vectors),
this module physically excludes pruned tokens from the attention computation.

For each ViT block >= prune_layer::

    1. GATHER   extract [CLS, kept_patches] into a compact sequence
    2. ATTEND   standard multi-head attention on the short sequence
    3. SCATTER  write attended outputs back; pruned tokens retain previous features
    4. FFN      MLP on all tokens (per-token, no cross-token dependency)

This preserves pretrained DINOv2 weight compatibility because:
  * Positional encoding is added once in ``prepare_tokens_with_masks()``
    before any block — each token's positional identity is already embedded.
  * Attention is content-based (no relative positional encoding).
  * All linear layers (qkv, proj, fc1, fc2) are per-token operations.
  * LayerScale gamma is a per-channel scalar, shape-agnostic over N.

Hardware story (ICCAD 2025)::

    SGM engine → confidence map → binary mask (1 LUT per token)
    Token dispatcher reads only kept tokens into attention PE array
    Attention PEs compute on N_keep tokens, FLOPs ∝ N_keep²
    Scatter unit writes back to original SRAM positions
    FFN PEs process all tokens in streaming order
    Reassembly unit: bilinear interpolation (reuses DPT pyramid HW)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def gas_block_forward(
    block: nn.Module,
    x: Tensor,
    keep_indices: Tensor,
    num_prefix: int = 1,
    ffn_on_all: bool = True,
) -> Tensor:
    """
    Gather-Attend-Scatter forward for a single ViT block (eval mode).

    Replicates the Block eval-mode forward::

        x = x + ls1(attn(norm1(x)))     # attention + residual
        x = x + ls2(mlp(norm2(x)))      # FFN + residual

    but restricts attention to only prefix (CLS/registers) and kept patches.

    Parameters
    ----------
    block : nn.Module
        A DINOv2 Block / NestedTensorBlock with attributes:
        ``norm1``, ``attn``, ``ls1``, ``norm2``, ``mlp``, ``ls2``.
    x : (B, L, D)
        Full token sequence ``[CLS | patch_0 | … | patch_{N-1}]``.
        ``L = num_prefix + N``.
    keep_indices : (N_keep,) LongTensor
        0-based indices into the **patch** portion (range ``[0, N-1]``)
        of tokens to keep for attention.
    num_prefix : int
        Number of prefix tokens before patches.  ``1`` for CLS only
        (ViT-S default with ``num_register_tokens=0``).
    ffn_on_all : bool
        If True, MLP runs on all tokens (pruned included).
        If False, MLP runs only on prefix + kept tokens.

    Returns
    -------
    x_out : (B, L, D)
    """
    B, L, D = x.shape

    # --- Build gather indices: [prefix..., kept_patches...] ---
    prefix_idx = torch.arange(num_prefix, device=x.device)
    kept_global = keep_indices + num_prefix
    gather_idx = torch.cat([prefix_idx, kept_global])       # (M,)

    # --- GATHER ---
    x_short = x[:, gather_idx, :]                           # (B, M, D)

    # --- ATTEND (standard block attention on short sequence) ---
    attn_out = block.ls1(block.attn(block.norm1(x_short)))  # (B, M, D)
    x_short = x_short + attn_out

    # --- SCATTER ---
    x_out = x.clone()
    x_out[:, gather_idx, :] = x_short

    # --- FFN ---
    if ffn_on_all:
        x_out = x_out + block.ls2(block.mlp(block.norm2(x_out)))
    else:
        x_ffn = x_out[:, gather_idx, :]
        ffn_out = block.ls2(block.mlp(block.norm2(x_ffn)))
        x_out[:, gather_idx, :] = x_ffn + ffn_out

    return x_out


def gas_get_intermediate_layers(
    backbone: nn.Module,
    x_input: Tensor,
    layer_indices: list[int],
    keep_indices: Tensor,
    prune_layer: int = 0,
    ffn_on_all: bool = True,
) -> list[tuple[Tensor, Tensor]]:
    """
    Run the DINOv2 backbone with GAS sparse attention, extracting
    intermediate features at specified block indices for the DPT decoder.

    Replaces ``backbone.get_intermediate_layers()`` for the GAS path.

    Parameters
    ----------
    backbone : DinoVisionTransformer
        The pretrained DINOv2 backbone (``model.pretrained``).
    x_input : (B, 3, H, W)
        Preprocessed input image tensor.
    layer_indices : list[int]
        Block indices at which to tap intermediate features.
        For ViT-S: ``[2, 5, 8, 11]``.
    keep_indices : (N_keep,) LongTensor
        0-based patch indices of tokens to keep.
    prune_layer : int
        First block index that uses GAS.
        Blocks ``[0, prune_layer)`` run dense.
        Blocks ``[prune_layer, depth)`` run GAS.
    ffn_on_all : bool
        Whether FFN is applied to all tokens or only kept tokens.

    Returns
    -------
    features : list of ``(patch_tokens, cls_token)``
        ``patch_tokens`` : (B, N, D) — full spatial grid (N = patch_h * patch_w)
        ``cls_token``    : (B, D)
        Format matches ``backbone.get_intermediate_layers(...,
        return_class_token=True)``.
    """
    nr = getattr(backbone, "num_register_tokens", 0)
    num_prefix = 1 + nr

    # Prepare tokens: patch embed + positional encoding + CLS + registers
    x = backbone.prepare_tokens_with_masks(x_input)  # (B, 1+nr+N, D)

    # Flat block list
    if getattr(backbone, "chunked_blocks", False):
        blocks = [b for chunk in backbone.blocks for b in chunk]
    else:
        blocks = list(backbone.blocks)

    layer_set = set(layer_indices)
    output = []

    for i, blk in enumerate(blocks):
        if i < prune_layer:
            # Dense forward — all tokens participate
            x = blk(x)
        else:
            # GAS forward — only kept tokens attend
            x = gas_block_forward(
                block=blk,
                x=x,
                keep_indices=keep_indices,
                num_prefix=num_prefix,
                ffn_on_all=ffn_on_all,
            )

        if i in layer_set:
            output.append(x)

    assert len(output) == len(layer_indices), (
        f"Expected {len(layer_indices)} intermediate outputs, got {len(output)}"
    )

    # Apply LayerNorm and separate patch_tokens / cls_token
    result: list[tuple[Tensor, Tensor]] = []
    for out in output:
        out_norm = backbone.norm(out)
        cls_token = out_norm[:, 0]                        # (B, D)
        patch_tokens = out_norm[:, num_prefix:]            # (B, N, D)
        result.append((patch_tokens, cls_token))

    return result
