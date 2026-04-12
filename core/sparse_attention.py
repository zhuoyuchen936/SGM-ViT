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

from core.adaptive_precision import fake_quantize_tensor


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


# -----------------------------------------------------------------------
# Token Merge: representative-token attention with full-grid scatter
# -----------------------------------------------------------------------

def merge_block_forward(
    block: nn.Module,
    x: Tensor,
    rep_patch_indices: Tensor,
    member_to_rep_local: Tensor,
    num_prefix: int = 1,
) -> Tensor:
    """
    Merge-aware forward for a single ViT block.

    The attention path operates only on ``[prefix, representatives]``.
    The attended representative outputs are then scattered back to the full
    patch grid so that all tokens in the same merge group share the same
    attention-updated value before the FFN.
    """
    prefix_idx = torch.arange(num_prefix, device=x.device)
    rep_patch_indices = rep_patch_indices.to(x.device)
    member_to_rep_local = member_to_rep_local.to(x.device)

    rep_global = rep_patch_indices + num_prefix
    gather_idx = torch.cat([prefix_idx, rep_global])

    x_short = x[:, gather_idx, :]
    attn_out = block.ls1(block.attn(block.norm1(x_short)))
    x_short = x_short + attn_out

    x_out = x.clone()
    x_out[:, :num_prefix, :] = x_short[:, :num_prefix, :]

    rep_out = x_short[:, num_prefix:, :]
    x_out[:, num_prefix:, :] = rep_out[:, member_to_rep_local, :]

    x_out = x_out + block.ls2(block.mlp(block.norm2(x_out)))
    return x_out


def gas_get_intermediate_layers_merge(
    backbone: nn.Module,
    x_input: Tensor,
    layer_indices: list[int],
    rep_patch_indices: Tensor,
    member_to_rep_local: Tensor,
    merge_layer: int = 0,
) -> list[tuple[Tensor, Tensor]]:
    """
    Run the DINOv2 backbone with confidence-guided token merge.

    Blocks before ``merge_layer`` run dense. Blocks from ``merge_layer`` onward
    use representative-token attention followed by full-grid scatter.
    """
    nr = getattr(backbone, "num_register_tokens", 0)
    num_prefix = 1 + nr

    x = backbone.prepare_tokens_with_masks(x_input)

    if getattr(backbone, "chunked_blocks", False):
        blocks = [b for chunk in backbone.blocks for b in chunk]
    else:
        blocks = list(backbone.blocks)

    layer_set = set(layer_indices)
    output = []

    for i, blk in enumerate(blocks):
        if i < merge_layer:
            x = blk(x)
        else:
            x = merge_block_forward(
                block=blk,
                x=x,
                rep_patch_indices=rep_patch_indices,
                member_to_rep_local=member_to_rep_local,
                num_prefix=num_prefix,
            )

        if i in layer_set:
            output.append(x)

    assert len(output) == len(layer_indices), (
        f"Expected {len(layer_indices)} intermediate outputs, got {len(output)}"
    )

    result: list[tuple[Tensor, Tensor]] = []
    for out in output:
        out_norm = backbone.norm(out)
        cls_token = out_norm[:, 0]
        patch_tokens = out_norm[:, num_prefix:]
        result.append((patch_tokens, cls_token))

    return result


# -----------------------------------------------------------------------
# CAPS Merge: confidence-aware adaptive precision on representative tokens
# -----------------------------------------------------------------------

def caps_merge_block_forward(
    block: nn.Module,
    x: Tensor,
    rep_patch_indices: Tensor,
    member_to_rep_local: Tensor,
    high_precision_local_mask: Tensor,
    num_prefix: int = 1,
    high_precision_bits: int = 8,
    low_precision_bits: int = 4,
) -> Tensor:
    """
    Merge-aware forward with confidence-aware precision scheduling.

    Representatives are gathered as in token merge. Sensitive groups are
    processed with ``high_precision_bits`` while the rest are fake-quantized to
    ``low_precision_bits`` before and after attention.
    """
    prefix_idx = torch.arange(num_prefix, device=x.device)
    rep_patch_indices = rep_patch_indices.to(x.device)
    member_to_rep_local = member_to_rep_local.to(x.device)
    hp_mask = high_precision_local_mask.to(x.device)

    rep_global = rep_patch_indices + num_prefix
    gather_idx = torch.cat([prefix_idx, rep_global])

    x_short = x[:, gather_idx, :]
    x_short_q = x_short.clone()

    x_short_q[:, :num_prefix, :] = fake_quantize_tensor(
        x_short_q[:, :num_prefix, :],
        high_precision_bits,
    )

    if hp_mask.any():
        hp_rep_idx = torch.where(hp_mask)[0] + num_prefix
        x_short_q[:, hp_rep_idx, :] = fake_quantize_tensor(
            x_short_q[:, hp_rep_idx, :],
            high_precision_bits,
        )
    if (~hp_mask).any():
        lp_rep_idx = torch.where(~hp_mask)[0] + num_prefix
        x_short_q[:, lp_rep_idx, :] = fake_quantize_tensor(
            x_short_q[:, lp_rep_idx, :],
            low_precision_bits,
        )

    attn_out = block.ls1(block.attn(block.norm1(x_short_q)))

    if hp_mask.any():
        hp_rep_idx = torch.where(hp_mask)[0]
        attn_out[:, hp_rep_idx + num_prefix, :] = fake_quantize_tensor(
            attn_out[:, hp_rep_idx + num_prefix, :],
            high_precision_bits,
        )
    if (~hp_mask).any():
        lp_rep_idx = torch.where(~hp_mask)[0]
        attn_out[:, lp_rep_idx + num_prefix, :] = fake_quantize_tensor(
            attn_out[:, lp_rep_idx + num_prefix, :],
            low_precision_bits,
        )

    x_short = x_short + attn_out

    x_out = x.clone()
    x_out[:, :num_prefix, :] = x_short[:, :num_prefix, :]
    rep_out = x_short[:, num_prefix:, :]
    x_out[:, num_prefix:, :] = rep_out[:, member_to_rep_local, :]
    x_out = x_out + block.ls2(block.mlp(block.norm2(x_out)))
    return x_out


def gas_get_intermediate_layers_caps_merge(
    backbone: nn.Module,
    x_input: Tensor,
    layer_indices: list[int],
    rep_patch_indices: Tensor,
    member_to_rep_local: Tensor,
    high_precision_local_mask: Tensor,
    merge_layer: int = 0,
    high_precision_bits: int = 8,
    low_precision_bits: int = 4,
) -> list[tuple[Tensor, Tensor]]:
    """
    Run the DINOv2 backbone with merge + adaptive precision on representatives.
    """
    nr = getattr(backbone, "num_register_tokens", 0)
    num_prefix = 1 + nr

    x = backbone.prepare_tokens_with_masks(x_input)

    if getattr(backbone, "chunked_blocks", False):
        blocks = [b for chunk in backbone.blocks for b in chunk]
    else:
        blocks = list(backbone.blocks)

    layer_set = set(layer_indices)
    output = []

    for i, blk in enumerate(blocks):
        if i < merge_layer:
            x = blk(x)
        else:
            x = caps_merge_block_forward(
                block=blk,
                x=x,
                rep_patch_indices=rep_patch_indices,
                member_to_rep_local=member_to_rep_local,
                high_precision_local_mask=high_precision_local_mask,
                num_prefix=num_prefix,
                high_precision_bits=high_precision_bits,
                low_precision_bits=low_precision_bits,
            )

        if i in layer_set:
            output.append(x)

    assert len(output) == len(layer_indices), (
        f"Expected {len(layer_indices)} intermediate outputs, got {len(output)}"
    )

    result: list[tuple[Tensor, Tensor]] = []
    for out in output:
        out_norm = backbone.norm(out)
        cls_token = out_norm[:, 0]
        patch_tokens = out_norm[:, num_prefix:]
        result.append((patch_tokens, cls_token))

    return result


# -----------------------------------------------------------------------
# Two-Pass GAS: pruned tokens recover context via cross-attention
# -----------------------------------------------------------------------

def gas_block_forward_twopass(
    block: nn.Module,
    x: Tensor,
    keep_indices: Tensor,
    prune_indices: Tensor,
    num_prefix: int = 1,
) -> Tensor:
    """
    Two-pass Gather-Attend-Scatter for a single ViT block.

    Pass 1: kept tokens + prefix do full self-attention (identical to GAS).
    Pass 2: pruned tokens do cross-attention to kept tokens (Q=pruned, KV=kept).
             This restores global context for pruned tokens without N² cost.

    Total attention cost: N_keep² + N_prune × N_keep  (vs N² for dense).
    """
    B, L, D = x.shape

    # --- Indices ---
    prefix_idx = torch.arange(num_prefix, device=x.device)
    kept_global = keep_indices + num_prefix
    pruned_global = prune_indices + num_prefix
    gather_idx = torch.cat([prefix_idx, kept_global])  # (M,)

    # === Pass 1: self-attention on kept tokens (same as GAS) ===
    x_short = x[:, gather_idx, :]                            # (B, M, D)
    attn_out = block.ls1(block.attn(block.norm1(x_short)))   # (B, M, D)
    x_short = x_short + attn_out

    x_out = x.clone()
    x_out[:, gather_idx, :] = x_short

    # === Pass 2: cross-attention for pruned tokens (Q=pruned, KV=kept) ===
    if pruned_global.numel() > 0:
        attn_mod = block.attn
        num_heads = attn_mod.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5

        # Normalise inputs
        x_pruned_norm = block.norm1(x_out[:, pruned_global, :])  # (B, Np, D)
        x_kept_norm = block.norm1(x_out[:, gather_idx, :])       # (B, M, D)

        # QKV projections reusing pretrained weights
        qkv_p = attn_mod.qkv(x_pruned_norm)  # (B, Np, 3D)
        qkv_k = attn_mod.qkv(x_kept_norm)    # (B, M, 3D)

        Np = pruned_global.shape[0]
        M = gather_idx.shape[0]

        q_p = qkv_p[:, :, :D].reshape(B, Np, num_heads, head_dim).permute(0, 2, 1, 3)
        k_k = qkv_k[:, :, D:2*D].reshape(B, M, num_heads, head_dim).permute(0, 2, 1, 3)
        v_k = qkv_k[:, :, 2*D:].reshape(B, M, num_heads, head_dim).permute(0, 2, 1, 3)

        attn_weights = (q_p @ k_k.transpose(-2, -1)) * scale  # (B, H, Np, M)
        attn_weights = attn_weights.softmax(dim=-1)

        cross_out = (attn_weights @ v_k).permute(0, 2, 1, 3).reshape(B, Np, D)
        cross_out = attn_mod.proj(cross_out)

        x_out[:, pruned_global, :] = x_out[:, pruned_global, :] + block.ls1(cross_out)

    # === FFN on all tokens ===
    x_out = x_out + block.ls2(block.mlp(block.norm2(x_out)))
    return x_out


def gas_get_intermediate_layers_twopass(
    backbone: nn.Module,
    x_input: Tensor,
    layer_indices: list[int],
    keep_indices: Tensor,
    prune_indices: Tensor,
    prune_layer: int = 0,
) -> list[tuple[Tensor, Tensor]]:
    """
    Run DINOv2 backbone with two-pass GAS sparse attention.

    Blocks [0, prune_layer): dense.
    Blocks [prune_layer, end): two-pass GAS (kept self-attn + pruned cross-attn).
    """
    nr = getattr(backbone, "num_register_tokens", 0)
    num_prefix = 1 + nr

    x = backbone.prepare_tokens_with_masks(x_input)

    if getattr(backbone, "chunked_blocks", False):
        blocks = [b for chunk in backbone.blocks for b in chunk]
    else:
        blocks = list(backbone.blocks)

    layer_set = set(layer_indices)
    output = []

    for i, blk in enumerate(blocks):
        if i < prune_layer:
            x = blk(x)
        else:
            x = gas_block_forward_twopass(
                block=blk, x=x,
                keep_indices=keep_indices,
                prune_indices=prune_indices,
                num_prefix=num_prefix,
            )
        if i in layer_set:
            output.append(x)

    assert len(output) == len(layer_indices)

    result: list[tuple[Tensor, Tensor]] = []
    for out in output:
        out_norm = backbone.norm(out)
        cls_token = out_norm[:, 0]
        patch_tokens = out_norm[:, num_prefix:]
        result.append((patch_tokens, cls_token))

    return result
