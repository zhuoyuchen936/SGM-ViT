"""
core/token_reassembly.py
========================
Token Re-assembly module for SGM-ViT.

After token pruning (zeroing pruned spatial tokens at the input of a chosen
ViT block), the DPT decoder receives intermediate feature maps where pruned
positions carry near-zero activations.  Even though the remaining tokens
exchange context via self-attention, the spatially-zeroed positions in the
feature volume reduce DPT reconstruction quality.

This module implements *feature-level token re-assembly*: before the DPT
decoder reshapes patch-token sequences into 2-D spatial maps, pruned
positions are filled via Gaussian-weighted interpolation from their kept
neighbours.  This restores spatial completeness at every decoder input
level, improving depth quality without modifying any model weights.

Algorithm (per intermediate feature level)
------------------------------------------
  1.  Reshape  (B, N, D)  →  (B, D, H_t, W_t)     where H_t × W_t = N
  2.  Zero pruned positions (enforced explicitly; already ~0 from hook)
  3.  Gaussian-spread kept features:  convolve feature map with a 2-D
      isotropic Gaussian kernel  (channel-wise, efficiently via groupwise
      single-channel convolution on the (B*D, 1, H, W) flattened tensor)
  4.  Gaussian-spread kept mask:      same kernel on the binary kept-mask
  5.  Normalised interpolation at every grid cell:  spread_features / spread_mask
  6.  Replace ONLY pruned positions with the interpolated values
  7.  Reshape back  (B, D, H_t, W_t)  →  (B, N, D)

At kept positions the output is numerically identical to the input.
At pruned positions the output is a locally-weighted average of
neighbouring kept token features, weighted by Gaussian proximity.

Hardware context
----------------
On the FPGA the re-assembly stage is a lightweight bilinear interpolation
unit operating on the kept-token stream (already present for the DPT
feature-pyramid reads).  In the software demo it runs on the host GPU/CPU
immediately before ``model.depth_head()``.

Author : [Your Name]
Venue  : ICCAD 2025 (submission)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reassemble_token_features(
    features: list[tuple[torch.Tensor, torch.Tensor]],
    prune_mask_2d: torch.Tensor,
    patch_h: int,
    patch_w: int,
    sigma: float = 3.0,
    kernel_size: int = 9,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Fill pruned token positions in intermediate ViT feature maps via
    Gaussian-weighted interpolation from kept neighbour tokens.

    Parameters
    ----------
    features : list of (patch_tokens, class_token)
        Output of ``pretrained.get_intermediate_layers(..., return_class_token=True)``.
        ``patch_tokens`` : (B, N, D),  N = patch_h * patch_w
        ``class_token``  : (B, D)  — passed through unchanged
    prune_mask_2d : (patch_h, patch_w) bool Tensor
        True at positions that were pruned (zeroed) by the pruning hook.
        Automatically moved to the device of the feature tensors.
    patch_h, patch_w : int
        Spatial dimensions of the token grid (H_t, W_t).
    sigma : float
        Gaussian spread radius in tokens.  Larger values fill from more
        distant neighbours at the cost of spatial blurring.  Default: 3.0.
    kernel_size : int
        Convolution kernel size; must be odd.  Default: 9.

    Returns
    -------
    list of (patch_tokens_filled, class_token)
        Same structure as input.  class_token is unchanged.
        patch_tokens_filled has pruned positions interpolated from neighbours.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    device    = features[0][0].device
    prune_2d  = prune_mask_2d.bool().to(device)        # (H_t, W_t)
    kept_mask = (~prune_2d).float()                    # 1.0 = kept, 0.0 = pruned

    # ----------------------------------------------------------------
    # Build 2-D Gaussian kernel — computed once, reused for every level
    # ----------------------------------------------------------------
    x_range  = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    gauss_1d = torch.exp(-x_range ** 2 / (2.0 * sigma ** 2))
    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)  # (k, k)
    gauss_2d = gauss_2d / gauss_2d.sum()
    kernel   = gauss_2d.view(1, 1, kernel_size, kernel_size)   # (1, 1, k, k)
    padding  = kernel_size // 2

    # ----------------------------------------------------------------
    # Spread the kept-mask — identical for all feature levels and channels
    # ----------------------------------------------------------------
    kept_4d     = kept_mask.view(1, 1, patch_h, patch_w)             # (1, 1, H, W)
    spread_mask = F.conv2d(kept_4d, kernel, padding=padding)          # (1, 1, H, W)
    spread_mask = spread_mask.clamp(min=1e-8)

    # Boolean prune mask broadcastable over (B, D, H, W)
    prune_hw = prune_2d.view(1, 1, patch_h, patch_w)                  # (1, 1, H, W)

    reassembled: list[tuple[torch.Tensor, torch.Tensor]] = []

    for patch_tokens, class_token in features:
        B, N, D = patch_tokens.shape

        # ---- Reshape to spatial ------------------------------------------------
        feat_2d = patch_tokens.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)

        # ---- Zero pruned positions (explicit; should already be ~0 from hook) ---
        prune_bd = prune_hw.expand(B, D, -1, -1)
        feat_2d  = feat_2d.masked_fill(prune_bd, 0.0)

        # ---- Gaussian-spread kept features (all D channels, one conv pass) ------
        # Flatten to (B*D, 1, H, W) for efficient single-channel convolution
        feat_flat   = feat_2d.reshape(B * D, 1, patch_h, patch_w)
        spread_feat = F.conv2d(feat_flat, kernel, padding=padding)    # (B*D, 1, H, W)
        spread_feat = spread_feat.reshape(B, D, patch_h, patch_w)

        # ---- Normalised interpolation at every spatial position ----------------
        interp = spread_feat / spread_mask                             # (B, D, H, W)

        # ---- Replace ONLY pruned positions with interpolated values -------------
        feat_filled = torch.where(prune_bd, interp, feat_2d)          # (B, D, H, W)

        # ---- Reshape back to sequence ------------------------------------------
        patch_tokens_out = feat_filled.reshape(B, D, N).permute(0, 2, 1)  # (B, N, D)

        reassembled.append((patch_tokens_out, class_token))

    return reassembled
