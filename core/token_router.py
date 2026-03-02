"""
core/token_router.py
====================
Hardware-aware, SGM confidence-guided token routing for sparse DepthAnythingV2.

Motivation
----------
Semi-Global Matching (SGM) produces a dense disparity map together with a
per-pixel **confidence score** that reflects the reliability of each
disparity estimate.  Regions with *high* SGM confidence (e.g., textured
surfaces with clear stereo correspondence) already have a good geometric
prior — they do NOT need the full, expensive multi-head self-attention of the
ViT backbone to refine their depth.

This module exploits that insight at the token level:

    High-confidence token  →  **Prune** (skip attention, reuse SGM prior)
    Low-confidence  token  →  **Keep**  (forward through ViT attention layers)

By selectively skipping ~40-60 % of tokens in well-conditioned regions, we
achieve a proportional reduction in attention FLOPs — the dominant cost of
DepthAnythingV2 on edge hardware.

This design maps naturally to an FPGA hardware accelerator (see hw/):
  * The SGM engine runs on dedicated DSP/BRAM slices.
  * The confidence threshold comparison is a single LUT operation.
  * Only the "keep" token stream is dispatched to the sparse attention PEs.

Reference pipeline
------------------
    RGB image
        │
        ├─► SGM Engine ──► Disparity Map + Confidence Map (C, H, W)
        │                          │
        │              SGMConfidenceTokenRouter
        │                    ┌─────┴─────┐
        │                keep_idx     prune_idx
        │                    │             │
        └─► ViT Patch Embed  │             │
                (B, N, D)    │             │
                    │        │             │
                Sparse Attn ◄┘             │
                    │                      │
                Dense Fusion ◄─────────────┘
                    │
                Depth Output

Author : [Your Name]
Venue  : ICCAD 2025 (submission)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGMConfidenceTokenRouter(nn.Module):
    """
    Routes ViT tokens based on SGM per-pixel confidence scores.

    Given a confidence map produced by the SGM algorithm and a ViT token
    tensor, this module partitions tokens into two disjoint sets:

      * **keep_idx**  — tokens in low-confidence regions that benefit from
                        full ViT attention (geometry uncertain, need semantics).
      * **prune_idx** — tokens in high-confidence regions where SGM already
                        provides a reliable depth prior; attention is skipped.

    Parameters
    ----------
    token_grid_size : int
        Side length of the square ViT token grid (e.g. 14 for ViT-B/16 on
        224×224 input, or 16 for 256×256 input).  The confidence map will be
        adaptive-average-pooled to (token_grid_size × token_grid_size).
    confidence_threshold : float
        Normalised confidence value in [0, 1].  Tokens whose pooled confidence
        exceeds this threshold are classified as "high-confidence" and pruned.
        Recommended starting point: 0.6.  Should be treated as a tunable
        hardware parameter (maps to a fixed-point comparison on FPGA).
    learnable_threshold : bool
        If True, the threshold is a learnable scalar initialised from
        ``confidence_threshold``.  Useful for end-to-end fine-tuning.
        Set to False for pure inference / FPGA deployment.
    """

    def __init__(
        self,
        token_grid_size: int = 14,
        confidence_threshold: float = 0.6,
        learnable_threshold: bool = False,
    ) -> None:
        super().__init__()

        self.token_grid_size = token_grid_size
        self.N = token_grid_size * token_grid_size  # total number of spatial tokens

        if learnable_threshold:
            # Expose as a trainable parameter so the network can adapt the
            # pruning aggressiveness end-to-end.
            self.threshold = nn.Parameter(
                torch.tensor(confidence_threshold, dtype=torch.float32)
            )
        else:
            # Fixed threshold: zero-overhead on FPGA (single comparator).
            self.register_buffer(
                "threshold",
                torch.tensor(confidence_threshold, dtype=torch.float32),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        confidence_map: torch.Tensor,
        tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Partition tokens into keep / prune sets.

        Parameters
        ----------
        confidence_map : torch.Tensor, shape (B, 1, H, W)  or  (B, H, W)
            Per-pixel SGM confidence scores in [0, 1].  A value of 1.0 means
            SGM is fully confident about the disparity at that pixel.
        tokens : torch.Tensor, shape (B, N, D)
            Flattened ViT patch tokens **before** any attention layer.
            N = token_grid_size², D = embedding dimension.

        Returns
        -------
        dict with keys:
            "keep_idx"   : LongTensor (B, N_keep)   — indices of kept tokens.
            "prune_idx"  : LongTensor (B, N_prune)  — indices of pruned tokens.
            "keep_tokens": FloatTensor (B, N_keep, D) — kept token features.
            "conf_grid"  : FloatTensor (B, G, G)    — pooled confidence map
                           aligned to the token grid (G = token_grid_size).
            "prune_ratio": float — fraction of tokens pruned (for monitoring).
        """
        B, N, D = tokens.shape
        assert N == self.N, (
            f"Expected {self.N} tokens (grid {self.token_grid_size}²), "
            f"got {N}."
        )

        # ------------------------------------------------------------------
        # Step 1: Align the SGM confidence map to the ViT token grid.
        # ------------------------------------------------------------------
        # The confidence map is at the original image resolution (H×W).
        # We downsample it to match the ViT patch-grid resolution via
        # adaptive average pooling — identical to how patch embeddings
        # aggregate pixel values.  On FPGA this is a sliding-window averager.
        conf = confidence_map
        if conf.dim() == 3:
            conf = conf.unsqueeze(1)  # (B, 1, H, W)

        # Normalise to [0, 1] if not already (SGM may output raw matching cost)
        conf = conf.clamp(0.0, 1.0)

        conf_grid = F.adaptive_avg_pool2d(
            conf, output_size=(self.token_grid_size, self.token_grid_size)
        )  # (B, 1, G, G)
        conf_grid = conf_grid.squeeze(1)  # (B, G, G)

        # ------------------------------------------------------------------
        # Step 2: Flatten the token-grid confidence to a 1-D score per token.
        # ------------------------------------------------------------------
        # Each score conf_flat[:, i] corresponds to token tokens[:, i, :].
        # The spatial ordering must match the ViT patch embedding scan order
        # (row-major / raster scan for standard ViT implementations).
        conf_flat = conf_grid.reshape(B, N)  # (B, N)

        # ------------------------------------------------------------------
        # Step 3: Threshold-based token routing.
        # ------------------------------------------------------------------
        # HIGH confidence  →  SGM prior is reliable  →  PRUNE  (skip attn)
        # LOW  confidence  →  need ViT semantics      →  KEEP   (run  attn)
        #
        # Hardware note: on FPGA this is a single fixed-point comparator
        # operating at the token dispatch rate (one decision per clock cycle).
        thr = self.threshold.clamp(0.0, 1.0)  # guard against gradient drift

        high_conf_mask = conf_flat > thr   # (B, N) bool
        low_conf_mask  = ~high_conf_mask   # (B, N) bool

        # ------------------------------------------------------------------
        # Step 4: Gather indices and features for the "keep" set.
        # ------------------------------------------------------------------
        # NOTE: The number of kept / pruned tokens varies per sample in the
        # batch, so we return index tensors rather than a packed dense tensor.
        # For batched training, consider padding or using a fixed keep-ratio.
        keep_idx_list  = []
        prune_idx_list = []

        for b in range(B):
            keep_idx_list.append(torch.where(low_conf_mask[b])[0])   # (N_keep,)
            prune_idx_list.append(torch.where(high_conf_mask[b])[0]) # (N_prune,)

        # For convenience, also return the gathered token features of the
        # "keep" set (they are immediately fed into the sparse attention).
        keep_tokens_list = [
            tokens[b][keep_idx_list[b]]  # (N_keep_b, D)
            for b in range(B)
        ]

        # Compute the overall prune ratio for FLOPs bookkeeping.
        total_pruned = sum(idx.numel() for idx in prune_idx_list)
        prune_ratio  = total_pruned / (B * N)

        return {
            "keep_idx":    keep_idx_list,    # list[LongTensor], length B
            "prune_idx":   prune_idx_list,   # list[LongTensor], length B
            "keep_tokens": keep_tokens_list, # list[FloatTensor (N_keep, D)]
            "conf_grid":   conf_grid,        # (B, G, G)  — for visualisation
            "prune_ratio": prune_ratio,      # float       — for FLOPs logging
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visualise_routing(self, conf_grid: torch.Tensor) -> torch.Tensor:
        """
        Return a binary mask on the token grid for quick inspection.

        Parameters
        ----------
        conf_grid : (B, G, G)

        Returns
        -------
        mask : (B, G, G) uint8 — 1 = keep, 0 = pruned
        """
        thr = self.threshold.clamp(0.0, 1.0)
        return (conf_grid <= thr).to(torch.uint8)

    def extra_repr(self) -> str:
        return (
            f"token_grid_size={self.token_grid_size}, "
            f"N_tokens={self.N}, "
            f"threshold={self.threshold.item():.3f}, "
            f"learnable_threshold={isinstance(self.threshold, nn.Parameter)}"
        )
