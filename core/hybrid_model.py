"""
core/hybrid_model.py
====================
SGMViTHybridModel — Assembly class that fuses the SGM-guided confidence router
with a sparse DepthAnythingV2 ViT backbone for edge-efficient depth estimation.

Architecture Overview
---------------------

    ┌──────────────────────────────────────────────────────────────┐
    │                     SGMViTHybridModel                        │
    │                                                              │
    │  Input: RGB image (B, 3, H, W)                               │
    │                                                              │
    │  ┌─────────────────────┐    ┌──────────────────────────────┐ │
    │  │   SGM Engine (CPU/  │    │  DepthAnythingV2 Patch Embed │ │
    │  │   FPGA HW block)    │    │  tokens (B, N, D)            │ │
    │  │                     │    └────────────┬─────────────────┘ │
    │  │  → Disparity Map    │                 │                   │
    │  │  → Confidence Map   │─────────────────▼                   │
    │  └─────────────────────┘    SGMConfidenceTokenRouter         │
    │                              ┌──────────┴──────────┐         │
    │                         keep_tokens           prune_idx      │
    │                              │                     │          │
    │                     Sparse ViT Attention           │          │
    │                     (N_keep tokens only)           │          │
    │                              │                     │          │
    │                     ┌────────▼─────────────────────▼──────┐  │
    │                     │     Token Re-assembly & Fusion       │  │
    │                     │  (keep attn output + SGM fill-in)    │  │
    │                     └────────────────────┬─────────────────┘  │
    │                                          │                    │
    │                               DPT Decoder Head                │
    │                                          │                    │
    │                               Depth Map (B, 1, H, W)         │
    └──────────────────────────────────────────────────────────────┘

Design Philosophy
-----------------
The model is intentionally split into clearly delineated software and hardware
stages so that the FPGA hardware boundary can be cleanly inserted between
``_sgm_forward`` (HW block) and ``_vit_forward`` (SW or accelerated block).

For the ICCAD submission, the key novelty lies in:
  1. The confidence-guided token gate (``token_router``).
  2. The token re-assembly strategy after sparse attention.
  3. The FLOPs-vs-accuracy trade-off characterisation across pruning ratios.

Author : [Your Name]
Venue  : ICCAD 2025 (submission)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_router import SGMConfidenceTokenRouter


class SGMViTHybridModel(nn.Module):
    """
    Hybrid depth estimation model combining SGM priors with sparse ViT.

    This class acts as a scaffold / assembly harness.  The actual
    DepthAnythingV2 backbone is injected at construction time so that the
    router and fusion logic remain independent of the backbone version.

    Parameters
    ----------
    vit_backbone : nn.Module
        A DepthAnythingV2 (or compatible) ViT encoder that exposes:
          * ``patch_embed(x)  → tokens (B, N, D)``
          * ``blocks          → nn.ModuleList of transformer blocks``
          * ``norm            → final layer norm``
        See ``depth_anything_v2/`` for the source.
    dpt_head : nn.Module
        Dense Prediction Transformer decoder head that maps ViT features to
        a dense depth map.
    token_grid_size : int
        ViT token grid side length (e.g. 14 for ViT-B with 16-px patches on
        224-px input).
    confidence_threshold : float
        SGM confidence pruning threshold passed to ``SGMConfidenceTokenRouter``.
    sgm_fill_weight : float
        When re-assembling tokens, pruned positions are filled with a
        weighted blend:
          ``token_out[prune] = sgm_fill_weight * sgm_prior[prune]
                             + (1 - sgm_fill_weight) * learned_default``
        Set to 1.0 to use raw SGM prior; 0.0 for a learned constant.
    """

    def __init__(
        self,
        vit_backbone: nn.Module,
        dpt_head: nn.Module,
        token_grid_size: int = 14,
        confidence_threshold: float = 0.6,
        sgm_fill_weight: float = 0.8,
    ) -> None:
        super().__init__()

        self.backbone = vit_backbone
        self.dpt_head = dpt_head

        self.token_router = SGMConfidenceTokenRouter(
            token_grid_size=token_grid_size,
            confidence_threshold=confidence_threshold,
            learnable_threshold=False,
        )

        # Embedding dimension inferred from backbone (populated on first forward)
        self._embed_dim: Optional[int] = None
        self.sgm_fill_weight = sgm_fill_weight

        # Learned default token for pruned positions (initialised to zero;
        # trained to represent the "average" high-confidence token).
        # Actual dimension is set lazily in _init_fill_token().
        self._fill_token: Optional[nn.Parameter] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        confidence_map: torch.Tensor,
        sgm_depth_prior: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        image : (B, 3, H, W)
            Normalised RGB input image.
        confidence_map : (B, 1, H, W) or (B, H, W)
            Per-pixel SGM confidence in [0, 1].
        sgm_depth_prior : (B, 1, H, W), optional
            SGM disparity/depth map.  Used to fill pruned token positions
            when ``sgm_fill_weight > 0``.

        Returns
        -------
        dict with:
            "depth"       : (B, 1, H, W) — predicted depth map.
            "prune_ratio" : float        — fraction of pruned tokens.
            "conf_grid"   : (B, G, G)   — token-grid confidence map.
        """
        B = image.shape[0]

        # ------------------------------------------------------------------
        # Stage 1: Patch embedding (shared with standard DepthAnythingV2)
        # ------------------------------------------------------------------
        tokens = self.backbone.patch_embed(image)  # (B, N, D)
        D = tokens.shape[-1]
        self._lazy_init_fill_token(D, device=tokens.device)

        # ------------------------------------------------------------------
        # Stage 2: SGM confidence-guided token routing
        # ------------------------------------------------------------------
        routing = self.token_router(confidence_map, tokens)
        keep_idx   = routing["keep_idx"]    # list[LongTensor]
        prune_idx  = routing["prune_idx"]   # list[LongTensor]
        keep_tokens = routing["keep_tokens"] # list[FloatTensor (N_keep, D)]

        # ------------------------------------------------------------------
        # Stage 3: Sparse attention — process only "keep" tokens
        # ------------------------------------------------------------------
        # NOTE: For simplicity in the PoC, we concatenate into a dense batch
        # after padding.  In the hardware implementation, tokens are streamed
        # without padding (variable-length FIFO per sample).
        attended_tokens = self._sparse_attention(keep_tokens, keep_idx, tokens)
        # attended_tokens : (B, N, D) — kept positions updated, pruned = fill

        # ------------------------------------------------------------------
        # Stage 4: Transformer tail (remaining backbone blocks after routing)
        # ------------------------------------------------------------------
        # In a full integration, the routing is inserted *between* blocks.
        # For the skeleton we apply all remaining blocks to the reassembled
        # token sequence (demonstrating the interface, not full efficiency).
        x = attended_tokens
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)  # (B, N, D)

        # ------------------------------------------------------------------
        # Stage 5: DPT decode head → depth map
        # ------------------------------------------------------------------
        depth = self.dpt_head(x)  # (B, 1, H, W)

        return {
            "depth":       depth,
            "prune_ratio": routing["prune_ratio"],
            "conf_grid":   routing["conf_grid"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lazy_init_fill_token(self, D: int, device: torch.device) -> None:
        """Lazily initialise the learnable fill token once D is known."""
        if self._fill_token is None:
            self._fill_token = nn.Parameter(
                torch.zeros(D, device=device)
            )
            self._embed_dim = D

    def _sparse_attention(
        self,
        keep_tokens: list[torch.Tensor],
        keep_idx: list[torch.Tensor],
        full_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the first transformer block on kept tokens only, then re-assemble.

        In the full model this will be a loop over all encoder blocks with
        the router deciding (possibly per-block) which tokens to keep.

        Parameters
        ----------
        keep_tokens : list of (N_keep_b, D)
        keep_idx    : list of LongTensor
        full_tokens : (B, N, D)  — original unprocessed tokens (used as base
                      for re-assembly so that pruned positions can be filled).

        Returns
        -------
        out : (B, N, D)
        """
        B, N, D = full_tokens.shape
        out = full_tokens.clone()

        for b in range(B):
            k_idx = keep_idx[b]

            if k_idx.numel() == 0:
                # All tokens pruned (degenerate case): keep full_tokens as-is.
                continue

            # Forward kept tokens through the first backbone block (PoC).
            # Shape: (1, N_keep, D)  — single-sample "batch" for the block.
            kept = keep_tokens[b].unsqueeze(0)
            attended = self.backbone.blocks[0](kept).squeeze(0)  # (N_keep, D)

            # Write attended features back to their original positions.
            out[b, k_idx] = attended

            # Fill pruned positions with the learnable default token.
            # In the hardware design, pruned positions are filled directly
            # from the SGM depth prior (converted to a feature embedding).
            prune_positions = torch.ones(N, dtype=torch.bool, device=out.device)
            prune_positions[k_idx] = False
            prune_idx_b = torch.where(prune_positions)[0]

            if prune_idx_b.numel() > 0 and self._fill_token is not None:
                out[b, prune_idx_b] = self._fill_token.unsqueeze(0).expand(
                    prune_idx_b.numel(), -1
                )

        return out  # (B, N, D)
