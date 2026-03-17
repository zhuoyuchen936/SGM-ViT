"""
core/ — SGM-ViT core algorithm modules.

Modules
-------
token_router        : Hardware-aware SGM confidence-guided token routing / pruning.
token_reassembly    : Feature-level token re-assembly before the DPT decoder —
                      fills pruned positions via Gaussian-weighted interpolation
                      from kept neighbour tokens.
sparse_attention    : Gather-Attend-Scatter (GAS) sparse attention for DINOv2 ViT
                      blocks — physically excludes pruned tokens from attention.
hybrid_model        : Assembly class that fuses the SGM-guided router with the
                      sparse DepthAnythingV2 ViT backbone.
sgm_wrapper         : Programmatic SGM wrapper that returns NumPy arrays
                      (disparity map + confidence map) instead of writing to disk.
pruning_strategies  : Alternative pruning mask generators for ablation studies
                      (random, top-K, inverse, checkerboard, CLS attention, hybrid).
eval_utils          : Shared evaluation utilities (FLOPs reduction, Pareto frontier,
                      confidence pooling, token grid geometry).
"""

from .token_router       import SGMConfidenceTokenRouter
from .token_reassembly   import reassemble_token_features
from .hybrid_model       import SGMViTHybridModel
from .sgm_wrapper        import run_sgm_with_confidence, confidence_to_token_grid
from .sparse_attention   import gas_block_forward, gas_get_intermediate_layers
from .pruning_strategies import (
    random_prune_mask, topk_confidence_mask, inverse_confidence_mask,
    spatial_checkerboard_mask, cls_attention_mask, hybrid_mask,
)
from .eval_utils import (
    compute_attn_reduction, pareto_frontier,
    pool_confidence, compute_token_grid_size,
)

__all__ = [
    "SGMConfidenceTokenRouter",
    "reassemble_token_features",
    "SGMViTHybridModel",
    "run_sgm_with_confidence",
    "confidence_to_token_grid",
    "gas_block_forward",
    "gas_get_intermediate_layers",
    "random_prune_mask",
    "topk_confidence_mask",
    "inverse_confidence_mask",
    "spatial_checkerboard_mask",
    "cls_attention_mask",
    "hybrid_mask",
    "compute_attn_reduction",
    "pareto_frontier",
    "pool_confidence",
    "compute_token_grid_size",
]
