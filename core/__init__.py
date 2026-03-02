"""
core/ — SGM-ViT core algorithm modules.

Modules
-------
token_router   : Hardware-aware SGM confidence-guided token routing / pruning.
hybrid_model   : Assembly class that fuses the SGM-guided router with the
                 sparse DepthAnythingV2 ViT backbone.
sgm_wrapper    : Programmatic SGM wrapper that returns NumPy arrays
                 (disparity map + confidence map) instead of writing to disk.
"""

from .token_router import SGMConfidenceTokenRouter
from .hybrid_model import SGMViTHybridModel
from .sgm_wrapper  import run_sgm_with_confidence, confidence_to_token_grid

__all__ = [
    "SGMConfidenceTokenRouter",
    "SGMViTHybridModel",
    "run_sgm_with_confidence",
    "confidence_to_token_grid",
]
