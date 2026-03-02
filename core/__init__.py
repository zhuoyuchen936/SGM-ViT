"""
core/ — SGM-ViT core algorithm modules.

Modules
-------
token_router   : Hardware-aware SGM confidence-guided token routing / pruning.
hybrid_model   : Assembly class that fuses the SGM-guided router with the
                 sparse DepthAnythingV2 ViT backbone.
"""

from .token_router import SGMConfidenceTokenRouter
from .hybrid_model import SGMViTHybridModel

__all__ = ["SGMConfidenceTokenRouter", "SGMViTHybridModel"]
