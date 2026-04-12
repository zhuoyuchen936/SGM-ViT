"""SGM-ViT core package.

This package intentionally keeps ``__init__`` lightweight so that importing a
single submodule does not immediately pull in heavy runtime dependencies such
as DA2 checkpoints or the SGM engine. Import concrete helpers from their
submodules, for example ``core.fusion_net`` or ``core.pipeline``.
"""

__all__: list[str] = []
