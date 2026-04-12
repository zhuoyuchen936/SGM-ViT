"""
Shared import path setup for SGM-ViT.

Ensures the project root and the bundled Depth-Anything-V2 checkout are on
``sys.path`` before importing model code.
"""
from __future__ import annotations

import os
import sys


_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DA2_DIR = os.path.join(_PROJECT_ROOT, "Depth-Anything-V2")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if _DA2_DIR not in sys.path:
    sys.path.insert(0, _DA2_DIR)
