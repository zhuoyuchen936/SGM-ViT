"""Shared experiment defaults for SGM-ViT scripts."""
from __future__ import annotations

import os


PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_KITTI_ROOT = os.environ.get(
    "SGMVIT_KITTI_ROOT",
    "/nfs/usrhome/pdongaa/dataczy/kitti",
)
DEFAULT_ETH3D_ROOT = os.environ.get(
    "SGMVIT_ETH3D_ROOT",
    "/nfs/usrhome/pdongaa/dataczy/eth3d",
)
DEFAULT_SCENEFLOW_ROOT = os.environ.get(
    "SGMVIT_SCENEFLOW_ROOT",
    "/nfs/usrhome/pdongaa/dataczy/sceneflow_official/extracted/driving",
)
DEFAULT_MIDDLEBURY_ROOT = os.environ.get(
    "SGMVIT_MIDDLEBURY_ROOT",
    "/nfs/usrhome/pdongaa/dataczy/Middelburry",
)
DEFAULT_WEIGHTS = os.environ.get(
    "SGMVIT_DA2_WEIGHTS",
    os.path.join(
        PROJECT_DIR,
        "Depth-Anything-V2",
        "checkpoints",
        "depth_anything_v2_vits.pth",
    ),
)
DEFAULT_ENCODER = os.environ.get("SGMVIT_ENCODER", "vits")

DEFAULT_INPUT_SIZE = 518
DEFAULT_PRUNE_THRESHOLD = float(os.environ.get("SGMVIT_PRUNE_THRESHOLD", "0.65"))
DEFAULT_PRUNE_LAYER = int(os.environ.get("SGMVIT_PRUNE_LAYER", "0"))
DEFAULT_ALIGN_CONF_THRESHOLD = float(os.environ.get("SGMVIT_ALIGN_CONF_THRESHOLD", "0.70"))
DEFAULT_FUSION_CONF_THRESHOLD = float(os.environ.get("SGMVIT_FUSION_CONF_THRESHOLD", "0.55"))
DEFAULT_CONF_SIGMA = float(os.environ.get("SGMVIT_CONF_SIGMA", "5.0"))
DEFAULT_DISPARITY_RANGE = int(os.environ.get("SGMVIT_DISPARITY_RANGE", "128"))
DEFAULT_PKRN_MIN_DIST = int(os.environ.get("SGMVIT_PKRN_MIN_DIST", "1"))
DEFAULT_MAX_SAMPLES = int(os.environ.get("SGMVIT_MAX_SAMPLES", "0"))

DEFAULT_FUSION_STRATEGY = os.environ.get("SGMVIT_FUSION_STRATEGY", "edge_aware_residual")
DEFAULT_FUSION_BACKEND = os.environ.get("SGMVIT_FUSION_BACKEND", "heuristic")
DEFAULT_FUSION_NET_WEIGHTS = os.environ.get(
    "SGMVIT_FUSION_NET_WEIGHTS",
    os.path.join(PROJECT_DIR, "artifacts", "fusion_net", "kitti_finetune", "best.pt"),
)
DEFAULT_EDGE_THETA_LOW = float(os.environ.get("SGMVIT_EDGE_THETA_LOW", "0.10"))
DEFAULT_EDGE_THETA_HIGH = float(os.environ.get("SGMVIT_EDGE_THETA_HIGH", "0.65"))
DEFAULT_EDGE_DETAIL_SUPPRESSION = float(os.environ.get("SGMVIT_EDGE_DETAIL_SUPPRESSION", "0.75"))
DEFAULT_EDGE_RESIDUAL_GAIN = float(os.environ.get("SGMVIT_EDGE_RESIDUAL_GAIN", "0.90"))

ALIGNMENT_CONFIDENCE_SOURCE = "lrcheck"
ROUTING_CONFIDENCE_SOURCE = "pkrn"
FUSION_CONFIDENCE_SOURCE = "pkrn"


def default_results_dir(name: str) -> str:
    return os.path.join(PROJECT_DIR, "results", name)
