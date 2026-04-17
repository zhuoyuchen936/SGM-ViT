"""
core/stereo_datasets.py — Dataset classes for SGM-ViT training / fine-tuning.

Provides PyTorch Dataset wrappers for stereo-matching datasets that include
pre-computed SGM disparities (sgm_hole) and mismatch/occlusion masks.

Supported datasets
------------------
ETH3DStereoDataset
    Root: /nfs/usrhome/pdongaa/dataczy/eth3d
    27 scenes, sparse GT (lidar), sgm_hole pre-computed.

SceneFlowDrivingDataset
    Root: /nfs/usrhome/pdongaa/dataczy/sceneflow_official/extracted/driving
    300 frames (Driving / 35mm_focallength / scene_forwards / fast).
    Dense synthetic GT, sgm_hole pre-computed.

MiddleburyDataset
    Root: /nfs/usrhome/pdongaa/dataczy/Middelburry   (note spelling)
    MiddEval3-Q resolution only.  sgm_hole NOT available — raises
    FileNotFoundError on construction until SGM pre-computation is run.

KITTIStereoDataset
    Root: /nfs/usrhome/pdongaa/dataczy/kitti
    KITTI-2015 + KITTI-2012 combined, sparse GT, sgm_hole pre-computed.

Usage example
-------------
>>> from core.stereo_datasets import ETH3DStereoDataset
>>> ds = ETH3DStereoDataset()
>>> sample = ds[0]
>>> sample.keys()
dict_keys(['left', 'right', 'gt_disp', 'valid', 'sgm_disp', 'mismatch', 'occlusion', 'scene'])
"""
from __future__ import annotations

import os
import struct
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Default roots — mirrors scripts/common_config.py but importable from core/
# without pulling in scripts dependencies.
# ---------------------------------------------------------------------------
_NFS = "/nfs/usrhome/pdongaa/dataczy"
_DEFAULT_ETH3D_ROOT       = os.environ.get("SGMVIT_ETH3D_ROOT",       f"{_NFS}/eth3d")
_DEFAULT_SCENEFLOW_ROOT   = os.environ.get("SGMVIT_SCENEFLOW_ROOT",   f"{_NFS}/sceneflow_official/extracted/driving")
_DEFAULT_MIDDLEBURY_ROOT  = os.environ.get("SGMVIT_MIDDLEBURY_ROOT",  f"{_NFS}/Middelburry")
_DEFAULT_KITTI_ROOT       = os.environ.get("SGMVIT_KITTI_ROOT",       f"{_NFS}/kitti")


# ---------------------------------------------------------------------------
# PFM reader
# ---------------------------------------------------------------------------

def _read_pfm(path: str) -> np.ndarray:
    """Read a .pfm disparity/depth file → float32 (H, W)."""
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        assert header in (b"PF", b"Pf"), f"Not a PFM file: {path}"
        w, h = map(int, f.readline().split())
        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), endian + "f").reshape(h, w)
    return np.ascontiguousarray(np.flipud(data)).astype(np.float32)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class StereoSGMDataset(Dataset):
    """Base dataset: list of sample dicts loaded lazily.

    Each sample dict returned by __getitem__ contains:
        left        : np.uint8 (H, W, 3)  left RGB image
        right       : np.uint8 (H, W, 3)  right RGB image
        gt_disp     : np.float32 (H, W)   ground-truth disparity (0 = invalid)
        valid       : np.bool_ (H, W)     GT valid mask
        sgm_disp    : np.float32 (H, W)   pre-computed SGM disparity
        mismatch    : np.bool_ (H, W)     mismatch mask (True = bad pixel)
        occlusion   : np.bool_ (H, W)     occlusion mask (True = occluded)
        scene       : str                  dataset-specific identifier
    """

    def __init__(self) -> None:
        self._samples: List[Dict] = []

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict:
        s = self._samples[idx]

        left  = self._load_image(s["left"])
        right = self._load_image(s["right"])

        gt_raw = self._read_gt(s)
        gt_disp, valid = self._make_valid(gt_raw)

        sgm_disp  = _read_pfm(s["sgm"])
        mismatch  = np.load(s["mismatch"]).astype(bool)
        occlusion = np.load(s["occlusion"]).astype(bool)

        return {
            "left":      left,
            "right":     right,
            "gt_disp":   gt_disp,
            "valid":     valid,
            "sgm_disp":  sgm_disp,
            "mismatch":  mismatch,
            "occlusion": occlusion,
            "scene":     s.get("scene", ""),
        }

    # ---- Subclasses implement these ------------------------------------

    def _read_gt(self, s: Dict) -> np.ndarray:
        raise NotImplementedError

    def _make_valid(self, gt_raw: np.ndarray):
        """Return (gt_disp_float32, valid_bool_mask)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ETH3D
# ---------------------------------------------------------------------------

class ETH3DStereoDataset(StereoSGMDataset):
    """ETH3D two-view stereo dataset (training split, 27 scenes).

    Directory layout expected:
        root/
          <scene>/
            im0.png          left image
            im1.png          right image
            disp0GT.pfm      sparse GT disparity (0 = invalid)
          training/
            sgm_hole/
              <scene>/
                <scene>.pfm                    SGM disparity
                <scene>_mismatches.npy
                <scene>_occlusion.npy

    Note: training/sgm_hole/ contains a stray top-level im0.pfm that is
    filtered out here using scene-name matching.
    """

    def __init__(self, root: str = _DEFAULT_ETH3D_ROOT) -> None:
        super().__init__()
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"ETH3D root not found: {root}\n"
                f"Set SGMVIT_ETH3D_ROOT or pass root= explicitly."
            )

        sgm_base = os.path.join(root, "training", "sgm_hole")
        if not os.path.isdir(sgm_base):
            raise FileNotFoundError(
                f"ETH3D sgm_hole directory not found: {sgm_base}\n"
                f"Expected pre-computed SGM disparities at {sgm_base}/<scene>/<scene>.pfm"
            )

        # Use scene-name matching to avoid misalignment with the stray im0.pfm.
        image_paths = sorted(glob(os.path.join(root, "*/im0.png")))
        loaded = 0
        skipped = []
        for img0 in image_paths:
            scene = Path(img0).parent.name
            # Skip if this "scene" is actually the training/ subdirectory.
            if scene == "training":
                continue
            sgm_dir = os.path.join(sgm_base, scene)
            sgm_pfm = os.path.join(sgm_dir, f"{scene}.pfm")
            mis_npy = os.path.join(sgm_dir, f"{scene}_mismatches.npy")
            occ_npy = os.path.join(sgm_dir, f"{scene}_occlusion.npy")

            img1 = os.path.join(Path(img0).parent, "im1.png")
            gt   = os.path.join(Path(img0).parent, "disp0GT.pfm")

            missing = [p for p in (img1, gt, sgm_pfm, mis_npy, occ_npy)
                       if not os.path.isfile(p)]
            if missing:
                skipped.append((scene, missing))
                continue

            self._samples.append({
                "left":     str(img0),
                "right":    str(img1),
                "gt":       str(gt),
                "sgm":      sgm_pfm,
                "mismatch": mis_npy,
                "occlusion":occ_npy,
                "scene":    scene,
            })
            loaded += 1

        if skipped:
            import warnings
            for sc, missing in skipped:
                warnings.warn(
                    f"ETH3D scene '{sc}' skipped — missing files: {missing}"
                )

        if loaded == 0:
            raise RuntimeError(
                f"No valid ETH3D samples found under {root}. "
                f"Check dataset structure."
            )

    def _read_gt(self, s: Dict) -> np.ndarray:
        return _read_pfm(s["gt"])

    def _make_valid(self, gt_raw: np.ndarray):
        valid = (gt_raw > 0) & np.isfinite(gt_raw)
        gt    = np.where(valid, gt_raw, 0.0).astype(np.float32)
        return gt, valid


# ---------------------------------------------------------------------------
# SceneFlow Driving subset
# ---------------------------------------------------------------------------

class SceneFlowDrivingDataset(StereoSGMDataset):
    """SceneFlow Driving subset (35mm_focallength / scene_forwards / fast).

    Only the 300 frames from the "fast" speed sequence are loaded, matching
    the subset used in rSGM_Mamba's ``SceneFlowDatasets._add_driving()``.

    Directory layout:
        root/
          frames_cleanpass/
            35mm_focallength/scene_forwards/fast/
              left/*.png
              right/*.png
          disparity/
            35mm_focallength/scene_forwards/fast/
              left/*.pfm      (GT, dense synthetic)
          sgm_hole/
            35mm_focallength/scene_forwards/fast/
              left/
                35mm_focallength_scene_forwards_fast_left_<frame>.pfm
                35mm_focallength_scene_forwards_fast_left_<frame>_mismatches.npy
                35mm_focallength_scene_forwards_fast_left_<frame>_occlusion.npy
    """

    _FOCAL  = "35mm_focallength"
    _SCENE  = "scene_forwards"
    _SPEED  = "fast"

    def __init__(self, root: str = _DEFAULT_SCENEFLOW_ROOT) -> None:
        super().__init__()
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"SceneFlow root not found: {root}\n"
                f"Set SGMVIT_SCENEFLOW_ROOT or pass root= explicitly."
            )

        subpath = os.path.join(self._FOCAL, self._SCENE, self._SPEED)
        right_glob = os.path.join(
            root, "frames_cleanpass", subpath, "right", "*.png"
        )
        right_images = sorted(glob(right_glob))
        if not right_images:
            raise FileNotFoundError(
                f"No SceneFlow Driving frames found at: {right_glob}"
            )

        sgm_dir = os.path.join(root, "sgm_hole", subpath, "left")
        loaded, skipped = 0, []

        for right_img in right_images:
            frame_num = Path(right_img).stem  # e.g. "0001"
            left_img  = right_img.replace("right", "left")
            gt_pfm    = os.path.join(
                root, "disparity", subpath, "left", f"{frame_num}.pfm"
            )
            basename = f"{self._FOCAL}_{self._SCENE}_{self._SPEED}_left_{frame_num}"
            sgm_pfm  = os.path.join(sgm_dir, f"{basename}.pfm")
            mis_npy  = os.path.join(sgm_dir, f"{basename}_mismatches.npy")
            occ_npy  = os.path.join(sgm_dir, f"{basename}_occlusion.npy")

            missing = [p for p in (left_img, gt_pfm, sgm_pfm, mis_npy, occ_npy)
                       if not os.path.isfile(p)]
            if missing:
                skipped.append((frame_num, missing))
                continue

            self._samples.append({
                "left":     left_img,
                "right":    right_img,
                "gt":       gt_pfm,
                "sgm":      sgm_pfm,
                "mismatch": mis_npy,
                "occlusion":occ_npy,
                "scene":    f"driving_{frame_num}",
            })
            loaded += 1

        if skipped:
            import warnings
            warnings.warn(
                f"SceneFlow Driving: {len(skipped)} frames skipped due to "
                f"missing files (first: {skipped[0]})"
            )

        if loaded == 0:
            raise RuntimeError(
                f"No valid SceneFlow Driving samples loaded from {root}."
            )

    def _read_gt(self, s: Dict) -> np.ndarray:
        return _read_pfm(s["gt"])

    def _make_valid(self, gt_raw: np.ndarray):
        # SceneFlow GT is dense synthetic — all positive finite values are valid.
        valid = (gt_raw > 0) & (gt_raw < 512) & np.isfinite(gt_raw)
        gt    = np.where(valid, gt_raw, 0.0).astype(np.float32)
        return gt, valid


# ---------------------------------------------------------------------------
# Middlebury
# ---------------------------------------------------------------------------

class MiddleburyDataset(StereoSGMDataset):
    """Middlebury stereo dataset — MiddEval3-Q only.

    WARNING: sgm_hole data is NOT present on the NFS share at
    /nfs/usrhome/pdongaa/dataczy/Middelburry.  This class will raise
    ``FileNotFoundError`` until SGM pre-computation has been run and the
    resulting .pfm / .npy files are placed under:
        root/MiddEval3/trainingQ/<scene>/sgm_hole/

    Only MiddEval3 Quarter (Q) resolution is available (Full and Half
    resolution images are not on this server).
    """

    def __init__(self, root: str = _DEFAULT_MIDDLEBURY_ROOT) -> None:
        super().__init__()
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Middlebury root not found: {root}\n"
                f"Set SGMVIT_MIDDLEBURY_ROOT or pass root= explicitly.\n"
                f"Note: NFS path is 'Middelburry' (double-l)."
            )

        scene_glob = os.path.join(root, "MiddEval3", "trainingQ", "*/im0.png")
        scenes = sorted(glob(scene_glob))
        if not scenes:
            raise FileNotFoundError(
                f"No Middlebury MiddEval3-Q scenes found at: {scene_glob}"
            )

        missing_sgm_scenes = []
        for img0 in scenes:
            scene_dir = Path(img0).parent
            scene     = scene_dir.name
            sgm_pfm   = str(scene_dir / "sgm_hole" / f"{scene}.pfm")
            if not os.path.isfile(sgm_pfm):
                missing_sgm_scenes.append(scene)

        if missing_sgm_scenes:
            raise FileNotFoundError(
                f"Middlebury sgm_hole data missing for {len(missing_sgm_scenes)} scenes "
                f"(e.g. '{missing_sgm_scenes[0]}').\n"
                f"Pre-compute SGM and save results to:\n"
                f"  {root}/MiddEval3/trainingQ/<scene>/sgm_hole/<scene>.pfm\n"
                f"  {root}/MiddEval3/trainingQ/<scene>/sgm_hole/<scene>_mismatches.npy\n"
                f"  {root}/MiddEval3/trainingQ/<scene>/sgm_hole/<scene>_occlusion.npy"
            )

        for img0 in scenes:
            scene_dir = Path(img0).parent
            scene = scene_dir.name
            img1 = str(scene_dir / "im1.png")
            gt = str(scene_dir / "disp0GT.pfm")
            sgm_dir = scene_dir / "sgm_hole"
            sgm = str(sgm_dir / f"{scene}.pfm")
            mismatch = str(sgm_dir / f"{scene}_mismatches.npy")
            occlusion = str(sgm_dir / f"{scene}_occlusion.npy")

            missing = [p for p in (str(img0), img1, gt, sgm, mismatch, occlusion) if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(
                    f"Middlebury scene '{scene}' is incomplete after SGM pre-computation: {missing}"
                )

            self._samples.append({
                "left": str(img0),
                "right": img1,
                "gt": gt,
                "sgm": sgm,
                "mismatch": mismatch,
                "occlusion": occlusion,
                "scene": scene,
            })

        if not self._samples:
            raise RuntimeError(
                f"No valid Middlebury samples loaded from {root}."
            )

    def _read_gt(self, s: Dict) -> np.ndarray:
        return _read_pfm(s["gt"])

    def _make_valid(self, gt_raw: np.ndarray):
        valid = (gt_raw > 0) & np.isfinite(gt_raw)
        gt    = np.where(valid, gt_raw, 0.0).astype(np.float32)
        return gt, valid


# ---------------------------------------------------------------------------
# KITTI (2015 + 2012 combined)
# ---------------------------------------------------------------------------

class KITTIStereoDataset(StereoSGMDataset):
    """KITTI-2015 + KITTI-2012 combined stereo dataset.

    Root layout:
        root/                   ← KITTI-2015
          training/image_2/*_10.png
          training/disp_occ_0/*_10.png
          training/sgm_hole/*_10.pfm
          kitti2012/            ← KITTI-2012
            training/colored_0/*_10.png
            training/disp_occ/*_10.png
            training/sgm_hole/*_10.pfm
    """

    def __init__(self, root: str = _DEFAULT_KITTI_ROOT) -> None:
        super().__init__()
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"KITTI root not found: {root}\n"
                f"Set SGMVIT_KITTI_ROOT or pass root= explicitly."
            )

        def sg(pat):
            return sorted(glob(pat))

        # KITTI-2015
        root_15 = root
        lefts  = sg(os.path.join(root_15, "training", "image_2",    "*_10.png"))
        rights = sg(os.path.join(root_15, "training", "image_3",    "*_10.png"))
        gts    = sg(os.path.join(root_15, "training", "disp_occ_0", "*_10.png"))
        sgms   = sg(os.path.join(root_15, "training", "sgm_hole",   "*_10.pfm"))
        mis    = sg(os.path.join(root_15, "training", "sgm_hole",   "*_10_mismatches.npy"))
        occ    = sg(os.path.join(root_15, "training", "sgm_hole",   "*_10_occlusion.npy"))
        for l, r, g, s, m, o in zip(lefts, rights, gts, sgms, mis, occ):
            self._samples.append({
                "left": l, "right": r, "gt": g, "sgm": s,
                "mismatch": m, "occlusion": o, "scene": f"kitti15_{Path(l).stem}",
            })

        # KITTI-2012
        root_12 = os.path.join(root, "kitti2012")
        lefts  = sg(os.path.join(root_12, "training", "colored_0", "*_10.png"))
        rights = sg(os.path.join(root_12, "training", "colored_1", "*_10.png"))
        gts    = sg(os.path.join(root_12, "training", "disp_occ",  "*_10.png"))
        sgms   = sg(os.path.join(root_12, "training", "sgm_hole",  "*_10.pfm"))
        mis    = sg(os.path.join(root_12, "training", "sgm_hole",  "*_10_mismatches.npy"))
        occ    = sg(os.path.join(root_12, "training", "sgm_hole",  "*_10_occlusion.npy"))
        for l, r, g, s, m, o in zip(lefts, rights, gts, sgms, mis, occ):
            self._samples.append({
                "left": l, "right": r, "gt": g, "sgm": s,
                "mismatch": m, "occlusion": o, "scene": f"kitti12_{Path(l).stem}",
            })

        if not self._samples:
            raise RuntimeError(
                f"No KITTI samples found under {root}. Check dataset structure."
            )

    def _read_gt(self, s: Dict) -> np.ndarray:
        raw = cv2.imread(s["gt"], cv2.IMREAD_ANYDEPTH)
        return raw.astype(np.float32) / 256.0

    def _make_valid(self, gt_raw: np.ndarray):
        valid = gt_raw > 0
        return gt_raw.astype(np.float32), valid


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== SGM-ViT stereo_datasets.py self-test ===\n")

    # ETH3D
    try:
        ds = ETH3DStereoDataset()
        s0 = ds[0]
        print(f"ETH3D:          {len(ds):3d} samples  |  "
              f"img {s0['left'].shape}  gt_valid {s0['valid'].sum()}")
    except Exception as e:
        print(f"ETH3D:          ERROR — {e}")

    # SceneFlow
    try:
        ds = SceneFlowDrivingDataset()
        s0 = ds[0]
        print(f"SceneFlow Drv:  {len(ds):3d} samples  |  "
              f"img {s0['left'].shape}  gt_valid {s0['valid'].sum()}")
    except Exception as e:
        print(f"SceneFlow Drv:  ERROR — {e}")

    # Middlebury (expected to raise until sgm_hole is pre-computed)
    try:
        ds = MiddleburyDataset()
        print(f"Middlebury:     {len(ds):3d} samples")
    except FileNotFoundError as e:
        print(f"Middlebury:     NOT READY — {str(e).splitlines()[0]}")
    except Exception as e:
        print(f"Middlebury:     ERROR — {e}")

    # KITTI
    try:
        ds = KITTIStereoDataset()
        s0 = ds[0]
        print(f"KITTI:          {len(ds):3d} samples  |  "
              f"img {s0['left'].shape}  gt_valid {s0['valid'].sum()}")
    except Exception as e:
        print(f"KITTI:          ERROR — {e}")
