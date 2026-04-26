# FusionNet Experiment Log

This file tracks the learned-fusion search after heuristic fusion was frozen as a fixed baseline.
It is intended to be updated alongside `results/` and `artifacts/` so that each accepted or rejected route has a paper-ready record.

## Current Best

- Current best learned fusion: `mask_residual_lite`
- Primary reference summary: `results/fusion_arch_search/mask_residual_lite/eval/summary.json`
- Primary checkpoint: `artifacts/fusion_arch_search/mask_residual_lite/kitti_finetune/best.pt`
- Current best KITTI metrics:
  - `EPE 1.8996`
  - `D1 15.0201`
  - `boundary_epe 1.8742`
  - `flat_region_noise 0.3819`

## Search Timeline

### 1. mask_residual_lite baseline

- Motivation: first learned residual fusion route that consistently constrained update regions instead of changing the whole fused map.
- Key changes: lightweight residual U-Net, explicit update mask, anchor suppression, disagreement/detail prior.
- Representative result:
  - KITTI: `EPE 1.8996`, `D1 15.0201`, `boundary_epe 1.8742`, `flat_region_noise 0.3819`
  - SceneFlow: `EPE 5.9743`
  - ETH3D: `EPE 0.8408`
  - Middlebury: `EPE 2.1442`
- Status: current best learned fusion baseline.

## Accepted Directions

- Keep `heuristic_fused` as the fixed base map and learn a small residual correction on top.
- Prefer small update masks and near-zero anchor-region activation.
- Use boundary-aware metrics and demo panels together; do not rely on scalar EPE alone.
- Treat external datasets as generalization constraints even when KITTI remains the main ranking target.

## Rejected Directions

### legacy unrestricted residual

- Problem: too much freedom, weak anchor protection, and unstable cross-dataset behavior.
- Conclusion: removed from the main learned-fusion path.

### detail_restore_v2

- Problem: the absolute-base/detail split did not outperform `mask_residual_lite` in the current implementation.
- Conclusion: kept only as a historical branch, not as the active search direction.

## Paper-ready Takeaways

- The strongest learned-fusion route so far is not a full replacement of heuristic fusion, but a constrained residual refinement over a fixed heuristic base.
- The main learned gain comes from changing fewer pixels more selectively, especially outside anchor regions.
- The next bottleneck is update-region modeling and supervision design, not simply adding a larger fusion backbone.
