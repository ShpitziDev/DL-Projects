# ETH3D evaluation plan

Status: design only. Quantitative evaluation is not implemented in Phase 5.

## Available reference data

The selected scenes provide:

- calibrated PINHOLE intrinsics for the supplied undistorted images (whose canvas is slightly larger than the nominal 6048 x 4032 capture size);
- world-to-camera poses in COLMAP `images.txt` format;
- cleaned laser scans and evaluation scan files in PLY format;
- an MLP scan-alignment description;
- per-image evaluation masks;
- an occlusion surface mesh and splat cloud.

No separate depth archive was downloaded. If depth evaluation is later justified, reference depth can be rendered from the aligned scan/occlusion representation or the official depth archive can be proposed as a separately approved acquisition.

## Candidate metrics

### Dense geometry

- **Accuracy:** nearest-neighbor distance from retained VGGT points to reference geometry.
- **Completeness:** nearest-neighbor distance from reference geometry to retained VGGT points.
- **Overall:** mean of accuracy and completeness, matching the convention reported by VGGT.
- **Precision/recall/F-score:** at fixed, predeclared distance thresholds as supplemental summaries.

### Camera geometry

- relative rotation error between image pairs;
- relative translation-direction angular error;
- camera-center error after similarity alignment;
- AUC across predeclared angular thresholds only if the protocol is frozen first.

### Robustness

- change in geometry and camera errors versus the exact matched baseline;
- runtime and peak VRAM versus view count;
- confidence-error calibration, treating predicted confidence as a model output rather than correctness.

## Coordinate and scale alignment

ETH3D poses transform global points into each camera frame. VGGT predictions use the first input camera as their reference and an internally learned normalized scale. Raw coordinates therefore cannot be compared directly.

The planned alignment is:

1. verify quaternion convention, world-to-camera direction, camera axes, and units using a small visual fixture;
2. convert ETH3D and VGGT camera centers into consistent world-coordinate representations;
3. estimate a 7-DoF similarity transform (rotation, translation, and uniform scale), preferably from corresponding camera centers using Umeyama alignment;
4. apply that transform to predicted cameras and points;
5. use official masks, bounding/evaluation regions, and occlusion handling before distance computation.

Alignment parameters must be estimated independently per scene/condition according to a frozen rule. Do not use ground-truth geometry to tune a transform beyond the declared alignment procedure.

## ICP

ICP may refine residual geometry alignment after camera-based similarity alignment, but it can hide camera or global reconstruction failures. Therefore:

- report **camera-similarity-aligned** results as primary;
- if ICP is used, report it separately as a post-alignment diagnostic;
- freeze ICP initialization, distance threshold, iteration count, point sampling, and convergence rule;
- never compare an ICP-refined condition against a non-ICP baseline.

## Limitations

- The seminar subset contains only two scenes and cannot establish broad benchmark ranking.
- VGGT paper sampling was random; deterministic seminar subsets will not exactly reproduce its aggregate numbers.
- High-resolution inputs are resized by official VGGT preprocessing, while ground truth retains higher spatial detail.
- Laser-scan visibility, masks, confidence filtering, density, and distance thresholds materially affect results.
- Camera-center alignment becomes weak with very few views or nearly collinear trajectories.
- ETH3D is nominally static; it does not support quantitative dynamic-object experiments.
- The absence of the optional pre-rendered depth bundle prevents direct pixel-depth scoring in Phase 5.

## Future work gate

Before implementing metrics, inspect the official ETH3D evaluator and VGGT evaluation code, freeze the exact mask and threshold protocol, validate coordinate conversions on known cameras, and test against a tiny synthetic fixture with a known transform. Only then run a two-view smoke evaluation followed by the planned 2/4/6/8/10-view study.
