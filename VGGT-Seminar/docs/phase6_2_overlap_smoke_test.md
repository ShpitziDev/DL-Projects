# Phase 6.2 overlap-aware S2 smoke test

Run date: 2026-07-20. Status: **passed**. This is a protocol-validation comparison, not benchmark accuracy.

## Purpose and frozen input

The run validates frozen protocol `eth3d-overlap-aware-nested-v1` before the view-count pilot. It used only `delivery_area` S2, original-order indices `[0, 6]`: `DSC_0675.JPG` and `DSC_0681.JPG`. S2 was verified as a strict subset of S4 `[0, 3, 6, 9]`; the runner loads these indices from the versioned Phase 6.1 config and forbids fallback.

Frozen selection evidence: camera-center spacing 4.1184, viewing-direction angle 4.2463°, relative rotation 4.2984°, 176 ORB ratio matches (normalized 0.07246), and 73 fundamental-matrix RANSAC inliers (ratio 0.4148). These select inputs; they do not measure VGGT accuracy.

## Environment and execution — observed facts

- Python 3.11.15; PyTorch 2.13.0+cu130; CUDA 13.0.
- NVIDIA GeForce RTX 5080; 15.71 GB free of 17.09 GB before allocation.
- BF16 autocast; Flash SDPA disabled; no CPU fallback.
- Maintained VGGT commit `a288dd0f14786c93483e45524328726ab7b1b4ce`.
- Local checkpoint SHA-256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`.
- Exactly one forward call; no network access and no `from_pretrained`.

| Stage | Seconds |
|---|---:|
| Architecture initialization | 5.4038 |
| Checkpoint loading | 1.1709 |
| GPU transfer | 0.9963 |
| Preprocessing/input transfer | 0.4429 |
| Synchronized inference | 0.9057 |
| Postprocessing/artifacts | 1.8591 |
| Total runner elapsed | 16.4181 |

Peak allocated VRAM was 5,648,585,216 bytes (5.26 GiB); peak reserved was 6,316,621,824 bytes (5.88 GiB).

## Output validity — observed facts

All tensors were 100% finite with zero NaNs and infinities.

| Output | Shape | Dtype |
|---|---|---|
| pose encoding | 1×2×9 | float32 |
| depth | 1×2×350×518×1 | float32 |
| depth confidence | 1×2×350×518 | float32 |
| world points | 1×2×350×518×3 | float32 |
| point confidence | 1×2×350×518 | float32 |
| track | 1×2×1×2 | float32 |
| visibility / tracking confidence | 1×2×1 | bfloat16 |
| decoded extrinsics | 1×2×3×4 | float32 |
| decoded intrinsics | 1×2×3×3 | float32 |
| depth-unprojected points | 2×350×518×3 | float64 CPU |

Point-confidence median was 2.38922. The corrected `confidence > median` rule retained 181,300 of 362,600 points (50.0%). Direct, unprojected, and filtered PLY exports all succeeded.

## Protocol-validation comparison with Phase 6

| Measure | Phase 6 endpoints `[0,43]` | Phase 6.2 frozen `[0,6]` |
|---|---:|---:|
| Saved input-overlap proxy | Not recorded | 176 matches / 73 inliers |
| Inference time | 1.3533 s | 0.9057 s |
| Peak allocated/reserved | 5.26 / 5.88 GiB | 5.26 / 5.88 GiB |
| Depth-confidence mean / median | 1.0016 / 1.0000 | 1.9159 / 1.5186 |
| Point-confidence mean / median | 1.0005 / 1.0000 | 2.4430 / 2.3892 |
| Finite outputs | 100% | 100% |
| Filtered points | 164,621 (45.4%) | 181,300 (50.0%) |
| VGGT camera separation | 0.3215 arbitrary units | 0.5340 arbitrary units |

Observed visually, `[0,6]` retains the doorway, striped pillar, ceiling, and adjacent delivery-area structure in both views. The endpoint pair shows substantially less common content. New direct and depth-unprojected previews agree on the main planar structure, decoded cameras are distinct, and confidence filtering retains recognizable high-confidence surfaces.

Interpretation: the frozen pair is a more defensible non-trivial two-view input and produces stronger confidence distributions without additional VRAM. Runtime variation is not a performance conclusion from one run. Camera separation is in VGGT’s arbitrary coordinate system and is not metric or accuracy evidence.

## Problems, fixes, and limitations

The model run itself required no fix. The maintained upstream source emitted its known deprecated-autocast warning; official code was not edited. Saved heatmaps/previews initially lacked embedded condition titles, so nine PNGs were retitled from existing artifacts without inference; the runner now applies the same wrapper-level title step. No ETH3D alignment, scan metric, accuracy calculation, repeatability study, or other frame count/scene was run.

## Scientific readiness decision

**A — Ready for the full delivery_area 2/4/6/8/10-view pilot, pending explicit user approval.** The exact frozen pair ran once, all required structures were finite, exports succeeded, shared scene content is visibly stronger than the endpoint smoke test, and no selection fallback occurred. The pilot must retain the frozen nested inputs and treat geometry as unaligned until a separate evaluation protocol is approved.
