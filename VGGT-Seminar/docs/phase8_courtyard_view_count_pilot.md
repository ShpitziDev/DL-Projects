# Phase 8 courtyard view-count pilot

Run date: 2026-07-20. Status: **passed**. This is a two-scene descriptive extension of Phase 7, not an ETH3D accuracy evaluation.

## Research design and frozen inputs

The question is whether the operational scaling and qualitative behavior observed on `delivery_area` also appear on a second frozen scene. View count is the independent variable. Model, checkpoint, preprocessing, BF16 precision, RTX 5080, original order, protocol `eth3d-overlap-aware-nested-v1`, confidence filtering, serialization, and plotting conventions match Phase 7.

Courtyard sets are S2 `[0,9]`, S4 `[0,4,6,9]`, S6 `[0,2,3,4,6,9]`, S8 `[0,2,3,4,5,6,7,9]`, and S10 `[0,1,2,3,4,5,6,7,8,9]`. They span `DSC_0286.JPG` through `DSC_0295.JPG`; exact filenames are frozen in the Phase 6.1 config.

## Environment and shared setup — measured results

- Pre-execution commit: `a1406fe`.
- Python 3.11.15, PyTorch 2.13.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 5080.
- Checkpoint SHA-256: `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`.
- Architecture initialization: 5.3332 s; checkpoint load: 1.1442 s; GPU transfer: 0.8915 s.
- Exactly five new CUDA forwards; no reuse, delivery-area inference, CPU fallback, or network access.

## Timing and VRAM — measured results

| Set | Preprocess s | Inference s | Postprocess s | Total s | Allocated GiB | Reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 0.471 | 0.876 | 3.163 | 4.510 | 5.261 | 5.883 |
| S4 | 0.983 | 0.713 | 5.918 | 7.614 | 5.631 | 7.023 |
| S6 | 1.434 | 0.727 | 8.698 | 10.858 | 6.084 | 8.074 |
| S8 | 1.966 | 0.974 | 11.556 | 14.496 | 6.535 | 9.166 |
| S10 | 2.447 | 1.055 | 15.143 | 18.645 | 6.599 | 9.230 |

## Confidence — measured results

| Set | Depth mean / median | Point mean / median | Retained |
|---|---:|---:|---:|
| S2 | 2.319 / 2.516 | 3.034 / 2.931 | 181,300 (50.0%) |
| S4 | 5.162 / 5.410 | 4.908 / 4.874 | 362,600 (50.0%) |
| S6 | 6.535 / 6.832 | 5.591 / 5.488 | 543,900 (50.0%) |
| S8 | 8.374 / 8.739 | 6.298 / 6.158 | 725,199 (49.9999%) |
| S10 | 8.012 / 8.372 | 6.151 / 6.019 | 906,499 (49.9999%) |

Every tensor and derived array was 100% finite with zero NaNs or infinities. Retention follows the strict `confidence > median` rule and is not an accuracy measure.

## Geometry output — measured results

| Set | Raw / valid points | Camera path | Maximum separation |
|---|---:|---:|---:|
| S2 | 362,600 / 362,600 | 0.7997 | 0.7997 |
| S4 | 725,200 / 725,200 | 0.8940 | 0.8931 |
| S6 | 1,087,800 / 1,087,800 | 0.9421 | 0.9406 |
| S8 | 1,450,400 / 1,450,400 | 0.9273 | 0.9258 |
| S10 | 1,813,000 / 1,813,000 | 0.9191 | 0.9174 |

All camera and geometry quantities are separately predicted, arbitrary, unaligned VGGT units—not metric scene distances or pose accuracy.

## Courtyard plots and visual observations

Eleven courtyard plots derive only from canonical `summary.csv`. Seven galleries cover frozen contacts, first-view depth and confidences, predicted cameras, direct points, and filtered points. Preview parameters match Phase 7: deterministic XY projection, uniform sampling up to 40,000 points, 1st–99th percentile per-condition auto-fit, no manual crop, and no cross-condition metric scale.

Observed visually, the brick façade, arched windows, tables, chairs, and ground plane remain recognizable across S2–S10. Depth layout is stable for the shared first view, and filtered previews preserve façade/window structure while density increases. The S6 direct preview appears more foreshortened under auto-fit than neighboring conditions, but its filtered preview, cameras, depth, and validity checks show no corresponding operational failure.

## Courtyard trends and anomalies

- Allocated/reserved VRAM rise identically to Phase 7 because tensor dimensions depend on view count, reaching 6.60/9.23 GiB at S10.
- Inference time is non-monotonic in single runs; the largest increase is S6→S8 (+0.248 s). No averaging or significance claim is made.
- Total time rises with view count because preprocessing and artifact serialization dominate.
- Confidence rises strongly through S8 and then dips 4.3% (depth) and 2.3% (points) at S10. Under the documented descriptive 5% criterion, it appears to be stabilizing at S8–S10.
- Camera path peaks at S6 and declines slightly at S8/S10. Arbitrary condition-dependent scale prevents physical interpretation.
- Geometry remains visibly coherent; additional views mostly add density/coverage, with diminishing qualitative change after S8.

## Two-scene descriptive comparison

Both scenes have identical allocated/reserved VRAM at each view count and similar sub-second-to-1.14-second inference ranges. Their total runtimes are close; courtyard is higher at S2/S6/S8/S10 and slightly lower at S4, with serialization variation likely contributing. This is not a repeated timing benchmark.

Courtyard model confidence is higher at every matched view count, but confidence alone cannot establish that courtyard is objectively easier. Trends differ: delivery-area confidence dips at S8 then rises at S10, while courtyard rises through S8 and slightly declines at S10. Courtyard camera-path values are larger at every count and peak at S6; delivery-area peaks at S8. Because each condition has arbitrary unaligned prediction scale, these are only within-output structural descriptions.

Shared qualitative behavior includes stable first-view depth layout, coherent dominant surfaces, increasing point density, and limited visible improvement at the largest counts. Scene-dependent content differs substantially: courtyard contains repeated windows, thin furniture, and outdoor ground, while delivery area is dominated by large indoor planes and door/pillar structure.

## Problems, limitations, and readiness

No execution fix was required. The maintained upstream implementation emitted its known deprecated-autocast warning; official source was not modified. Automated anomaly wording inherited from Phase 7 initially named the wrong subset; it was generalized to derive dips from the actual aggregate and the saved trend JSON was corrected without inference.

This study has only two scenes and one run per condition. It provides no timing variance, scan alignment, pose/point accuracy, confidence calibration, statistical inference, or basis for model comparison/generalization claims. Auto-fit previews support structural inspection but not metric cross-condition size comparisons.

**Readiness:** the two-scene frozen view-count study is complete enough to begin report writing. Accuracy evaluation, order sensitivity, degradations, and fine-tuning remain separate unapproved experiments.
