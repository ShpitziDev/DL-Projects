# Phase 7 delivery-area view-count pilot

Run date: 2026-07-20. Status: **passed**. This is a single-scene, single-run-per-condition pilot, not ETH3D accuracy evaluation.

## Research design

The question is how operational cost, model confidence, predicted camera structure, and visible reconstruction completeness change as frozen input view count increases. The independent variable is view count: 2, 4, 6, 8, or 10. Scene, protocol `eth3d-overlap-aware-nested-v1`, checkpoint, preprocessing, original order, BF16 precision, RTX 5080 device, Flash-SDPA setting, tracking query, confidence rule, serialization, and visualization code were held constant.

Frozen sets were S2 `[0,6]`, S4 `[0,3,6,9]`, S6 `[0,3,4,5,6,9]`, S8 `[0,1,2,3,4,5,6,9]`, and S10 `[0,1,2,3,4,5,6,7,8,9]`. Exact filenames remain versioned in the Phase 6.1 config. The validated Phase 6.2 S2 result passed protocol, checkpoint, schema, artifact, finite-value, BF16, and one-forward checks and was referenced rather than rerun. S4–S10 each executed once after one shared model/checkpoint load.

## Environment and one-time setup — observed measurements

- Pre-execution commit: `313d171`.
- Python 3.11.15, PyTorch 2.13.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 5080.
- Checkpoint SHA-256: `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`.
- Architecture initialization: 5.2138 s; checkpoint load: 1.1082 s; GPU transfer: 0.8948 s.
- Four new forward passes; five represented configurations. No CPU fallback or network access.

## Timing and memory — observed measurements

Subset totals exclude one-time model setup and equal preprocessing + inference + postprocessing.

| Set | Preprocess s | Inference s | Postprocess s | Subset total s | Allocated GiB | Reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 0.443 | 0.906 | 1.859 | 3.208 | 5.261 | 5.883 |
| S4 | 0.932 | 0.971 | 5.905 | 7.808 | 5.631 | 7.023 |
| S6 | 1.374 | 0.853 | 8.485 | 10.712 | 6.084 | 8.074 |
| S8 | 1.799 | 0.897 | 11.202 | 13.900 | 6.535 | 9.166 |
| S10 | 2.316 | 1.142 | 13.928 | 17.386 | 6.599 | 9.230 |

## Confidence — observed measurements

| Set | Depth mean / median | Point mean / median | Threshold | Retained |
|---|---:|---:|---:|---:|
| S2 | 1.916 / 1.519 | 2.443 / 2.389 | 2.389 | 181,300 (50.0%) |
| S4 | 3.609 / 3.949 | 3.711 / 3.390 | 3.390 | 362,600 (50.0%) |
| S6 | 4.196 / 4.528 | 4.242 / 3.892 | 3.892 | 543,900 (50.0%) |
| S8 | 3.964 / 4.097 | 3.996 / 3.583 | 3.583 | 725,200 (50.0%) |
| S10 | 5.017 / 5.552 | 4.477 / 3.796 | 3.796 | 906,499 (50.0%) |

Every saved tensor and derived array was fully finite. The near-exact 50% retention is expected from the strict `confidence > median` rule; it is not evidence that half the geometry is accurate.

## Geometry output — observed measurements

| Set | Raw / valid points | Camera path | Maximum separation |
|---|---:|---:|---:|
| S2 | 362,600 / 362,600 | 0.5340 | 0.5340 |
| S4 | 725,200 / 725,200 | 0.7415 | 0.7414 |
| S6 | 1,087,800 / 1,087,800 | 0.7702 | 0.7700 |
| S8 | 1,450,400 / 1,450,400 | 0.8311 | 0.8307 |
| S10 | 1,813,000 / 1,813,000 | 0.7476 | 0.7472 |

Camera quantities are VGGT arbitrary units in separately predicted, unaligned frames. They are structural diagnostics, not metric ETH3D distances or pose accuracy.

## Plots and qualitative comparison

Eleven plots are generated exclusively from canonical `summary.csv`: inference/total time, allocated/reserved VRAM, mean and median confidences, retained count/percentage, camera path, and maximum separation. Galleries compare frozen contacts, first-view depth/confidence, camera paths, direct previews, and filtered previews.

Observed visually, the same doorway/pillar/ceiling surfaces remain recognizable from S2 through S10. Added views increase scene coverage and point density without obvious catastrophic geometry failure. Depth for the shared first frame is stable in broad layout. Camera paths are finite and ordered, but their normalized shape/extent changes across independently predicted conditions. Point previews use deterministic XY projection, uniform sampling to at most 40,000 points, and independent 1st–99th percentile auto-fit; this makes structure visible but prevents direct crop/scale comparison.

## Descriptive trends and anomalies

- Allocated memory increases from 5.26 to 6.60 GiB; reserved memory rises to 9.23 GiB and nearly plateaus from S8 to S10.
- Inference time is sublinear/non-monotonic for these single runs; the largest increase is S8→S10 (+0.244 s). No timing-average claim is made.
- Postprocessing and total subset time rise approximately with saved view count and dominate the runner by S10.
- Retained point count rises monotonically because output pixels scale with view count and the median rule retains about half.
- Confidence improves overall but dips at S8, then rises at S10; it does not yet appear stabilized.
- Camera path rises through S8 and falls at S10. This may reflect condition-dependent arbitrary normalization, so no physical-motion explanation is inferred.

## Problems and fixes

The first launcher attempt failed before model construction because direct script execution did not resolve the `scripts` namespace. The import was made compatible with both module and direct execution; focused tests passed and no output directory or forward pass existed before restart. After execution, reused S2 total time was corrected to exclude one-time setup, matching Phase 7 subset semantics. The contact gallery was changed from tiny horizontal thumbnails to readable vertical panels. Both corrections used saved artifacts only.

## Interpretation, limitations, and readiness

Interpretation: the frozen delivery-area series is operationally coherent through ten views. It demonstrates manageable memory scaling, valid outputs, stable shared structure, and increasing represented geometry. It does not demonstrate metric accuracy, statistical significance, cross-scene generalization, or confidence calibration against scan error.

Unresolved questions include whether courtyard exhibits similar behavior, whether predicted cameras align accurately to calibration, whether direct or unprojected points are more accurate, and whether confidence correlates with scan error.

**Readiness decision:** the identical frozen original-order courtyard pilot is operationally justified, pending explicit approval. Courtyard remains a new scene and must not be assumed to reproduce delivery-area trends.
