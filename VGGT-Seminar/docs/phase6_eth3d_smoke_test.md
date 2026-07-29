# Phase 6 ETH3D two-view smoke test

Run date: 2026-07-20. Status: **passed as a pipeline smoke test**. This is one configuration, not an ETH3D benchmark result.

## Approved configuration

| Field | Value |
|---|---|
| Dataset / scene | ETH3D high-resolution training / `delivery_area` |
| Original ETH3D indices | 0, 43 |
| Filenames | `DSC_0675.JPG`, `DSC_0718.JPG` |
| Original resolution | 6208 x 4135 for both images |
| Selection | `round(linspace(0, 43, 2)) = [0, 43]` |
| Order | Original |
| Preprocessing | Official `load_and_preprocess_images(..., mode="crop")` |
| Preprocessed tensor | 2 x 3 x 350 x 518 |
| Tracking | One center query, enabled in the same forward pass |
| Forward passes | Exactly one |

Both source images loaded successfully and remained unmodified. Their RGB statistics and the contact sheet are saved in `selected_frames.json` and `visualizations/contact_sheet.jpg`.

## Environment and model - observed facts

- Project root: `D:\cv_sem`
- Python executable: project-local `vggt-seminar` environment
- Python: 3.11.15
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- GPU: NVIDIA GeForce RTX 5080
- Pre-run GPU memory: 15,712,911,360 bytes free of 17,094,475,776 bytes total
- Precision: CUDA BF16 autocast
- Flash SDPA: explicitly disabled
- Official maintained VGGT commit: `a288dd0f14786c93483e45524328726ab7b1b4ce`
- Checkpoint: `local_assets/checkpoints/model.pt`, 5,026,874,952 bytes
- Checkpoint SHA-256: `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`
- Checkpoint loading: explicit local `torch.load(..., weights_only=True, mmap=True)`; no network and no `from_pretrained`

## Runtime - observed facts

| Stage | Seconds |
|---|---:|
| Model architecture initialization | 5.6631 |
| Checkpoint loading | 1.3536 |
| Model transfer to GPU | 1.0924 |
| Preprocessing and input transfer | 0.4905 |
| Single synchronized inference | 1.3533 |
| Postprocessing and artifact generation | 2.4040 |
| Total runner elapsed time | 18.6739 |

- Baseline allocated VRAM immediately before inference: 5,035,973,120 bytes (4.69 GiB).
- Peak allocated VRAM: 5,648,585,216 bytes (5.26 GiB).
- Peak reserved VRAM: 6,316,621,824 bytes (5.88 GiB).

The total includes checkpoint SHA-256 verification and other preflight work not assigned to the stage subtimings.

## Output validity - observed facts

Every tensor and derived array below was 100% finite, with zero NaNs and zero infinities.

| Tensor | Shape | Dtype | Finite |
|---|---|---|---:|
| `pose_enc` | 1 x 2 x 9 | float32 | 100% |
| `pose_enc_list` | 4 items, each 1 x 2 x 9 | float32 | 100% |
| `depth` | 1 x 2 x 350 x 518 x 1 | float32 | 100% |
| `depth_conf` | 1 x 2 x 350 x 518 | float32 | 100% |
| `world_points` | 1 x 2 x 350 x 518 x 3 | float32 | 100% |
| `world_points_conf` | 1 x 2 x 350 x 518 | float32 | 100% |
| `track` | 1 x 2 x 1 x 2 | float32 | 100% |
| `vis` | 1 x 2 x 1 | bfloat16 | 100% |
| `conf` | 1 x 2 x 1 | bfloat16 | 100% |
| `images` | 1 x 2 x 3 x 350 x 518 | float32 | 100% |
| decoded `extrinsic` | 1 x 2 x 3 x 4 | float32 | 100% |
| decoded `intrinsic` | 1 x 2 x 3 x 3 | float32 | 100% |
| depth-unprojected points | 2 x 350 x 518 x 3 | float64 CPU | 100% |

Full minima, maxima, means, medians, device names, and invalid counts are preserved in `tensor_summaries.json` rather than duplicated here.

## Visual inspection

### Observed facts

- The selected endpoint images depict different parts/directions of the delivery area and have limited obvious visual overlap.
- Both depth maps contain structured scene-dependent variation rather than constant or invalid output.
- Direct and depth-unprojected point previews both form large surfaces and room-like structure, though they differ visibly.
- Decoded camera centers are distinct and finite.
- The point-confidence distribution has a hard floor/median of 1.0. Filtering with `>= median` initially retained all 362,600 points.
- After changing the rule to `> median`, 164,621 points were retained. This repair used saved CPU tensors and did not perform another inference.
- The single tracking query is finite, but tracking confidence ranges from 0.00255 to 1.0 across the two frames.

### Interpretations

- The model and serialization pipeline behave correctly on two genuine high-resolution ETH3D inputs.
- The low tracking confidence on one view is consistent with weak endpoint overlap, but one query is not enough to establish a failure cause.
- Visual coherence is encouraging but is not metric accuracy and has not been compared with the laser scan.

### Unresolved questions

- Whether camera and geometry estimates are accurate after a frozen similarity alignment.
- Whether the endpoint pair has sufficient shared content for a meaningful two-view evaluation.
- Whether direct points or depth-unprojected points score better under the official masks.
- How confidence relates to actual scan error.

## Camera-convention findings

VGGT decodes OpenCV-style camera-from-world (world-to-camera) matrices `[R|t]`. Camera centers follow `C = -R^T t`. ETH3D/COLMAP uses the same camera-axis convention and transformation direction, but VGGT uses the first input image as its reference and predicts arbitrary normalized scale. The world frames are therefore not aligned. Quantitative comparison requires a documented similarity transform; axis/handedness must still be verified with a projection fixture. See `docs/eth3d_camera_conventions.md`.

## Artifacts

The unique ignored run directory is:

`outputs/predictions/phase6_eth3d_smoke/delivery_area/2_views_evenly_spaced_original/`

It contains the resolved configuration, preflight metadata, selected-frame record, runtime/VRAM data, tensor summaries, camera parameters, compact PyTorch arrays, three PLY exports, contact sheet, six depth/confidence maps, camera-center plot, three point-cloud previews, execution log, and an empty error log.

## Problems and fixes

1. **Confidence filter retained all points.** Cause: confidence median equals the activation floor 1.0 and the initial rule used `>=`. Fix: use `confidence > median`; regenerate only the filtered PLY/preview from saved tensors.
2. **PyTorch warning.** The maintained official VGGT source uses deprecated `torch.cuda.amp.autocast(enabled=False)` internally. It did not affect completion. Official source was not edited.
3. **Selection quality concern.** Deterministic full-sequence spacing selects endpoints with weak overlap. No data/code was altered, but the pilot design should add an overlap-aware nested frame rule before treating results as quantitative evidence.

## Readiness decision

The inference, output parsing, camera decoding, serialization, visualization, and memory path are ready for a controlled pilot. The project is **not yet ready to interpret the full 2/4/6/8/10 matrix scientifically** until overlap-aware nested subsets and the alignment/evaluation protocol are frozen. No laser-scan metric was computed in this phase.

