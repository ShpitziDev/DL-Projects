# Final experiment plan

Status: planning only. Assumes approval of ETH3D as the primary benchmark. No download or execution is authorized by this document.

## Research question

How robust is pretrained VGGT’s camera and dense geometry prediction as view count, overlap, ordering, and image quality vary on laser-scanned real scenes, and how do those measured behaviors compare with qualitative behavior on our own photographs?

## Two-part seminar design

### Part A - ETH3D quantitative benchmark

Select two **training** scenes after inspecting the official download table:

- one indoor scene, provisionally `delivery_area`;
- one outdoor scene, provisionally `courtyard`.

Download only the modalities required for those scenes: undistorted JPEG images, COLMAP-format calibration, cleaned/evaluation laser scan, evaluation/mask data, and optional provided depth. Do not download raw NEF images, all scenes, videos, or test data unless a later need is justified.

### Part B - Original captures

Use at least:

- one controlled rigid object capture;
- one room or outdoor scene;
- optionally one targeted weakness such as reflection, weak texture, or limited motion.

Preserve originals and manifests. These runs are qualitative unless independently measured ground truth becomes available.

## Experimental units

A condition consists of one scene, an ordered list of source frames, one derived image transformation, one VGGT checkpoint/code revision, and one metric protocol. All comparisons change only one intended variable.

### Baseline

- 10 approximately evenly spaced, high-overlap ETH3D views, matching the paper’s reported frame count where practical.
- Original order determined by image/camera sequence metadata; never filesystem enumeration order.
- Official undistorted JPEGs, no degradation.
- Direct pretrained VGGT inference, no BA and no fine-tuning.

### Controlled conditions

| Axis | Conditions | Supported by ETH3D? | Purpose |
|---|---|---:|---|
| Number of views | 2, 4, 6, 8, 10 | Y | Accuracy/runtime/VRAM scaling |
| Order/reference frame | original, reversed, deterministic shuffle, alternate first frame | Y | Canonical-frame and permutation sensitivity |
| Overlap | high, moderate, reduced | Y, selected from calibrated trajectories | Determine breakdown as shared content decreases |
| Blur | fixed Gaussian levels | Y, derived copy | Robustness to motion/focus blur |
| JPEG compression | fixed quality levels | Y, derived copy | Robustness to compression |
| Brightness | deterministic darkening/brightening | Y, derived copy | Photometric robustness |
| Resolution | fixed downsample/upscale levels | Y, derived copy | Loss of detail |
| Reflections | only if visibly present in selected scene | Limited | Descriptive, not a clean controlled factor |
| Dynamic objects | No for static ETH3D benchmark | N | Test only on our captures; qualitative |

Run a pilot first: baseline plus 2/4/6/8/10 views on one scene. Expand only if metrics, alignment, VRAM, and output review are correct.

## Frame selection protocol

1. Parse official calibration and establish a stable ordered candidate set.
2. Exclude unusable frames only by a written rule, never by looking at VGGT outputs.
3. For view-count comparisons, use nested subsets so every smaller set is contained in the larger set where possible.
4. For overlap comparisons, use calibration to select camera pairs/sets by shared trajectory proximity and verify actual visual overlap manually before inference.
5. Record image names and SHA-256 hashes in every run manifest.

## Quantitative metrics

### Dense geometry - primary

Reproduce the paper’s ETH3D-style evaluation as closely as the available official evaluator permits:

- transform VGGT points to the ETH3D reference using a similarity alignment estimated by a documented method;
- apply official valid-region and occlusion handling;
- report accuracy (prediction-to-ground-truth distance), completeness (ground-truth-to-prediction distance), and overall mean;
- optionally report precision, recall, and F-score at predeclared distance thresholds.

Do not tune confidence thresholds separately for every condition. Select thresholds on a pilot and freeze them, or show a small threshold curve.

### Cameras - secondary

- align predicted camera centers to ground-truth centers with a similarity transform;
- report relative rotation error and translation-direction error where conventions are verified;
- report camera-center error after alignment as an additional descriptive metric;
- test order variants using invariant relative quantities, not raw world coordinates.

### Efficiency

- model inference time after CUDA warm-up;
- end-to-end time separately;
- peak allocated and reserved VRAM;
- processed frame count and resolution.

### Stability under degradations

Always compare a degraded condition to the exact same original frames. Report geometry/camera metric deltas, not only absolute scores.

## Qualitative rubric

For every benchmark and custom scene, inspect fixed viewpoints and record:

- global shape coherence;
- holes, floaters, duplication, and warped planes;
- foreground/background leakage;
- camera flips, jumps, or collapsed baselines;
- failure regions versus predicted confidence;
- track continuity where tracking is intentionally enabled.

Use identical rendering bounds, point sizes, confidence thresholds, and screenshots for paired comparisons. Do not call confidence an accuracy metric.

## Planned hypotheses

1. Increasing views from 2 toward 8–10 improves completeness until memory or weak-overlap views introduce diminishing returns.
2. Relative geometry is mostly stable under order changes after alignment, but changing the first frame may expose measurable reference-frame sensitivity.
3. Reduced overlap causes a larger failure than mild JPEG compression or brightness changes.
4. Blur and resolution loss reduce completeness and confidence before gross camera failure.
5. Curated ETH3D scenes outperform reflective, low-texture, or dynamic custom captures qualitatively.

These are hypotheses, not results.

## Minimal run matrix

Stage 1, one ETH3D scene:

- baseline with 10 views;
- nested 2/4/6/8-view subsets;
- one reversed-order and one shuffled-order 10-view run;
- one moderate reduced-overlap set;
- one level each of blur, JPEG compression, and darkening.

Stage 2, second ETH3D scene:

- baseline;
- the two most informative view counts;
- the two degradations that caused the clearest Stage 1 changes.

Stage 3, our images:

- controlled-object baseline and selected view counts;
- one natural scene baseline;
- one deliberately difficult condition if captured safely.

This is approximately 18–22 inference conditions, not a combinatorial grid.

## Runtime and resource budget

- Use the already validated local checkpoint and pinned maintained VGGT code.
- Start with at most 10 views at the project’s standard preprocessing size.
- Run one warm-up outside timed measurements, then at least three timed repetitions only if inference cost permits.
- Treat 16 GB VRAM as a hard constraint; stop increasing views when measured headroom becomes unsafe.
- **Planning estimate:** inference should be seconds per condition, but preprocessing, evaluation, exports, and manual QA make a full pilot a multi-hour activity. Replace this estimate with local measurements after the first approved benchmark run.

## Evidence and reporting rules

- Label paper numbers as **author-reported**.
- Label our computed ETH3D metrics as **local benchmark measurements** with exact protocol and sample count.
- Label custom-image judgments as **qualitative local observations**.
- Do not claim official benchmark reproduction unless preprocessing, split, alignment, masks, and metrics match the paper/evaluator.
- Report failures and excluded cases; do not select only visually successful outputs.
- Because ETH3D was not listed as VGGT training data, describe it as an evaluation benchmark, but avoid claiming guaranteed absence from every upstream/pretraining source.

## Acquisition gate for the next phase

Before downloading anything:

1. Re-read and record the current ETH3D license/terms.
2. Confirm the two selected scene archives and their exact compressed/uncompressed sizes.
3. Confirm at least 20 GB free workspace capacity for archives, extraction, and derived outputs.
4. Freeze which modalities are necessary.
5. Obtain explicit approval for those exact URLs and estimated bytes.

## Implementation sequence after approval

1. Download only the approved ETH3D scene archives with checksums where available.
2. Add an ignored benchmark-data directory and immutable acquisition manifest.
3. Implement an ETH3D adapter outside `external/`.
4. Unit-test camera parsing, coordinate conventions, alignment, masks, and metrics using tiny fixtures.
5. Run one two-view smoke condition, inspect outputs, then run the minimal pilot.
6. Decide whether DTU adds enough complementary evidence before any secondary download.

## Stop conditions

Stop and reassess if the license is unsuitable, required ground truth is unavailable, coordinate conventions cannot be verified, an evaluator cannot be validated on a known fixture, or the pilot exceeds the approved time/storage/VRAM budget.

