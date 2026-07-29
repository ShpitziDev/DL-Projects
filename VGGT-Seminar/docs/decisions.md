# Decision log

## 2026-07-20 — third-party integration (completed)

- **Our decision:** use a normal ignored clone at `external/vggt`, with tracked pin metadata outside it, and install editable with `--no-deps`.
- **Repository:** `https://github.com/facebookresearch/vggt.git`.
- **Commit:** `a288dd0f14786c93483e45524328726ab7b1b4ce`, current `origin/main` when cloned, commit date 2026-05-18. The checkout was clean. The repository has no tags/releases in Git metadata.
- **Reason:** simpler than a submodule for a seminar workspace while remaining reproducible through `external/VGGT_PIN.json`; prevents copied third-party source from entering our history.
- **Expected checkpoints:** research checkpoint `facebook/VGGT-1B`; alternative gated commercial checkpoint `facebook/VGGT-1B-Commercial`. Do not download either without approval.
- **License:** repository uses VGGT License v1 (updated 2025-07-29) with an Acceptable Use Policy. The official README states only the newer commercial checkpoint permits commercial use; original weights remain non-commercial. Re-check terms at acquisition time.

## 2026-07-20 — platform and dependencies

- Start native Windows because core requirements are ordinary Python packages and WSL is absent.
- **Our decision:** project-local Micromamba 2.8.1, environment `vggt-seminar`, Python 3.11.15 at `.micromamba/envs/vggt-seminar`.
- Do not reuse the global Python 3.12 nightly Torch environment for reproducible experiments.
- **Our decision:** replace historical Torch 2.3.1/torchvision 0.18.1 with stable Torch 2.13.0/torchvision 0.28.0 official CUDA 13.0 wheels. CUDA validation passed on compute capability 12.0.
- Treat optional demo/BA packages separately; Linux/WSL becomes a decision only if a required experiment dependency proves unsupported on Windows.

## 2026-07-20 — revision selection

- **Official code:** current main includes training code, a May 2026 memory-retention fix, chunked DPT decoding, and relicensing changes that postdate the CVPR artifact.
- **Our decision:** pin current main for initial import/inference work because it contains the upstream memory fix important for a 16 GB GPU. Treat it as a maintained implementation, not the exact paper-era revision.
- **Still unresolved:** whether a paper-era commit is useful as a controlled comparison. No tags identify one, and checkpoint compatibility must be established before switching revisions.

## 2026-07-20 — Phase 3 checkpoint and proof run

- **Official source:** public, ungated `facebook/VGGT-1B` revision `860abec7937da0a4c03c41d3c269c366e82abdf9`.
- **Our decision:** download only the official `model.pt`, not its duplicate safetensors representation, into ignored `local_assets/checkpoints/`.
- **Verified artifact:** 5,026,874,952 bytes; SHA-256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`.
- **Our decision:** load the explicit local path with `weights_only=True` and `mmap=True`; force Hugging Face offline flags; never call `from_pretrained` during the proof run.
- **Our decision:** use the smallest bundled official single-cartoon example and execute exactly one forward pass, including one center tracking query so every prediction head is exercised.
- **Result:** successful CUDA inference on RTX 5080; outputs and measurements saved under `outputs/predictions/phase3_first_run/`.

## 2026-07-20 — Phase 3.5 live notebook

- **Our decision:** create an unexecuted, presentation-oriented notebook that exposes each verified Phase 3 stage rather than invoking the script as a black box.
- **Our decision:** move checksum, asset verification, tensor schema, heatmap, PLY export, tracking-query, and dependency-free point-preview helpers into `src/vggt_seminar/live_demo.py`; keep the Phase 3 script using the shared helpers.
- **Our decision:** use OpenCV, Pillow, NumPy, and IPython display only; do not add Matplotlib, Plotly, pandas, or another visualization package.
- **Environment update:** after explicit user approval, `ipykernel 7.3.0` was installed from conda-forge into `vggt-seminar`; no global environment was modified. The pin is recorded in `requirements/notebook.txt`.

## 2026-07-20 — Phase 4 experiment framework

- **Our experiment:** evaluate robustness across controlled custom multi-view scenes and non-ideal variants; do not present the Phase 3 cartoon as experimental evidence.
- **Our decision:** require explicit scene manifests and preserve originals; derived degradations are generated only under output directories.
- **Our decision:** make the main notebook fail on empty/malformed scenes and gate model allocation/inference behind `APPROVE_INFERENCE=False`.
- **Our decision:** report qualitative evidence and lightweight internal diagnostics without calling them ground-truth accuracy.
- **Our decision:** create fine-tuning feasibility criteria but perform no fine-tuning until a repeated failure and valid evaluation design justify it.

## 2026-07-20 — Phase 5 ETH3D subset

- **Our decision:** integrate only the ETH3D high-resolution training scenes `delivery_area` and `courtyard` for an indoor/outdoor seminar subset.
- **Acquired modalities:** undistorted DSLR images with COLMAP calibration/poses, cleaned and evaluation laser scans, image masks, and occlusion data. Raw/distorted images, depth archives, rig data, test data, and all other scenes were omitted.
- **License:** the official ETH3D homepage states CC BY-NC-SA 4.0 and requests citation of Schops et al., CVPR 2017.
- **Boundary:** Phase 5 adds data access and experiment planning only; it does not run VGGT or implement quantitative metrics.

## 2026-07-20 — Phase 6 ETH3D smoke test

- **Our execution:** exactly one BF16 CUDA forward pass on `delivery_area`, using evenly-spaced original-order image indices 0/43 (`DSC_0675.JPG`, `DSC_0718.JPG`).
- **Result:** all required outputs and decoded cameras were 100% finite; inference took 1.3533 seconds with 5.26 GiB peak allocated and 5.88 GiB peak reserved VRAM.
- **Our fix:** because confidence has a floor/median of 1.0, the filtered point cloud now uses `confidence > median`; the saved tensors were reprocessed without another inference.
- **Caution:** the endpoint frames have weak visible overlap. Pipeline plumbing is ready, but overlap-aware nested subsets and coordinate-alignment rules must be frozen before the pilot matrix is interpreted.
# Phase 6.1: freeze overlap-aware nested ETH3D inputs

- Selected `D_hybrid_pose_feature` after pose, ORB-match, RANSAC-inlier, and contact-sheet review.
- Frozen deterministic nested 2/4/6/8/10-view subsets for `delivery_area` and `courtyard` in `phase6_1_overlap_aware_frames.yaml`.
- The notebook must fail closed when a frozen scene/count is absent; sequential/evenly-spaced selection is not an acceptable fallback for this strategy.
- No VGGT checkpoint, forward pass, laser ground truth, or pilot experiment was used in Phase 6.1.

## 2026-07-20 — Phase 6.2 overlap-aware S2 smoke test

- **Our execution:** exactly one BF16 CUDA forward pass on frozen `delivery_area` S2 indices `[0, 6]`; no fallback or other condition ran.
- **Observed:** all required tensors were 100% finite; inference took 0.9057 seconds with 5.26/5.88 GiB peak allocated/reserved VRAM; `confidence > median` retained 50.0% of points.
- **Protocol validation:** saved contact sheets show materially more shared scene content than Phase 6 endpoints `[0, 43]`, while direct and unprojected geometry remain structurally coherent.
- **Decision:** readiness A—scientifically ready for the frozen delivery-area view-count pilot, but that pilot still requires explicit approval.

## 2026-07-20 — Phase 7 delivery-area view-count pilot

- **Execution:** reused strictly validated Phase 6.2 S2 and ran exactly four new original-order BF16 CUDA forwards for frozen S4/S6/S8/S10, loading the model/checkpoint once.
- **Observed:** all five configurations are fully finite; peak allocated VRAM grew from 5.26 to 6.60 GiB and reserved VRAM to 9.23 GiB. Confidence increased overall with an S8 dip; camera-path extent peaked at S8 and changed at S10 in arbitrary unaligned scale.
- **Boundary:** this is descriptive single-scene evidence, not metric ETH3D accuracy or a repeated performance benchmark.
- **Decision:** delivery-area outputs are coherent enough to justify the identical courtyard pilot, subject to explicit approval.

## 2026-07-20 — Phase 8 courtyard and two-scene pilot

- **Execution:** exactly five original-order BF16 CUDA forwards for frozen courtyard S2/S4/S6/S8/S10, with one shared model/checkpoint load and no delivery-area rerun.
- **Observed:** every output was fully finite; VRAM scaling matched delivery area exactly, confidence rose through S8 then softened slightly at S10, and courtyard geometry remained visibly coherent.
- **Comparison boundary:** courtyard confidence was higher at matched counts, but this is not evidence of objective scene difficulty or accuracy; predicted camera scales remain arbitrary and unaligned.
- **Decision:** the two-scene view-count study is complete enough for report writing.

## 2026-07-20 — Phase 9 report synthesis

- **Our decision:** build the seminar draft exclusively from validated Phase 3–8 saved artifacts; no VGGT import, checkpoint load, CUDA work, alignment, metric evaluation, or download is permitted.
- **Provenance:** delivery-area results use tracked commit `5ef68d3`; courtyard and cross-scene results use tracked commit `ccb5487`. Canonical source hashes are recorded in `report/data/report_provenance.json`.
- **Deliverables:** a ten-row unified dataset, eight report tables, ten report figures, complete Markdown source, editable DOCX, and an unexecuted saved-results notebook section.
- **Interpretation boundary:** confidence is model confidence rather than accuracy, timings are single runs without uncertainty estimates, and geometry remains in arbitrary unaligned scale.

## 2026-07-20 — Version 2 visual report

- **Our decision:** preserve Version 1 and create a separate A4, visual-first seminar report under `report/v2/`, using saved canonical tables, images, camera plots, and PLY files only.
- **Design:** use an original academic visual system with a hero composition, study dashboard, conceptual VGGT diagram, project pipeline, enlarged nested-frame/results panels, and compact main-body tables; exact numeric and provenance detail moves to a six-page supplement.
- **Saved-geometry rendering:** deterministic fixed-stride sampling, median centering, 97th-percentile radial normalization, saved RGB colors, two-pixel splats, fixed perspective/elevation, and 80 azimuth frames. These choices affect presentation only and retain the arbitrary-unaligned-units warning.
- **Scientific decision:** no further inference is required to satisfy the report redesign. Similarity alignment and quantitative ETH3D evaluation remain future work, not part of Version 2.

## 2026-07-29 — Version 3 synthetic dataset

- **Our decision:** add TartanAir V2 `ArchVizTinyHouseDay/easy/P000/lcam_front`
  as the initial synthetic evaluation scene. The VGGT paper notes do not list
  TartanAir among the training sources; this improves independence relative to
  Kubric or HyperSim, but does not prove absence from unreported training data.
- **Acquired:** official front RGB and depth archives from
  `theairlabcmu/tartanair2`, 703,890,455 compressed bytes total. The archives
  contain P000-P006 because that is the smallest official archive unit; V3 is
  restricted to P000.
- **Locally verified:** P000 contains 116 matched 640 x 640 RGB frames, encoded
  metric-depth frames, and `xyz + xyzw` poses. Archive sizes and SHA-256 hashes
  are frozen in `configs/datasets/tartanair_v3_pilot.yaml`.
- **Boundary:** acquisition and planning only. No VGGT import, checkpoint load,
  CUDA work, alignment, or inference was performed. Frame selection, coordinate
  validation, and an S2 smoke run remain gated steps.

## 2026-07-29 — Version 3 assignment traceability and team

- **Team:** the V3 report will credit Peleg Shpitzer and Razi Mreeh.
- **Our decision:** include Ilan's original Hebrew instructions and a
  requirement-to-evidence table in V3 rather than leaving assignment compliance
  implicit.
- **Fine-tuning design:** reserve TartanAir P000 as an untouched test trajectory,
  use P001-P005 only as candidate training data, and P006 for validation/model
  selection. The first candidate freezes the aggregator and adapts camera/depth
  heads without tracking.
- **Boundary:** this records a proposed controlled experiment, not a completed
  fine-tune. Training-code audit, a no-step memory probe, an explicit resource
  budget, and separate execution approval are required before training.

## 2026-07-29 — Version 3 measured baseline and adaptation

- **Frozen test:** TartanAir P000 frames 20-29, nested S2/S4/S6/S8/S10. The
  selected contiguous window has at least 136 adjacent RANSAC inliers and
  1.91 m of ground-truth camera motion.
- **Pretrained result:** scale-aligned depth AbsRel increases from 3.14% at S2
  to 6.89% at S10; delta-1 falls from 0.978 to 0.949. Mean camera rotation
  error stays below 0.77 degrees. Confidence/error Spearman changes from
  -0.641 at S2 to +0.066 at S10.
- **Execution correction:** an initial diagnostic run disabled memory-efficient
  SDPA and produced an S10 OOM plus pathological S6/S8 timings. It is preserved
  but excluded. Canonical baseline results are the fresh, consistent
  `v3_tartanair_pretrained_20260729_sdpa_corrected` run.
- **Fine-tuning:** froze the aggregator and updated camera/depth heads for 30
  two-view steps on P001-P005. P006 selected step 15. The forward/backward probe
  used 5.95 GiB allocated and 6.52 GiB reserved on RTX 5080; adaptation took
  25.8 seconds.
- **Held-out decision:** the adapted heads are not promoted. They slightly
  reduce RMSE but worsen AbsRel, delta-1, and camera error; S10 AbsRel rises to
  12.79%. This is reported as measured over-specialization.
- **Deliverable:** eight-page V3 DOCX/PDF credited to Peleg Shpitzer and Razi
  Mreeh, including Ilan's Hebrew brief. Microsoft Word export and full-page PNG
  inspection passed after one references-numbering correction.

## 2026-07-29 — Submission-facing seminar report

- **Framing:** the submission is presented as a controlled VGGT study rather
  than a development chronology. The assignment brief remains in project
  documentation but is not reproduced in the visible report.
- **Evidence boundary:** canonical ETH3D and TartanAir results are unchanged.
  The report build recomputes quantitative metrics from saved outputs and runs
  no checkpoint loading, adaptation, dataset acquisition, or VGGT forwards.
- **Additional generalization check:** no second independent TartanAir
  environment was local, so the planned S2/S6/S10 check was not executed.
  Office/Data_easy/front is recorded as the recommended follow-up; it is not
  presented as a result.
- **Adaptation interpretation:** the selected step-15 heads remain a controlled
  negative result. A saved-output depth-range diagnostic suggests that the
  small RMSE reduction is concentrated at far range while near relative error
  worsens; this is reported as diagnostic evidence, not causal proof.
- **Deliverable:** Peleg Shpitzer and Razi Mreeh are credited. The main
  narrative occupies 12 pages, followed by one back-matter page. DOCX/PDF,
  source Markdown, validation data, and full-page renders are stored under
  `report/v4/`.
