# Version 3 synthetic-data plan

## Team and assignment fit

**Team:** Peleg Shpitzer and Razi Mreeh.

V3 is structured directly around Ilan's instructions:

| Instructor requirement | Evidence planned for V3 |
|---|---|
| Base the work on the presented paper | Reproduce and study VGGT using the official code and checkpoint |
| Define one shared direction for the pair | Joint focus on VGGT robustness and synthetic-domain adaptation |
| Try different inputs | ETH3D indoor/outdoor scenes plus TartanAir synthetic RGB sequences and controlled view counts |
| Find strengths and weaknesses | Measured accuracy, confidence calibration, resource scaling, failure cases, and qualitative panels |
| Fine-tuning is desirable if feasible | Gated frozen-backbone camera/depth-head adaptation with a held-out test trajectory |
| Submit a document or presentation | Preserve V2 and produce a self-contained V3 report credited to both partners |

The final report should include the instructor's original Hebrew brief, followed
by this traceability table, so it is immediately clear how the submission
answers the assignment.

## Decision

Add **TartanAir V2 / ArchVizTinyHouseDay / easy / P000 / lcam_front** as a
held-out synthetic evaluation scene. TartanAir is preferable to Kubric or
HyperSim for this extension because the VGGT paper notes list Kubric and
HyperSim among the training sources, while TartanAir is not listed. This is
evidence of a more independent evaluation domain, not proof that no unreported
training overlap exists.

The official archive granularity is larger than one trajectory. The acquired
RGB and depth archives contain seven trajectories, P000-P006, but the initial
protocol is restricted to P000. Raw archives and extracted files remain under
ignored `local_assets/datasets/tartanair_v2/`.

## What was acquired and verified

- Official Hugging Face dataset: `theairlabcmu/tartanair2`
- Environment/difficulty/camera: `ArchVizTinyHouseDay / easy / lcam_front`
- Two intact archives, 703,890,455 bytes total, with SHA-256 values frozen in
  `configs/datasets/tartanair_v3_pilot.yaml`
- P000 has 116 RGB frames, 116 metric-depth frames, and 116 seven-value poses
- RGB is 640 x 640; depth is float32 encoded in the four PNG bytes
- Pose rows are translation followed by quaternion in `xyzw` order
- TartanAir dataset license: CC BY 4.0; toolkit license: MIT

No VGGT checkpoint was loaded and no inference was run during acquisition.

## Scientific question

Does VGGT's qualitative behavior on the two real ETH3D scenes carry over to a
synthetic indoor scene when accuracy can be measured against exact depth and
camera poses?

The V3 contribution is therefore not “one more attractive reconstruction.” It
adds ground-truth error, confidence calibration, and a real-versus-synthetic
discussion to the report.

## Phased protocol

### V3-A — dataset adapter and audit (next)

1. Implement reusable TartanAir RGB, depth, pose, and intrinsic readers under
   `src/vggt_seminar/`; keep the notebook display-only.
2. Unit-test byte-exact depth decoding, pose convention conversion, frame
   correspondence, and project/unproject round trips.
3. Freeze a documented valid-depth mask before metrics; decoded samples contain
   very large far/background values that must not be silently treated as useful
   geometry.
4. Generate a contact sheet and pose-baseline table for P000 without VGGT.
5. Choose one high-overlap, non-degenerate 10-frame window and freeze nested
   S2/S4/S6/S8/S10 frame indices in a new experiment config.

Gate: reprojection tests pass and the frozen window visibly shares scene
content. This phase requires no checkpoint or CUDA work.

### V3-B — one-run smoke test

After explicit approval, run only frozen S2 in a unique output directory.
Record the resolved configuration, hashes, environment, model/checkpoint pins,
runtime, VRAM, tensor finiteness, and raw predictions.

Gate: finite predictions, valid coordinate conversion, and a successful
similarity alignment to the two GT camera centers.

### V3-C — nested quantitative pilot

If S2 passes and another execution is approved, run S4/S6/S8/S10 with one
shared model load. Never rerun S2. Save each condition separately and build one
combined table.

Primary metrics:

- camera rotation error in degrees;
- camera-center ATE and relative-pose translation error after Sim(3) alignment;
- scale-aligned depth AbsRel, RMSE, and delta-1 on valid pixels;
- confidence/error calibration by confidence bins and Spearman correlation.

Secondary diagnostics:

- valid-depth coverage;
- median and upper-tail errors;
- runtime and peak allocated/reserved VRAM;
- qualitative RGB/depth/point-map panels using fixed visualization settings.

Do not score raw translation or raw depth before alignment: VGGT reconstruction
scale is arbitrary. Do not describe confidence as accuracy.

### V3-D — report Version 3

Preserve V2. Create `report/v3/` with:

1. the existing real ETH3D two-scene view-count evidence;
2. a new synthetic ground-truth evaluation section;
3. a real-versus-synthetic comparison that separates confidence, accuracy,
   resource cost, and visual coherence;
4. a limitations box covering single synthetic scene, possible unreported
   training overlap, alignment dependence, and lack of repeated timing trials.

The title page and document metadata must credit **Peleg Shpitzer and Razi
Mreeh**. The report must reproduce the instructor's Hebrew instructions in a
short “Assignment brief” box and map every instruction to concrete evidence.

### V3-E — optional measured fine-tuning study

Fine-tuning is scientifically useful only as a pre-trained-versus-adapted
comparison. It must not replace the baseline evaluation.

Frozen split:

- **Test:** P000. Never used for training, checkpoint selection, or early
  stopping.
- **Candidate training:** P001-P005.
- **Validation:** P006.

Initial adaptation scope:

1. audit the pinned post-publication VGGT training implementation and record
   any difference from the CVPR system;
2. freeze the 1.2B-parameter aggregator;
3. train only the camera and depth heads; omit tracking and point-map losses for
   the first feasibility run;
4. run a one-batch mixed-precision memory probe with no optimizer step;
5. if it fits, obtain approval for a small, fixed step/time budget and save
   checkpoints in a unique ignored output directory;
6. select a checkpoint using P006 only;
7. compare the untouched official checkpoint and adapted checkpoint on the
   exact same frozen P000 S2/S4/S6/S8/S10 inputs and metrics.

Promotion criteria:

- the baseline must first expose a measurable error worth adapting;
- training and validation losses must be finite;
- the run must fit the 16 GB GPU with safety margin;
- P000 improvement must be reported together with regressions and qualitative
  failures, not only the best metric;
- because this is one environment, any improvement is domain adaptation, not a
  claim of general VGGT improvement.

If the memory probe fails, the report should still include the measured
feasibility result. The fallback is depth-head-only training or a smaller
adapter/LoRA investigation after verifying that the official module structure
supports it. No full-backbone fine-tuning is planned.

## Optional expansion, not currently acquired

If the single-scene result is informative, the best controlled extension is
the matching `ArchVizTinyHouseNight` front-camera pair, testing illumination
shift with the same modalities. It should be downloaded only after P000 metrics
work and with a separate approval and checksum manifest.

Kubric is better reserved for a later custom-generation experiment because it
was used in VGGT training and adds Blender/PyBullet generation complexity.
HyperSim is also a VGGT training source and is much larger, so it is a weaker
next download for this particular held-out evaluation goal.
