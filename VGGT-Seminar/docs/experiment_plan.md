# Provisional experiment plan

Do not run the full matrix initially. Begin with one official example, then a small pilot that verifies output capture and measurement quality.

## Taxonomy

1. **Views:** 1, 2, 4, 8, 16, then higher only if measured memory permits.
2. **Overlap:** high, moderate, low, intentionally none.
3. **Scenes:** indoor room; outdoor building; street/campus; isolated object; LEGO; reflective; transparent; low texture; repeated texture; painting/drawing/illustration.
4. **Quality:** blur, low light, compression, reduced resolution, exposure variation.
5. **Camera:** moderate and extreme viewpoint changes; orientation mixtures; fisheye/panorama as intentional unsupported inputs.
6. **Consistency:** static; moving people; moved object; non-rigid motion.
7. **Order:** shuffled order; alternate first frames; repeated images; unrelated accidental image.

## Required run record

Each run records: experiment ID, timestamp, hypothesis, scene category, image count, immutable input manifest and hashes, preprocessing, model/checkpoint/repository commit, GPU/driver/Torch, seed/device/precision, runtime protocol and values, peak VRAM, raw/derived output paths, metrics, qualitative observations, failure category, and conclusion. The resolved config is copied into the unique run directory.

## Candidate quantitative pilots

- Synthetic or calibrated small scene: camera relative-pose error and depth alignment.
- Controlled degradations of identical inputs: stability/error deltas.
- Permutations: align reconstructions and compare pairwise camera/point geometry.
- Confidence calibration: bin confidence and plot empirical geometry error.

## Fine-tuning feasibility (preliminary; no implementation)

Options, from least to most risky:

1. Freeze aggregator and adapt camera/depth heads to a narrowly defined domain.
2. Fine-tune a small downstream tracking or prediction head on existing features.
3. Small custom-domain experiment with most backbone layers frozen.
4. LoRA/PEFT only after checking whether attention module structure, training code, and memory savings justify integration.

The official post-publication training reimplementation supplies losses and a Co3D example; it defaults to a frozen aggregator and suggests omitting tracking under constrained memory. However, its documented command assumes four GPUs, dataset preparation is substantial, and 16 GB single-GPU feasibility is unverified. Gradient accumulation reduces activation pressure from effective batch size but does not make a 1.2B model’s optimizer/activation footprint disappear. Fine-tuning is optional under the assignment and should proceed only when it tests a meaningful failure hypothesis with a legally usable, small dataset and measured memory headroom.

## Version 3 synthetic pilot

The selected quantitative pilot is TartanAir V2
`ArchVizTinyHouseDay/easy/P000/lcam_front`. It adds exact pose and metric-depth
ground truth in a synthetic domain not listed in the paper's training-data
inventory. The frozen acquisition contract is
`configs/datasets/tartanair_v3_pilot.yaml`; the staged execution and reporting
protocol is `docs/v3_synthetic_dataset_plan.md`.

Do not run VGGT until the dataset adapter passes reprojection tests and a
high-overlap nested S2/S4/S6/S8/S10 window is frozen. S2 requires its own
explicit inference approval; later counts require a separate promotion.

## Promotion gate

Do not start fine-tuning until: official inference is reproducible; a failure is demonstrated; a metric exists; license/data rights are clear; pinned training code is audited; a one-batch memory probe succeeds; and a stop budget is approved.
