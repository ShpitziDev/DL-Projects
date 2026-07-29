# Paper notes

Primary source: Wang et al., “VGGT: Visual Geometry Grounded Transformer,” CVPR 2025, pp. 5294–5306. Page references below use PDF pages (1-based); section/table references are preferred where stable.

## Source labels

- **[Authors]** Statement or result reported in the paper/supplement.
- **[Code]** Verified in the current official repository documentation/source available online on 2026-07-20.
- **[Local]** Measured in this workspace.
- **[Assumption]** Working hypothesis, not yet verified.
- **[Open]** Requires later code inspection or experiment.

## Problem and motivation

**[Authors]** VGGT asks whether a single feed-forward network can infer the major 3D properties of a scene from one to hundreds of same-scene RGB views, reducing dependence on iterative SfM, global alignment, and bundle adjustment (Sec. 1, pp. 1–2). Earlier learned systems tend to specialize by task or operate pairwise; VGGT aims for a shared, general-purpose 3D backbone.

## Main contributions

**[Authors]** (1) a unified transformer for cameras, depth, point maps, and tracks; (2) competitive or state-of-the-art results across camera estimation, multi-view depth, reconstruction, and tracking; (3) features useful for downstream non-rigid tracking and novel-view synthesis; and (4) fast direct inference with little 3D-specific inductive bias beyond alternating attention (Sec. 1).

## Inputs, outputs, and inference

Input is one or more RGB images of a common scene. Image order matters because predictions are normalized to the first camera frame. Outputs are camera tokens decoded to extrinsics/intrinsics, per-pixel depth and confidence, per-pixel point maps and confidence, and point tracks/visibility queried from dense features (Sec. 3; supplement pp. 1–2). **[Authors]** Directly unprojecting predicted depth with predicted cameras can be more accurate than the dedicated point-map head (Sec. 1).

**[Code]** The official quick start exposes `pose_enc`, `depth`, `depth_conf`, `world_points`, `world_points_conf`, `images`, and tracking via a separate query path. **[Open]** Confirm exact tensor shapes and all intermediate-feature hooks at the pinned commit.

## Architecture

**[Authors]** DINOv2 patch tokens feed an aggregator that alternates frame-wise self-attention (within each image) with global self-attention (across all image tokens). Camera and register tokens are included. Camera parameters are iteratively predicted from camera tokens. Dense DPT-style heads decode depth and point maps using intermediate transformer layers. A CoTracker-related tracking head consumes dense features and query points (Sec. 3; Fig. 2; supplement Sec. B).

### Alternating attention

Frame attention scales independently per frame and global attention communicates across views. The alternation supplies the model’s main explicit multi-view structure while retaining a standard transformer design. **[Authors]** Ablation favors alternating attention over global-only and other arrangements (Sec. 4.5). **[Open]** Measure memory scaling with view count on the 16 GB GPU.

### Prediction heads

- Camera head: iterative token-based pose encoding for extrinsics and intrinsics.
- Depth head: DPT dense decoder, predicts depth and uncertainty/confidence.
- Point-map head: DPT dense decoder, predicts world-frame 3D per pixel and confidence.
- Tracking head: query-conditioned iterative tracker with visibility estimates.

## Training objectives and data

**[Authors]** Multi-task training combines camera, depth, point-map, and tracking objectives (paper Eq. 1; supplement Sec. B). Dense losses include confidence-weighted regression and gradient terms; tracking uses correspondence regression plus visibility BCE. Ground truth is expressed in the first-camera coordinate system and scale-normalized (supplement pp. 1–2).

Training combines Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Habitat, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin, and an artist-created synthetic asset set (supplement p. 2). **[Authors]** Training used 160k iterations on 64 A100 GPUs for nine days.

## Evaluation tasks and metrics

- Camera pose: relative rotation/translation accuracy and AUC at angular thresholds on benchmarks including Co3D, RealEstate10K, ScanNet, and IMC; optional BA is reported separately.
- Multi-view depth: standard scale-aligned depth metrics and completeness/accuracy-style measures depending on dataset.
- Point clouds: reconstruction accuracy/completeness and F-score-style metrics.
- Tracking: TAP-Vid-style average Jaccard, position accuracy, and occlusion accuracy.
- Downstream tasks: non-rigid tracking and feed-forward novel-view synthesis.

**[Open]** Freeze exact metric definitions and evaluation splits from the pinned evaluation code before quantitative work.

## Ablations

**[Authors]** The paper studies task co-training, attention design, image tokenization, model scale, training data/scale, and inference choices. DINOv2 tokenization trains more stably and performs better than a learned 14×14 convolutional patchifier (supplement p. 5). Joint related predictions generally improve performance, even when some outputs are redundant. **[Open]** Transcribe exact table values only for comparisons we actually reproduce.

## Downstream fine-tuning

**[Authors]** VGGT features improve non-rigid point tracking and feed-forward novel-view synthesis (Sec. 4). **[Code]** Training code released after publication includes a Co3D fine-tuning example that freezes the aggregator and suggests camera/depth heads when memory is limited. This post-publication implementation describes itself as a reimplementation, not necessarily the original training stack.

## Reported strengths

- Unified outputs and cross-task sharing.
- One-to-many-view operation, including surprising zero-shot single-view behavior.
- Strong direct camera and geometry predictions with fast feed-forward inference.
- Broad generalization from diverse real and synthetic training data.
- Useful geometry-aware features for downstream tasks.

## Reported limitations

**[Authors]** Performance can degrade with limited overlap, difficult appearance, motion/dynamics, and large view counts due to global-attention memory. Predictions use a learned canonical scale/reference convention rather than solving gauge freedom explicitly. Direct inference can still benefit substantially from BA for demanding camera estimation. **[Open]** Collect the exact limitation wording from the rendered supplement figures/captions during the next artifact review.

## Claims to verify locally

1. All principal outputs are accessible and internally consistent.
2. Depth-plus-camera unprojection is often better than direct point maps.
3. Input order/first frame changes the canonical reconstruction but should preserve relative geometry.
4. Runtime is seconds or less for small sets, with memory increasing sharply by view count.
5. High-overlap static scenes are robust; low overlap, repeated texture, reflective/transparent surfaces, and dynamics expose failures.
6. Single-view inference works without duplicating the image.
7. Confidence correlates with actual error or visible failure.

## Evidence status

No local VGGT result exists yet. All performance statements above are author-reported unless explicitly tagged **[Code]** or **[Local]**.

## Local artifact and code reconciliation (2026-07-20)

- **[Paper]** Local authoritative paper inspected: 13 pages.
- **[Supplement]** Local authoritative supplement inspected: 8 pages.
- **[Code]** Pinned maintained implementation uses 1024-wide tokens, 16 heads, 24 frame and 24 global attention modules, DINOv2 ViT-L/14, four register tokens, and DPT taps 4/11/17/23.
- **[Code]** Current main postdates publication and contains a memory fix plus released training reimplementation; its behavior must not be described as an untouched paper snapshot.
