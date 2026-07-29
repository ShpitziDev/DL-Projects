# Supplementary-material notes

Primary source: official CVPR 2025 supplementary PDF (8 pages). Labels follow `paper_notes.md`.

## Architecture and implementation

- **[Authors, p. 2]** 24 global-attention layers and 24 frame-wise-attention layers are used in alternation; total size is about 1.2B parameters. The “24 blocks” wording can be ambiguous, so the pinned code must establish whether its aggregator API counts pairs or individual attention layers.
- **[Authors, Sec. B / Fig. details]** Transformer feature width and head counts are architecture-specific; **[Open]** confirm exact values from the pinned `aggregator.py` rather than rely on memory or secondary summaries.
- **[Authors, p. 5]** Images are patchified at 14×14 resolution using pretrained DINOv2; this outperforms and stabilizes training versus a learned convolutional patchifier.
- **[Authors]** DPT dense heads fuse multiple intermediate aggregator layers. **[Open]** Record exact layer indices from pinned code; do not assume a layer list across repository revisions.
- **[Authors]** Camera/register tokens are processed with image tokens; dense tokens support depth, point maps, and tracking.

## Multi-task loss

**[Authors, supplement pp. 1–2]** The total objective combines camera, depth, point-map, and track losses. Dense geometry uses confidence-weighted regression, spatial-gradient consistency, and a log-confidence regularizer. Track regression is paired with visibility BCE following CoTracker2. Camera/geometry targets are expressed relative to the first camera and normalized by average 3D-point distance.

## Optimization and schedule

- AdamW, 160,000 iterations.
- Cosine learning-rate schedule, peak LR `2e-4`, 8,000-iteration warm-up.
- Gradient-norm clipping at 1.0.
- bfloat16 and gradient checkpointing.
- **[Authors, p. 2]** 64 NVIDIA A100 GPUs over nine days.

## Frame sampling and batches

**[Authors, p. 2]** Sample 2–24 frames from a randomly selected scene. The total is held at 48 frames per batch while frames per scene vary. Datasets receive different but roughly similar sampling weights; scenes are uniform within a selected dataset. Training sequences shorter than 24 frames are excluded. Frames too dissimilar to the tracking query may be excluded, and tracking loss is omitted if correspondence is invalid.

## Image preprocessing and augmentation

**[Authors, p. 2]** RGB, depth, and point maps are isotropically resized so the long side is 518 px. The short dimension is cropped around the principal point to 168–518 px in multiples of the 14 px patch size, producing aspect ratios around 0.33–1.0. Color jitter, Gaussian blur, and grayscale are applied, with aggressive color augmentation independently per frame to improve lighting robustness.

## Training data

Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Habitat, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin, and an artist-created synthetic set comparable in spirit to Objaverse (p. 2). Sources mix sensors, renderers, and SfM annotations.

## Runtime and memory

**[Authors]** Main-paper claims are under one second for representative input sets and seconds for up to hundreds of images, depending on task/hardware (main Fig. 1/Sec. 4). Supplement IMC reports about 0.2 s for feed-forward VGGT and 1.8 s with BA for its setup (supplement Table A, p. 4). These are not local expectations until input count, resolution, GPU, warm-up, precision, and measurement protocol are matched.

Global attention creates a view-count memory bottleneck. **[Code]** The repository announced a May 2026 fix for retained intermediate tensors, claiming roughly 2–3× more frames under the same memory budget. This postdates the paper, so repository commit and paper behavior must not be conflated.

## Stated limitations and cautions

- Reconstruction remains ambiguous up to global frame and scale; targets impose a learned convention.
- Geometry can require BA for top camera-pose accuracy.
- Dense global attention limits scaling.
- Tracking is trained from depth-derived correspondences and can be absent for invalid samples.
- Dataset overlap and benchmark split hygiene matter (supplement IMC discussion, pp. 2–3).
- **[Open]** Verify exact qualitative limitation wording from all rendered pages and record failure examples relevant to our taxonomy.

## Code-verified facts (online inspection, not a local checkout)

- `pyproject.toml` requires Python >=3.10 and dependencies including NumPy<2, Pillow, Hugging Face Hub, einops, safetensors, and OpenCV.
- `requirements.txt` currently pins Torch 2.3.1 and torchvision 0.18.1.
- Training README describes a reimplementation, a Co3D setup, four-GPU DDP example, frozen aggregator default, gradient accumulation controls, and optionally omitting the tracking head for limited memory.

## Assumptions/open questions

- Exact feature dimension, attention-head count, DPT tap layers, loss weights, and current memory behavior must be verified against the pinned checkout.
- Windows compatibility of optional demo, evaluation, BA, and training dependencies remains untested.
- Fine-tuning VRAM on a 16 GB RTX 5080 is unknown; no feasibility claim is made.

## Resolved from pinned official code

- **Official code:** feature dimension 1024; 16 attention heads; 24 frame blocks plus 24 global blocks arranged in alternating pairs.
- **Official code:** DINOv2 ViT-L/14 register-token backbone, four register tokens, and separate learned first-frame versus remaining-frame camera/register token banks.
- **Official code:** DPT taps are layers 4, 11, 17, and 23 after concatenating frame/global features.
- **Official code:** default preprocessing target is 518 and dimensions are multiples of 14; quick-start supports center-crop and square-pad modes.
- **Still unresolved:** exact correspondence between post-publication memory changes and paper timing/memory tables requires a controlled revision comparison, which is outside Phase 2.
