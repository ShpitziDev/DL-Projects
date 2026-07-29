# Pinned VGGT code inspection

## Provenance

- URL: `https://github.com/facebookresearch/vggt.git`
- Branch at clone: `main`
- Commit: `a288dd0f14786c93483e45524328726ab7b1b4ce`
- Commit date: 2026-05-18 20:39:35 -0700
- Tags/releases: none in cloned Git metadata
- Status after clone/import install: clean
- License: VGGT License v1; checkpoint-specific terms differ

## Core package

`vggt.models.vggt.VGGT` composes `Aggregator`, `CameraHead`, point/depth `DPTHead`s, and `TrackHead`. `PyTorchModelHubMixin.from_pretrained` is the checkpoint-loading entry point and must not be called until acquisition is approved. Manual README loading also points to a Hugging Face `model.pt` URL.

The forward dictionary contains `pose_enc`, iterative `pose_enc_list`, `depth`, `depth_conf`, `world_points`, `world_points_conf`, and, when query points are supplied, `track`, `vis`, and `conf`; evaluation mode also retains input `images`.

## Preprocessing and geometry

`load_and_preprocess_images` converts RGBA to RGB on white, rescales to 518, enforces multiples of 14, and offers crop/pad modes. Camera pose decoding utilities convert the 9D encoding into extrinsics/intrinsics. Geometry utilities can unproject depth with predicted cameras, which must remain distinct from direct point-head output.

## Entry points and optional surfaces

- Core: package imports and README quick start.
- Visualization: `visual_util.py`, Viser demo, track visualization utilities.
- Demo: Gradio and Viser scripts.
- COLMAP/BA: `demo_colmap.py` plus pycolmap, pyceres, and LightGlue.
- Training: post-publication `training/` reimplementation with Co3D example and configurable frozen aggregator.
- Evaluation: no general evaluation package on main; README points to a separate evaluation branch for Co3D camera reproduction.

## Platform risks

Core imports work on Windows. Shell preprocessors under training (for example `vkitti.sh`), multi-GPU launch assumptions, pyceres/pycolmap, LightGlue Git installation, and some multiprocessing paths require later Windows-specific validation. PyTorch SDPA is sufficient for core attention, so native-Windows `flash_attn` compilation is unnecessary.

## Paper versus current main

Current main is not a paper snapshot: training code, commercial relicensing, DPT frame chunking, and the May 2026 intermediate-tensor memory fix were added later. Use current main for the first hardware-constrained study, record the commit on every run, and attribute code-era improvements separately from paper claims.
