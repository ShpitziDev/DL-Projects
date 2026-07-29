# VGGT checkpoint notes

## Official source

- Hugging Face repository: `facebook/VGGT-1B`
- Repository revision inspected: `860abec7937da0a4c03c41d3c269c366e82abdf9`
- Official code pin: `a288dd0f14786c93483e45524328726ab7b1b4ce`
- Selected file: `model.pt`
- Published size: 5,026,874,952 bytes
- Published SHA-256: `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`
- License shown by the model card: CC-BY-NC-4.0; the original research checkpoint is non-commercial.
- Access: public and ungated; no authentication required.

The repository also contains `model.safetensors` (5,026,367,224 bytes), plus small `config.json` and README files. We intentionally download only `model.pt` because the pinned official README documents direct loading of that exact file. Downloading both weight formats would duplicate approximately 5 GB without helping this proof-of-execution phase.

## Loading behavior

`VGGT.from_pretrained("facebook/VGGT-1B")` delegates to `PyTorchModelHubMixin` and may populate the default Hugging Face user cache. Phase 3 does not use that route. `scripts/download_checkpoint.py` downloads the selected file into `local_assets/checkpoints/`, validates size and SHA-256, and records a manifest. `scripts/run_phase3_first_inference.py` constructs `VGGT`, loads the local state dictionary with `torch.load(..., weights_only=True, mmap=True)`, and never calls `from_pretrained`.

## Cache and offline support

The downloader uses `local_dir=local_assets/checkpoints`; Hugging Face transfer metadata is kept beside the local download rather than relying on the default user cache. The inference runner sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` before importing/loading VGGT. Because loading uses an explicit local file, inference is fully offline after acquisition.

## Phase 3 input

Use the smallest bundled official example:

`external/vggt/examples/single_cartoon/images/model_was_never_trained_on_single_image_or_cartoon.jpg`

It is part of the pinned source clone, so no sample or benchmark dataset download is required.
