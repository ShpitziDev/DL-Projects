# VGGT Seminar Project

Reproducible Deep Learning Seminar investigation of **VGGT: Visual Geometry Grounded Transformer** (Wang et al., CVPR 2025). The project wraps the official implementation, evaluates varied inputs on ETH3D and TartanAir, and documents a bounded domain-adaptation experiment.

## Final submission

- [Submission-ready PDF](report/submission/Peleg_Shpitzer_Razi_Mreeh_VGGT_Seminar_Report.pdf)
- [Editable DOCX](report/submission/Peleg_Shpitzer_Razi_Mreeh_VGGT_Seminar_Report.docx)
- [Markdown source](report/submission/Peleg_Shpitzer_Razi_Mreeh_VGGT_Seminar_Report.md)
- [Validation record](report/submission/validation/final_submission_report.json)

The 16-page report is credited to **Peleg Shpitzer** and **Razi Mreeh**. It includes the reproduced inference path, two enlarged ETH3D evidence pages, quantitative TartanAir evaluation, properly rendered metric definitions, and the held-out adaptation result.

## Current Reproduction Status

The reproduction and frozen two-scene view-count study are complete:

- Native Windows Micromamba environment: created with Python 3.11.15
- PyTorch 2.13.0 + torchvision 0.28.0, CUDA 13.0 wheels: validated on RTX 5080
- Official VGGT repository: pinned at `a288dd0f14786c93483e45524328726ab7b1b4ce`
- Core VGGT imports: validated in forced Hugging Face offline mode
- Official research checkpoint: downloaded locally, SHA-256 verified, never committed
- Official inference: exactly one successful offline CUDA run on one bundled example
- Frozen protocol: `eth3d-overlap-aware-nested-v1`
- ETH3D study: `delivery_area` and `courtyard`, each at 2/4/6/8/10 views
- Final report: PDF, DOCX, Markdown, figures, and validation under `report/submission/`
- Synthetic evaluation: TartanAir P000 at nested S2/S4/S6/S8/S10 view counts
- Bounded adaptation: camera/depth heads trained on P001-P005; P006 selected step 15; P000 remained untouched
- Validation: 88 download-free repository tests pass

First-run outputs are under `outputs/predictions/phase3_first_run/`. The run processed one 518x518 image in 1.346 seconds on the RTX 5080 and recorded 5,741,071,360 bytes peak allocated GPU memory.

## Phase 3.5 live notebook

Open `notebooks/01_live_vggt_inference_demo.ipynb` in VS Code and select the existing interpreter at `.micromamba/envs/vggt-seminar/python.exe`. The notebook exposes verification, preprocessing, model construction, local checkpoint loading, GPU transfer, one forward pass, every output head, visualization/export, timing, and cleanup as separate teaching cells. It is saved with no outputs and no execution counts.

The environment includes `ipykernel 7.3.0` from conda-forge. In VS Code, select `vggt-seminar (Python 3.11.15)` or the interpreter at `.micromamba/envs/vggt-seminar/python.exe`.

## Phase 4 main experiment notebook

`notebooks/02_vggt_multi_input_experiments.ipynb` is the primary seminar workflow. It inventories `data/custom_inputs`, requires a valid scene manifest, constructs controlled frame-count/order/degradation plans, and keeps inference behind `APPROVE_INFERENCE=False`. It never falls back to the Phase 3 cartoon. Capture guidance is in `docs/input_capture_guide.md`; the central question and evidence boundaries are in `docs/phase4_research_question.md`; the controlled comparison protocol is in `docs/phase4_experiment_taxonomy.md`.

## ETH3D benchmark subset

Phase 5 adds two official ETH3D high-resolution training scenes: indoor `delivery_area` and outdoor `courtyard`. Downloaded data and its local manifest live under ignored `local_assets/datasets/eth3d/`; tracked documentation is in `docs/eth3d_dataset.md` and the future metric design is in `docs/eth3d_evaluation_plan.md`.

In `notebooks/02_vggt_multi_input_experiments.ipynb`, use `INPUT_SOURCE = "eth3d"` with `SCENE_NAME = "delivery_area"`, or use `INPUT_SOURCE = "custom"` for a manifest-backed scene under `data/custom_inputs`. A future adapter should follow the same source-selection branch and return an ordered image list plus metadata without adding dataset-specific logic to inference. To add another ETH3D scene later, obtain explicit download approval, preserve the official scene layout under `local_assets/datasets/eth3d/<scene>/`, and append its acquisition record to the ignored dataset manifest.

## Phase 6 smoke test

Exactly one approved two-view `delivery_area` inference completed through `scripts/run_phase6_eth3d_smoke.py`. The ignored artifacts are under `outputs/predictions/phase6_eth3d_smoke/delivery_area/2_views_evenly_spaced_original/`; measured results and limitations are documented in `docs/phase6_eth3d_smoke_test.md`. The main experiment notebook remains clean and unexecuted. Do not rerun the smoke-test command because its runner refuses to overwrite the completed directory.

## Completed view-count study and report

Phases 7 and 8 completed the frozen `delivery_area` and `courtyard` pilots. Canonical aggregates remain under `outputs/experiments/phase7_delivery_area_view_count/` and `outputs/experiments/phase8_courtyard_view_count/`; the matched cross-scene comparison is under the latter directory's `comparisons/` folder. Phase 9 consolidates those saved artifacts into `report/data/`, eight report tables, ten report figures, and the complete seminar draft. `scripts/build_phase9_report.py` is deterministic and saved-artifact-only: it validates hashes and contracts, never imports VGGT or Torch, and never loads the checkpoint.

The final section of `notebooks/02_vggt_multi_input_experiments.ipynb` displays the unified dataset, selected tables, and selected figures without inference. The notebook is intentionally unexecuted.

### Visual report Version 2

The submission-ready visual redesign is `report/v2/vggt_seminar_report_v2.pdf`, with an editable DOCX, Markdown source, and `report/v2/supplementary/vggt_supplementary.pdf`. Original diagrams, larger result figures, deterministic saved-PLY keyframes, and 8-second GIF/MP4 rotations are stored below `report/v2/`. Rebuild the animations with `scripts/build_v2_animations.py`, then rebuild the documents with `scripts/build_phase10_report_v2.py`; both scripts consume saved artifacts only and contain no inference path.

## Layout

- `src/vggt_seminar/`: reusable project code
- `configs/`: defaults and future experiment overrides
- `docs/`: paper review, decisions, scope, and plans
- `external/`: third-party checkouts (ignored except its README)
- `data/`: trackable sample/custom input structure; downloaded data ignored
- `outputs/`: generated runs and machine reports (ignored except README)
- `notebooks/`: thin exploration notebooks
- `report/`: final-report material
- `tests/`: download-free unit tests

## Native Windows environment

Micromamba 2.8.1 is stored locally in ignored `.tools/`; the named environment is under ignored `.micromamba/envs/vggt-seminar`.

Recreate from a PowerShell terminal at the repository root:

```powershell
.\.tools\micromamba.exe create -y -r .micromamba -n vggt-seminar -c conda-forge python=3.11 pip=25.1
.\.tools\micromamba.exe run -r .micromamba -n vggt-seminar python -m pip install -r requirements\pytorch-cu130.txt
.\.tools\micromamba.exe run -r .micromamba -n vggt-seminar python -m pip install -r requirements\vggt-inference.txt -r requirements\base.txt
.\.tools\micromamba.exe run -r .micromamba -n vggt-seminar python -m pip install -e . --no-deps
```

Activate interactively after initializing the shell, or avoid shell mutation with `micromamba run`:

```powershell
.\.tools\micromamba.exe shell hook -s powershell | Out-String | Invoke-Expression
micromamba activate D:\cv_sem\.micromamba\envs\vggt-seminar
```

## Official VGGT integration

The official checkout is an ignored normal clone, not a submodule. Recreate and pin it with:

```powershell
git clone https://github.com/facebookresearch/vggt.git external/vggt
git -C external/vggt checkout --detach a288dd0f14786c93483e45524328726ab7b1b4ce
.\.tools\micromamba.exe run -r .micromamba -n vggt-seminar python -m pip install -e external\vggt --no-deps
```

Pin metadata is tracked in `external/VGGT_PIN.json`. The original non-commercial research checkpoint `facebook/VGGT-1B/model.pt` is stored under ignored `local_assets/checkpoints/` with a tracked checksum manifest. Licensing differs from `facebook/VGGT-1B-Commercial`.

## Verification

```powershell
.\.tools\micromamba.exe run -r .micromamba -n vggt-seminar python scripts\check_environment.py
.\.tools\micromamba.exe run -r .micromamba -n vggt-seminar python scripts\validate_cuda.py
.\.tools\micromamba.exe run -r .micromamba -n vggt-seminar python scripts\verify_installation.py
.\.tools\micromamba.exe run -r .micromamba -n vggt-seminar python -m pytest
```

The environment checker always writes `outputs/environment/environment_report.json` and is safe when Torch is absent.

The exact Phase 3 inference command was:

```powershell
.\.tools\micromamba.exe run -r .micromamba -n vggt-seminar python scripts\run_phase3_first_inference.py
```

The runner refuses to overwrite the completed output directory. Do not rerun it without defining a new approved run.

## Reproducibility principles

All paths are repository-relative, every run receives a unique directory and frozen configuration, inputs remain immutable, third-party source is not edited, and paper-reported results are never presented as local measurements.

## Next step

Version 3 is complete under `report/v3/`. It adds a checksum-verified TartanAir
synthetic benchmark, a frozen P000 S2-S10 baseline, confidence calibration,
an RTX 5080 frozen-aggregator fine-tuning feasibility study, and an honest
held-out negative adaptation result. The final DOCX/PDF credits Peleg Shpitzer
and Razi Mreeh and maps Ilan's Hebrew instructions to measured evidence.
