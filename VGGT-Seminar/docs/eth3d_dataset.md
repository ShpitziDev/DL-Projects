# ETH3D dataset integration

## Overview

This workspace contains a deliberately small subset of the ETH3D high-resolution multi-view stereo **training** data. ETH3D combines calibrated high-resolution photographs with high-precision laser-scan geometry and official reconstruction-evaluation assets.

Official sources:

- Website and license statement: <https://www.eth3d.net/>
- Per-scene downloads: <https://eth3d.ethz.ch/datasets>
- Format and evaluation documentation: <https://www.eth3d.net/documentation>

## Why ETH3D was selected

VGGT reports point-map accuracy, completeness, and overall error on ETH3D using 10 randomly sampled frames, similarity alignment, and official masks. The same benchmark is also used for architecture and multi-task ablations. ETH3D was not named in the VGGT supplement's training-data list, has both indoor and outdoor high-resolution scenes, supplies calibrated cameras and laser-scan geometry, and is practical to use scene by scene.

## Included scenes and modalities

| Scene | Domain | Images | Resolution | Included modalities |
|---|---|---:|---:|---|
| `delivery_area` | Indoor | 44 | 6208 x 4135 | Undistorted JPEGs, COLMAP calibration/poses, cleaned and evaluation scans, 42 supplied image masks, occlusion data |
| `courtyard` | Outdoor | 38 | Four calibrated canvases, 6198–6208 x 4129–4135 | Undistorted JPEGs, COLMAP calibration/poses, cleaned and evaluation scans, 38 supplied image masks, occlusion data |

The ignored machine manifest at `local_assets/datasets/eth3d/dataset_manifest.json` records exact URLs, byte counts, locally computed SHA-256 hashes, extraction sizes, and file counts. Undistortion changes the canvas dimensions from the original Nikon capture size, so the manifest records locally observed undistorted dimensions rather than the nominal 6048 x 4032 camera size.

## Intentionally omitted

All other training and test scenes, benchmark-wide archives, distorted JPEGs, raw NEF images, raw scans, pre-rendered depth archives, multi-camera rig data, and two-view data were not downloaded. Depth can later be rendered from the supplied aligned scan/occlusion representation if the evaluation protocol requires it.

## License and citation

The official ETH3D homepage states that its data is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. This seminar use must remain non-commercial, attribute ETH3D, apply share-alike requirements when distributing covered adaptations, and not commit or redistribute the downloaded data from this repository.

The official site asks users of the stereo data to cite:

> Thomas Schops, Johannes L. Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, Andreas Geiger. “A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos.” CVPR, 2017.

Use the BibTeX supplied by the official ETH3D homepage in the final report.

## Local organization

```text
local_assets/datasets/eth3d/
├── dataset_manifest.json
├── courtyard/
│   ├── archives/
│   ├── images/dslr_images_undistorted/
│   ├── dslr_calibration_undistorted/
│   ├── masks_for_images/dslr_images/
│   ├── scan_clean/
│   ├── dslr_scan_eval/
│   └── occlusion/
└── delivery_area/
    └── ... same modality layout ...
```

The entire `local_assets/datasets/` directory is ignored by Git. Raw downloaded and extracted files must remain immutable.

## Seminar use

`src/vggt_seminar/eth3d.py` discovers scenes, parses the official COLMAP text calibration, orders images consistently, exposes masks and scan metadata, and constructs controlled frame subsets. The main experiment notebook can switch between `INPUT_SOURCE = "custom"` and `INPUT_SOURCE = "eth3d"`.

Phase 5 prepares experiment plans only. Quantitative evaluation, VGGT inference, bundle adjustment, and fine-tuning are separate future phases requiring explicit execution.
