# Dataset study for VGGT evaluation

Research date: 2026-07-20. No dataset was downloaded during this study.

## Decision

**Primary dataset: ETH3D high-resolution multi-view training data.** Start with one indoor and one outdoor scene, using the undistorted JPEG images, calibration, cleaned laser scan, evaluation data, and occlusion information. ETH3D is the best balance of direct relevance to the VGGT paper, real-world diversity, quantitative geometry, official evaluation tools, manageable selective download, and seminar clarity.

**Optional secondary dataset: DTU MVS.** Use only a small standard evaluation subset if time and storage permit. DTU provides a controlled object-scale complement and VGGT reports dense MVS results on it, but it is less representative of in-the-wild scenes and its packaging/evaluation conventions require more preparation.

The recommendation is deliberately for the **training portions**, whose ground truth can be evaluated locally. Hidden test ground truth and leaderboard submission are unnecessary for this seminar.

## Evidence labels and caveats

- **Paper:** stated or reported by Wang et al.
- **Official:** stated by a dataset owner or official benchmark site.
- **Plan:** our proposed use or runtime estimate; not a measured result.
- **Unknown:** not clearly granted or documented by the official public page; verify before acquisition.

“Ground truth” is not uniform. ETH3D and Tanks and Temples use laser scans; DTU uses structured-light scans; ScanNet++ includes registered laser scans; CO3D and MegaDepth annotations are largely reconstruction-derived. Estimated storage varies by modality and version. Download-page figures take precedence over estimates at acquisition time.

## VGGT paper audit

### Training data

The supplement lists Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Habitat, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin, and an artist-created synthetic asset dataset. It does **not** list ETH3D, DTU, Tanks and Temples, or ScanNet++ as training datasets. The paper says the learnable camera-pose methods were trained on CO3Dv2; it explicitly labels RealEstate10K as unseen. For ScanNet-1500 evaluation, corresponding ScanNet scenes were excluded from training.

### Direct evaluation

| VGGT task | Dataset | Paper protocol |
|---|---|---|
| Camera pose | CO3Dv2, RealEstate10K | 10 random frames; AUC@30 from relative rotation/translation accuracy |
| Camera pose, supplement | IMC | Phototourism pose AUC; direct VGGT and VGGT + BA |
| Multi-view depth | DTU | Accuracy, completeness, and overall/Chamfer-style average |
| Point-map reconstruction | ETH3D | 10 random frames; Umeyama alignment; official masks; accuracy/completeness/overall |
| Point-map ablations | ETH3D | Attention and multi-task ablations |
| Camera evaluation mentioned in supplement | ScanNet-1500 | Matching evaluation scenes excluded from training |
| Novel-view synthesis after adaptation | Google Scanned Objects | PSNR, SSIM, LPIPS |
| Dynamic tracking after adaptation | TAP-Vid Kinetics, RGB-Stacking, DAVIS | AJ, visible-point accuracy, occlusion accuracy |

### Qualitative-only material

The main paper and supplement show additional single-view and multi-view reconstructions without identifying every image as belonging to a named public benchmark. Figure 4 uses Google Scanned Objects for novel-view synthesis and Figure 5 visualizes rigid/dynamic tracking. These are useful illustrations, but they should not be reclassified as direct pretrained reconstruction benchmarks.

### Credibility implication

Using ETH3D or DTU strengthens the seminar because it connects our protocol to an exact evaluation task in the paper. ETH3D is particularly strong: the paper evaluates both final point maps and architecture/task ablations on it, and it was not listed among training sources. CO3D offers exact camera-pose comparability but is also training data, so it is weaker evidence of generalization unless a clean held-out protocol is enforced.

## Capability comparison

Legend: Y = supplied; P = partial/derived/available for some subsets; N = generally absent; H = hidden test ground truth.

| Dataset | Indoor | Outdoor | Calibrated cameras | Multi-view | Dense geometry | Point cloud | Mesh | Depth maps | Evaluation tools | Quantitative locally? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tanks and Temples | Y | Y | P | Y | Y/H | Y | P | N | Y, server + scripts | Y on training scenes |
| DTU MVS | Y | N | Y | Y | Y | Y | N/P | P/derivable | Y, standard scripts | Y |
| ETH3D high-res MVS | Y | Y | Y | Y | Y | Y | N/P | Y for training | Y, source + Windows binaries | Y |
| ScanNet++ | Y | N | Y | Y | Y | Y | Y | Y for iPhone stream | Task-dependent | Y, but integration-heavy |
| CO3Dv2 | P | P | Y, reconstruction-derived | Y | P | Y for subset | N | P/derived | Code and splits | Y for pose; geometry caveats |
| MegaDepth | P | Y | Y, SfM-derived | Y | P, MVS-derived | P | N | Y, derived | Community protocols | Y, but not sensor GT |
| RealEstate10K | Y | P | Y, SfM-derived | Y/video | N | N | N | N | Community protocols | Pose only |
| IMC | P | Y | Y/reference reconstructions | Y | P | P | N | N | Official challenge tooling | Pose only, split-dependent |
| ScanNet (not ++) | Y | N | Y | Y/RGB-D | Y | Y | Y | Y | Established | Y; training-overlap controls needed |

## Practical comparison

| Dataset | Typical scale | Download/storage burden | Download friction | VGGT fit | Current-literature use | Seminar verdict |
|---|---|---|---|---|---|---|
| ETH3D | 13 train + 12 test DSLR scenes; official train images 5.5 GB, GT 1.8 GB, occlusions 1.2 GB | Low–moderate; individual scenes available | Easy; direct 7z files | Excellent; exact paper point-map protocol | Common MVS generalization benchmark | **Primary** |
| DTU | 124 captured scenes, 80 used in original evaluation | Moderate to high; full releases/mirrors can be tens of GB | Moderate; legacy packaging and preprocessing | Excellent; exact paper dense-MVS task | Very common learned-MVS benchmark | **Secondary** |
| Tanks and Temples | 7 training, 8 intermediate, 6 advanced scenes | Moderate/high; 4K video or sampled images plus scans | Easy downloader; leaderboard rules add work | Good for reconstruction stress testing | Very common MVS benchmark | Reject as primary: hidden test GT and large scenes |
| ScanNet++ | 1,006 scenes with DSLR, iPhone, and laser modalities | Very high; full dataset is multi-terabyte class | Access application and modality tooling | Technically excellent, operationally excessive | Increasingly common indoor benchmark | Reject: disproportionate storage/preparation |
| CO3Dv2 | ~18.6k videos, 1.5M camera-annotated frames, 5,625 point-cloud videos | Very high for full release; subset still manageable | Downloader and metadata stack | Excellent camera/object input; trained-on distribution | Common object-centric benchmark | Reject as primary: training overlap and scale |
| MegaDepth | Internet landmark collections; full processed release commonly hundreds of GB; `megadepth1500` subset 1.3 GB | Low for 1500 subset, very high for full data | Mirrors/versions and Flickr provenance complicate use | Good for pose/depth stress tests | Very common matching/relative-pose data | Reject: pseudo-ground-truth and training overlap |
| RealEstate10K | ~10k real-estate videos | High full download; original video availability can be brittle | URL/video acquisition is fragile | Excellent unseen pose benchmark in paper | Still common in pose/NVS literature | Reject: no dense geometry ground truth |
| IMC | Challenge-dependent phototourism scenes | Moderate and split-specific | Kaggle/challenge conventions | Exact supplementary pose benchmark | Highly current for image matching/SfM | Reject: pose-only and protocol overhead |
| ScanNet | 1,513 RGB-D scans, 2.5M views | Large | License agreement/download script | Good indoor pose/depth/mesh | Very common | Reject: listed training data; ScanNet++ is higher fidelity |

“RTX 5080 suitability” is primarily governed by selected view count and resized input resolution, not total dataset size. All candidates can supply small batches. ETH3D’s 10-frame paper protocol and per-scene downloads make it easiest to bound on a 16 GB GPU.

## Dataset profiles

### 1. ETH3D - Multi-view Stereo Benchmark

- **Publication/citation:** Thomas Schöps et al., “A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos,” CVPR 2017.
- **Official source:** [ETH3D overview](https://eth3d.ethz.ch/overview), [datasets](https://eth3d.ethz.ch/datasets), and [documentation](https://www.eth3d.net/documentation).
- **License:** the official download/site terms must be accepted and rechecked before download; no broad SPDX-style grant was found in the reviewed documentation. Treat as research-only until verified.
- **Intended task:** calibrated two-view and multi-view stereo / dense 3D reconstruction.
- **Contents:** high-resolution DSLR images, COLMAP-format calibration, high-precision laser-scan point clouds, evaluation masks/occlusions, derived training depth, indoor and outdoor scenes, and official evaluators.
- **Size/difficulty:** official high-res training bundles list 5.5 GB undistorted JPEGs, 1.8 GB ground truth, and 1.2 GB occlusions; individual scenes are downloadable. 7z extraction is the main inconvenience.
- **VGGT evaluation:** directly supports the paper’s 10-view point-cloud accuracy/completeness/overall protocol after similarity alignment and masking. Camera-pose errors can also be added using supplied calibration.
- **Advantages:** sensor-quality ground truth, mixed domains, exact paper relevance, small scene count, official Windows evaluator.
- **Disadvantages:** 24 MP originals require VGGT resizing; point-cloud alignment/masking must be implemented carefully; meshes are not the canonical target.
- **Expected project runtime (Plan):** 1–5 seconds per 10-view inference after model load is a conservative planning band until measured locally; a 2-scene × 10-condition pilot should fit within roughly 30–90 minutes including exports, while human review takes longer.

### 2. DTU Robot Image MVS Dataset

- **Publication/citation:** Rasmus Jensen et al., “Large Scale Multi-view Stereopsis Evaluation,” CVPR 2014.
- **Official source:** [DTU Robot Image Data Sets](https://roboimagedata.compute.dtu.dk/) and [MVS 2014 page](https://roboimagedata.compute.dtu.dk/?page_id=36).
- **License:** official site describes it as freely available **citeware**. This is a citation condition, not a complete permissive software license; verify redistribution rules before download.
- **Intended task:** controlled multi-view stereo evaluation.
- **Contents:** 124 tabletop scenes, controlled robot-camera viewpoints and lighting, calibrated cameras, structured-light geometry/point clouds, and standard evaluation protocols. It is indoor/object-centric and is not a natural dynamic-scene dataset.
- **Size/difficulty:** full variants and preprocessed forms are commonly tens of GB; exact chosen release must be recorded. Legacy organization and community preprocessing scripts add friction.
- **VGGT evaluation:** exact paper dense-MVS dataset and metrics; quantitative geometry is strong.
- **Advantages:** reproducible capture, strong ground truth, widely recognized, easy view-count and overlap control.
- **Disadvantages:** domain is narrow; known cameras are traditionally assumed by MVS methods, while VGGT predicts them; full release is unnecessary for a seminar.
- **Expected runtime (Plan):** similar per condition to ETH3D after equal VGGT preprocessing; dataset preparation is likely longer than inference.

### 3. Tanks and Temples

- **Publication/citation:** Arno Knapitsch et al., “Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction,” ACM TOG 2017.
- **Official source:** [download page](https://www.tanksandtemples.org/download/), [evaluation submission format](https://www.tanksandtemples.org/uploadpreparations/), and [license](https://www.tanksandtemples.org/license/).
- **License:** the page names CC BY 4.0 but also adds research/non-commercial and redistribution restrictions. Because those terms are internally narrower than plain CC BY, obey the full site terms and do not summarize it as unrestricted CC BY.
- **Intended task:** large-scale image-based 3D reconstruction.
- **Contents:** 4K videos or sampled frames, training laser scans, intermediate/advanced hidden test geometry, crop/alignment data, example COLMAP cameras; exact intrinsics are not explicitly supplied.
- **Quantitative use:** official precision/recall/F-score evaluation on training scenes; server evaluation requires complete test groups.
- **Advantages:** visually compelling, indoor/outdoor, difficult large-scale geometry, standard benchmark.
- **Disadvantages:** heavier inputs, hidden test GT, leaderboard overhead, and large scene reconstructions are awkward for a short 16 GB study.

### 4. ScanNet++

- **Publication/citation:** Chandan Yeshwanth et al., “ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes,” ICCV 2023; use the current version citation required by the site.
- **Official source:** [dataset site](https://scannetpp.mlsg.cit.tum.de/scannetpp/) and [documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation).
- **License/access:** registration/application and dataset terms apply; verify the current agreement before acquisition.
- **Intended task:** high-fidelity indoor reconstruction, novel view synthesis, and semantic understanding.
- **Contents:** 1,006 scenes, registered 33 MP DSLR images, metric COLMAP cameras, laser scans/meshes, iPhone RGB-D streams and depth.
- **Advantages:** excellent metric geometry and camera truth; modern and high quality.
- **Disadvantages:** multi-terabyte-class full data, access friction, indoor-only, modality complexity, excessive preparation for this seminar.
- **Quantitative use:** yes, but only after substantial conversion and careful split/task selection.

### 5. CO3Dv2 - Common Objects in 3D

- **Publication/citation:** David Novotny et al., “Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction,” ICCV 2021; cite the CO3Dv2 release metadata as applicable.
- **Official source:** [Meta CO3D page](https://ai.meta.com/datasets/CO3D-dataset/) and [official code](https://github.com/facebookresearch/co3d).
- **License:** dataset-specific terms in the download flow/repository apply; verify before acquisition and redistribution.
- **Intended task:** category-level object reconstruction and novel-view synthesis.
- **Contents:** approximately 18,619 crowd-sourced videos, 50 categories, 1.5M camera-annotated frames, and 5,625 point-cloud-annotated sequences. Cameras and point clouds are SfM/reconstruction-derived rather than sensor ground truth.
- **Advantages:** exact VGGT camera benchmark, many object types, easy view/order sampling, common in current 3D learning.
- **Disadvantages:** VGGT trained on Co3Dv2; full scale and metadata dependencies are unnecessary; object masks/content vary; dense sensor truth is absent.
- **Quantitative use:** pose AUC on a held-out test protocol is credible; claims of out-of-distribution generalization are not.

### 6. MegaDepth

- **Publication/citation:** Zhengqi Li and Noah Snavely, “MegaDepth: Learning Single-View Depth Prediction from Internet Photos,” CVPR 2018.
- **Official source:** [project page](https://www.cs.cornell.edu/projects/megadepth/) and [ETH-hosted files](https://cvg-data.inf.ethz.ch/megadepth/).
- **License:** images originate from Flickr and processed artifacts have dataset terms; rights may vary by source image. Verify the precise release and intended use before download.
- **Intended task:** depth learning and correspondence/pose evaluation from Internet landmark photos.
- **Contents:** multi-view Internet images, SfM cameras, MVS-derived depth and sparse/dense reconstruction artifacts; mainly outdoor/landmark content.
- **Size/difficulty:** the compact MegaDepth-1500 evaluation archive is listed as 1.3 GB; the full training corpus is much larger and version/mirror management is difficult.
- **Advantages:** challenging illumination, time, camera, and appearance variation; extremely common for matching and pose research.
- **Disadvantages:** VGGT training includes MegaDepth; “ground truth” is reconstruction-derived; Flickr provenance and scene overlap complicate claims.
- **Quantitative use:** useful for relative pose/matching-style evaluation, less clean for absolute dense geometry.

### 7. Additional paper-referenced datasets

**RealEstate10K.** Zhou et al., “Stereo Magnification,” SIGGRAPH 2018. It supplies real-estate videos with SfM camera trajectories and is an explicitly unseen VGGT camera-pose evaluation set. It is attractive scientifically, but original YouTube-based acquisition is fragile and it lacks dense geometry. Use only if pose estimation becomes the central research question.

**Image Matching Challenge (IMC).** The supplement reports phototourism camera-pose AUC on IMC and discusses MegaDepth scene overlap. IMC has strong reference reconstructions and current relevance, but challenge-specific splits, downloads, and pose-only evaluation make it less direct than ETH3D for the assignment’s “strengths and weaknesses” story.

**ScanNet.** The original ScanNet dataset provides RGB-D frames, camera poses, meshes, and indoor annotations. It is in VGGT training, and the supplement discusses excluding ScanNet-1500 evaluation scenes. It remains important literature context but is not preferred over the non-training ETH3D benchmark.

## Why each non-selected candidate was rejected

- **Tanks and Temples:** compelling but larger, test GT is hidden, calibration is less direct, and official submission expects complete groups.
- **ScanNet++:** technically rich but access, storage, and conversion costs overwhelm the seminar scope.
- **CO3Dv2:** direct paper benchmark but also training data; full dataset is enormous and geometry annotations are reconstruction-derived.
- **MegaDepth:** compact evaluation subset exists, but it is training data and its depth/cameras are pseudo-ground-truth.
- **RealEstate10K:** valuable unseen pose benchmark, but no dense geometry and brittle acquisition.
- **IMC:** excellent for camera pose, but challenge overhead and no direct dense reconstruction score.
- **ScanNet:** excellent indoor labels, but training overlap weakens generalization evidence.
- **DTU:** selected only as secondary because controlled tabletop scenes provide less diversity than ETH3D.

## Final seminar structure

### Part A - Public benchmark

Use two ETH3D training scenes: one indoor and one outdoor. Reproduce a reduced form of the paper’s 10-frame point-map evaluation, add predicted-camera comparison, and run controlled view-count/overlap/degradation variants. Report geometry errors only after a documented alignment, mask, and threshold protocol.

### Part B - Our photographs

Use one controlled object and one natural indoor/outdoor capture from `data/custom_inputs`. These images have no geometry ground truth, so evaluate them qualitatively and with internal stability diagnostics only. They add originality, recognizable failure cases, and direct compliance with “different inputs.”

The combination is stronger than either part alone: ETH3D supplies quantitative credibility and comparability to VGGT; our captures supply originality and non-ideal conditions not curated by a benchmark.

## Bibliography and official links

1. J. Wang et al., “VGGT: Visual Geometry Grounded Transformer,” CVPR 2025. [Paper and supplement](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html).
2. T. Schöps et al., “A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos,” CVPR 2017. [ETH3D](https://eth3d.ethz.ch/).
3. R. Jensen et al., “Large Scale Multi-view Stereopsis Evaluation,” CVPR 2014. [DTU MVS](https://roboimagedata.compute.dtu.dk/?page_id=36).
4. A. Knapitsch et al., “Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction,” ACM TOG 2017. [Benchmark](https://www.tanksandtemples.org/).
5. C. Yeshwanth et al., “ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes,” ICCV 2023. [Dataset](https://scannetpp.mlsg.cit.tum.de/scannetpp/).
6. D. Novotny et al., “Common Objects in 3D,” ICCV 2021. [CO3D](https://ai.meta.com/datasets/CO3D-dataset/).
7. Z. Li and N. Snavely, “MegaDepth,” CVPR 2018. [Project](https://www.cs.cornell.edu/projects/megadepth/).
8. T. Zhou et al., “Stereo Magnification,” ACM TOG 2018. [RealEstate10K project](https://google.github.io/realestate10k/).
9. A. Dai et al., “ScanNet,” CVPR 2017. [Dataset](http://www.scan-net.org/).
10. Image Matching Challenge. [Official challenge organization](https://www.kaggle.com/competitions/image-matching-challenge-2024).

