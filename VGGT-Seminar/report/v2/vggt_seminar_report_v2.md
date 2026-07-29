# Reproducing VGGT: A Controlled Study of View-Count Scaling with Overlap-Aware ETH3D Inputs

Peleg Shpitzer · Course / Instructor · University of Haifa · July 2026

## Abstract
Visual Geometry Grounded Transformer (VGGT) predicts cameras, depth, point maps, confidence, and tracks from one or more images in a single feed-forward model. We reproduced the maintained official implementation and public VGGT-1B checkpoint locally, then designed a controlled study of how operational behavior changes with 2, 4, 6, 8, and 10 overlapping views. Two calibrated ETH3D scenes, delivery_area and courtyard, were sampled with a deterministic overlap-aware protocol combining pose and feature evidence. Across ten configurations, all required outputs remained finite. Allocated memory rose from about 5.26 to 6.60 GiB, synchronized inference stayed near one second, and complete processing time grew with the amount of per-view postprocessing. Model confidence generally increased but was not monotonic, while visible gains appeared to diminish around eight to ten views. Dominant scene structure remained qualitatively coherent. Because predictions were not aligned to ETH3D coordinates, this report makes no quantitative reconstruction-accuracy claim. The results support reliable local operation and a defensible view-count methodology, while identifying aligned evaluation as the main next scientific step.

## Study at a Glance
Ten saved configurations span two scenes and five nested view counts. The design isolates view count while retaining earlier frames in every larger set.

## Introduction
VGGT asks whether a shared transformer can replace much of a conventional multi-stage geometry pipeline with direct prediction [1]. This project tests that proposition at the level appropriate for a seminar reproduction: operational correctness, a controlled input study, resource measurements, and bounded qualitative analysis. The research question is how runtime, memory, confidence, predicted camera structure, and visible completeness change as overlapping input views increase from two to ten.

## VGGT in Brief
Multi-view RGB images are tokenized by a DINOv2 encoder [5]. An aggregator alternates frame-local attention with global cross-view attention, after which dedicated heads predict cameras, depth, point maps, confidence, and tracks. Predictions use the first camera as a reference. The conceptual diagram in this report is original and based on the architecture description in VGGT [1], not a reproduction of its figure.

## Reproduction Setup
The maintained official repository was pinned at `a288dd0f14786c93483e45524328726ab7b1b4ce`; the public checkpoint hash was `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. Inference used Python 3.11.15, PyTorch 2.13.0+cu130, CUDA 13.0, an RTX 5080, BF16 autocast, and disabled Flash SDPA. The official example and all documented output heads were validated. This establishes operational reproduction, not independent reproduction of every benchmark result reported by the authors.

## Experimental Methodology
ETH3D provides calibrated high-resolution imagery and laser scans [2]. We selected delivery_area and courtyard as complementary indoor and outdoor cases. Laser geometry was deliberately not used for scoring because coordinate conventions and similarity alignment must be validated first. The independent variable was view count; model, checkpoint, preprocessing, precision, hardware, ordering, postprocessing, and the strict `confidence > median` visualization rule were controlled. There was one run per condition, so timings are descriptive and have no error bars.

## Overlap-Aware Frame Selection
Uniform endpoints `[0,43]` increased sequence coverage but shared little visible structure. The selected hybrid method combined camera-center distance, viewing-direction and relative-rotation angles, ORB correspondence counts [3], and fundamental-matrix RANSAC inliers [4]. It froze nested subsets satisfying `S2 ⊂ S4 ⊂ S6 ⊂ S8 ⊂ S10`. Nesting reduces frame-selection confounding because every larger condition retains all evidence from the smaller condition.

## Experimental Results
All ten configurations produced finite outputs. Allocated VRAM rose from 5.261 GiB at S2 to 6.599 GiB at S10 in both scenes; reserved memory approached 9.23 GiB. Synchronized inference remained between 0.713 and 1.142 seconds, whereas total processing reached 17.386 seconds for delivery_area and 18.645 seconds for courtyard at S10. The divergence is explained by CPU transfer, decoding, validation, visualization, and serialization scaling with dense per-view outputs.

Confidence increased overall but not monotonically. Delivery depth confidence dipped at S8 before recovering at S10; courtyard peaked at S8 and softened at S10. Courtyard confidence was higher at matched counts, but confidence is model output - not calibrated accuracy - and cannot establish that one scene was easier or reconstructed more accurately. Predicted camera-path extent also changed non-monotonically in arbitrary unaligned units.

## Qualitative Results
Shared-first-view depth retained the large door plane and ceiling structure in delivery_area and the facade, windows, foreground furniture, and ground in courtyard. Confidence concentrated around recognizable edges and structures, while point-cloud previews preserved dominant surfaces across counts. Visible differences became smaller near S8-S10. These are structured visual observations, not metric completeness or accuracy measurements.

## Strengths and Failure Modes
Observed strengths include reliable operation of all heads, coherent dominant structure, moderate allocated-memory growth, improved common visibility from overlap-aware inputs, and generally higher confidence with more views. Difficult behavior includes non-monotonic confidence, changing camera extent, diminishing high-view gains, and sensitivity of qualitative comparisons to auto-fit framing. No catastrophic failure was observed, and none is manufactured here.

## Discussion
Near-constant inference and growing total time describe different system boundaries: the model forward is efficient for this range, while artifact production grows with views. Moderate memory growth may reflect parameter reuse, chunked decoding, and allocator behavior; this is a hypothesis rather than an isolated causal measurement. Confidence changes may reflect redundant, useful, or inconsistent added evidence. The two-scene pattern suggests practical robustness, but not broad generalization.

## Limitations
The study covers two scenes and one local trajectory window per scene, with one timing run per condition. It includes no alignment, pose/depth/scan error, alternative model, order sensitivity, degradation study, robustness matrix, or fine-tuning. Confidence is not accuracy. Geometry has arbitrary scale, and independently auto-fitted previews can exaggerate apparent size differences. These boundaries prevent claims of benchmark accuracy, statistical significance, or superiority.

## Future Work
The priority is validated similarity alignment to ETH3D, followed by pose and depth or point-to-scan evaluation. Further work should add scenes, repeated timing trials, input-order tests, degradations, alternative selection strategies, and a reconstruction baseline. Fine-tuning should be considered only after aligned evaluation identifies a repeated domain-specific failure and suitable labels and resources exist.

## Conclusion
VGGT was successfully reproduced and operated reliably on consumer hardware. Overlap-aware nested sampling enabled a controlled two-scene 2/4/6/8/10-view study. Additional views generally increased confidence and visible completeness, but gains were non-monotonic and appeared to diminish at high counts. No additional inference is required for this report redesign; aligned quantitative evaluation is the appropriate next scientific phase.

## References
[1] J. Wang et al., “VGGT: Visual Geometry Grounded Transformer,” CVPR, 2025.  
[2] T. Schöps et al., “A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos,” CVPR, 2017.  
[3] E. Rublee et al., “ORB: An Efficient Alternative to SIFT or SURF,” ICCV, 2011.  
[4] M. A. Fischler and R. C. Bolles, “Random Sample Consensus,” Communications of the ACM, 1981.  
[5] M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” Transactions on Machine Learning Research, 2024.
