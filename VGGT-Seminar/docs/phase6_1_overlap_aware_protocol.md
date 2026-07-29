# Phase 6.1 overlap-aware nested protocol

The frozen strategy is `overlap_aware_nested`, protocol `eth3d-overlap-aware-nested-v1`. Its source of truth is `configs/experiments/phase6_1_overlap_aware_frames.yaml`.

Phase 6 exposed the problem: full-sequence uniform sampling chose global endpoints with limited common content. Multi-view reconstruction needs shared observations to relate cameras and geometry, so view count must be varied inside one coherent region rather than across unrelated endpoints.

The procedure ranks contiguous ten-frame windows using normalized pose displacement, viewing-angle coherence, ratio-test feature matches, and fundamental-matrix inliers. Within the winning hybrid window it selects deterministic, strictly nested sets S2 ⊂ S4 ⊂ S6 ⊂ S8 ⊂ S10. It prefers baseline-bearing anchors, rejects candidates below 20 inliers or 0.015 normalized matches, preserves original order, and breaks ties by lowest original index.

| Scene | S2 | S4 | S6 | S8 | S10 |
|---|---|---|---|---|---|
| delivery_area | 0,6 | 0,3,6,9 | 0,3,4,5,6,9 | 0,1,2,3,4,5,6,9 | 0–9 |
| courtyard | 0,9 | 0,4,6,9 | 0,2,3,4,6,9 | 0,2,3,4,5,6,7,9 | 0–9 |

The centered method was rejected because its arbitrary location does not use overlap evidence. Pose-only cannot detect occlusion or texture failure. Feature-only can prefer repeated texture or minimal baseline. The hybrid uses both independent signals and won on coherent trajectory plus verified correspondences; its CPU cost was under one minute for both scenes after limiting pairs to a 12-frame neighborhood.

Nesting makes the view-count comparison interpretable: every larger condition retains all evidence from the smaller one and adds views in the same local geometry. It therefore avoids changing scene region together with view count. Still uncontrolled are model stochastic/numeric effects, occlusion, illumination, texture distribution, baseline distribution within each subset, and the fact that sparse matches are not dense geometric overlap.

Exact filenames are frozen in the YAML. Loading is strict: an unknown scene or unavailable frame count raises an error; there is no fallback to sequential or evenly-spaced selection. This prevents silently changing the experimental input.

This phase authorizes only selection analysis. The config explicitly sets `approve_inference: false`; it does not load a checkpoint, call VGGT, run a pilot matrix, or evaluate laser-scan ground truth.

ETH3D poses and image correspondences are used only to define fixed inputs, not to score VGGT predictions or claim model accuracy.
