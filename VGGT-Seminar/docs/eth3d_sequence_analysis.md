# ETH3D sequence analysis

Phase 6.1 examined `delivery_area` (44 frames) and `courtyard` (38 frames) without loading VGGT. Lexical filename order agrees with calibration order; timestamps are unavailable, so no timestamp claim is made.

## Evidence

The analysis converts ETH3D world-to-camera poses to camera centers with `C = -R^T t` and viewing directions with `R^T [0,0,1]`. It measures camera-center displacement and relative viewing angle, then computes deterministic ORB correspondences on images downscaled to at most 960 pixels. Matches pass a 0.75 Lowe ratio test; geometric support is the fundamental-matrix RANSAC inlier count (1.5 px, confidence 0.999). These are selection proxies, not reconstruction metrics.

Four ten-frame proposals were compared: centered contiguous, pose-constrained, feature-constrained, and hybrid pose/feature. The hybrid proposal was selected because it jointly favors coherent motion, useful baseline, and verified image overlap. It selected frames 0–9 for both scenes. Visual contact-sheet inspection confirmed continuous content and viewpoint motion.

For `delivery_area`, the chosen window has mean neighbor displacement 0.7251, angle 3.6058°, 708.3 ratio matches, and 530.9 RANSAC inliers. Later frames contain sharp orientation changes and weak-overlap discontinuities, including only 9–10 inliers near indices 33–39.

For `courtyard`, the chosen window has mean neighbor displacement 0.6364, angle 2.8218°, 536.8 ratio matches, and 431.6 inliers. Later sequence sections contain frequent turns and a large 8.9241-unit jump after index 24 with 19 inliers.

Diagnostics are reproducibly generated under ignored `outputs/analysis/eth3d_overlap/`: trajectories, neighbor plots, overlap heatmaps, candidate/final contact sheets, CSV tables, and the JSON recommendation record. The completed CPU analysis took 49.88 seconds; reruns reuse pair-metric caches and refuse to overwrite the result directory.

## Limitations

Calibration and sparse local features only approximate suitability for VGGT. Repeated texture can inflate matches, featureless regions can suppress them, and RANSAC inliers do not guarantee dense 3D quality. ETH3D laser scans were not consulted, and no model inference or evaluation was performed.
