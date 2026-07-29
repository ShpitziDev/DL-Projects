# ETH3D and VGGT camera conventions

## Verified code conventions

ETH3D stores COLMAP image poses as a Hamilton quaternion and translation that transform a point from the global frame into the image camera frame:

\[
\mathbf{x}_c = \mathbf{R}_{cw}\mathbf{x}_w + \mathbf{t}_{cw}.
\]

The camera center in world coordinates is therefore:

\[
\mathbf{C}_w = -\mathbf{R}_{cw}^{\mathsf{T}}\mathbf{t}_{cw}.
\]

ETH3D documents camera axes as OpenCV/COLMAP style: x right, y down, z forward. Its global frame has arbitrary origin and orientation but is tied to the metric laser-scan reconstruction.

VGGT's `pose_encoding_to_extri_intri` returns a 3 x 4 `extrinsic = [R|t]` in the same OpenCV axis convention and explicitly documents it as **camera-from-world**, i.e. world-to-camera. VGGT's unprojection helper inverts this matrix before mapping camera points into the predicted world frame, which independently confirms the direction.

Code references:

- `external/vggt/vggt/utils/pose_enc.py`, `pose_encoding_to_extri_intri`;
- `external/vggt/vggt/utils/geometry.py`, `unproject_depth_map_to_point_map` and `depth_to_world_points`;
- `src/vggt_seminar/eth3d.py`, `load_camera_poses`.

## Reference frame and scale

VGGT training expresses scene quantities relative to the first input camera and normalizes scene scale. Thus the first selected image, `DSC_0675.JPG` in the Phase 6 smoke test, defines the reconstruction reference. VGGT's output coordinates are not automatically the ETH3D global coordinates and are not guaranteed to have ETH3D metric scale.

Directly subtracting VGGT and ETH3D translations would therefore be invalid. A later evaluation must compute camera centers using \(C=-R^Tt\), match corresponding frames, and estimate a 7-DoF similarity transform (rotation, translation, uniform scale), with an explicitly frozen alignment protocol.

## Axes and handedness

Both interfaces claim OpenCV-style camera axes, so an axis flip is not assumed. However, their **world frames differ**, and conversion code must still be validated by projecting known points and visualizing camera frustums before quantitative pose errors are computed. A handedness or transpose mistake remains possible at the integration boundary until that fixture is completed.

## Phase 6 boundary

The smoke test decodes cameras, derives centers, and visualizes their arbitrary-scale X-Z layout. It does not align the coordinate systems, estimate a pose error, use ICP, or claim metric reconstruction accuracy.

