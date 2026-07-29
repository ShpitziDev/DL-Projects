import numpy as np

from vggt_seminar.tartanair import (
    apply_similarity,
    camera_to_world_opencv,
    intrinsics,
    scale_aligned_depth_metrics,
    umeyama_similarity,
    valid_depth_mask,
    world_to_camera_opencv,
)


def test_pose_conversion_round_trip():
    pose = np.array([1, 2, 3, 0, 0, 0, 1], dtype=float)
    c2w = camera_to_world_opencv(pose)
    w2c = world_to_camera_opencv(pose)
    assert np.allclose(np.vstack([w2c, [0, 0, 0, 1]]) @ c2w, np.eye(4))
    assert np.allclose(c2w[:3, 2], [1, 0, 0])  # OpenCV forward -> NED forward.


def test_intrinsics_are_centered_90_degree_fov():
    k = intrinsics()
    assert np.allclose(np.diag(k), [320, 320, 1])
    assert np.allclose(k[:2, 2], [319.5, 319.5])


def test_depth_validity_and_metrics():
    depth = np.array([[0, 1, 2, 101, np.inf]], dtype=float)
    assert valid_depth_mask(depth).tolist() == [[False, True, True, False, False]]
    gt = np.array([1.0, 2.0, 4.0])
    metrics = scale_aligned_depth_metrics(gt / 2, gt, np.ones(3, dtype=bool))
    assert metrics["abs_rel"] == 0
    assert metrics["delta1"] == 1


def test_umeyama_recovers_similarity():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    target = 2.5 * (source @ rotation.T) + np.array([3, 4, 5])
    aligned = apply_similarity(source, umeyama_similarity(source, target))
    assert np.allclose(aligned, target)
