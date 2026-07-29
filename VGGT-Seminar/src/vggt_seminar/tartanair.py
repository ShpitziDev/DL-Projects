"""TartanAir V2 loading, geometry conversion, selection, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CAMERA_FROM_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


@dataclass(frozen=True)
class TartanTrajectory:
    name: str
    root: Path
    image_paths: tuple[Path, ...]
    depth_paths: tuple[Path, ...]
    poses_xyzw: np.ndarray

    def __len__(self) -> int:
        return len(self.image_paths)


def load_trajectory(root: Path, trajectory: str) -> TartanTrajectory:
    path = root / trajectory
    images = tuple(sorted((path / "image_lcam_front").glob("*.png")))
    depths = tuple(sorted((path / "depth_lcam_front").glob("*.png")))
    poses = np.loadtxt(path / "pose_lcam_front.txt", dtype=np.float64)
    if not images or len(images) != len(depths) or poses.shape != (len(images), 7):
        raise ValueError(f"Broken TartanAir trajectory contract: {path}")
    image_ids = [p.name[:6] for p in images]
    depth_ids = [p.name[:6] for p in depths]
    if image_ids != depth_ids:
        raise ValueError(f"RGB/depth frame mismatch: {path}")
    return TartanTrajectory(trajectory, path, images, depths, poses)


def read_depth(path: Path) -> np.ndarray:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.dtype != np.uint8 or encoded.ndim != 3 or encoded.shape[2] != 4:
        raise ValueError(f"Expected lossless four-byte TartanAir depth PNG: {path}")
    return encoded.view("<f4").squeeze(-1).copy()


def valid_depth_mask(depth: np.ndarray, minimum: float = 0.1, maximum: float = 100.0) -> np.ndarray:
    return np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)


def intrinsics(size: int = 640) -> np.ndarray:
    focal = size / 2.0
    center = (size - 1.0) / 2.0
    return np.array([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]])


def camera_to_world_opencv(pose_xyzw: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_xyzw, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix() @ CAMERA_FROM_OPENCV
    result[:3, 3] = pose[:3]
    return result


def world_to_camera_opencv(pose_xyzw: np.ndarray) -> np.ndarray:
    return np.linalg.inv(camera_to_world_opencv(pose_xyzw))[:3]


def resize_square_ground_truth(
    depth: np.ndarray, target: int = 518, source_intrinsics: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth.shape != (depth.shape[0], depth.shape[0]):
        raise ValueError("V3 adapter currently requires square TartanAir frames")
    source_intrinsics = intrinsics(depth.shape[0]) if source_intrinsics is None else source_intrinsics
    resized = cv2.resize(depth, (target, target), interpolation=cv2.INTER_NEAREST)
    scale = target / depth.shape[0]
    target_intrinsics = source_intrinsics.copy() * scale
    target_intrinsics[2, 2] = 1.0
    mask = valid_depth_mask(resized)
    resized = np.where(mask, resized, 0.0).astype(np.float32)
    return resized, mask, target_intrinsics.astype(np.float32)


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 2:
        raise ValueError("Similarity alignment requires matched Nx3 arrays with N>=2")
    src_mean, dst_mean = source.mean(0), target.mean(0)
    src_centered, dst_centered = source - src_mean, target - dst_mean
    covariance = dst_centered.T @ src_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, alignment: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = alignment
    return scale * (np.asarray(points) @ rotation.T) + translation


def rotation_errors_deg(pred_c2w: np.ndarray, gt_c2w: np.ndarray, alignment_rotation: np.ndarray) -> np.ndarray:
    errors = []
    for pred, gt in zip(pred_c2w, gt_c2w):
        aligned = alignment_rotation @ pred[:3, :3]
        delta = gt[:3, :3].T @ aligned
        cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        errors.append(np.degrees(np.arccos(cosine)))
    return np.asarray(errors)


def scale_aligned_depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p, g = np.asarray(pred)[mask], np.asarray(gt)[mask]
    scale = float(np.median(g) / max(float(np.median(p)), 1e-8))
    p = p * scale
    ratio = np.maximum(p / g, g / np.maximum(p, 1e-8))
    return {
        "scale": scale,
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "valid_pixels": int(mask.sum()),
    }


def orb_pair_score(left: Path, right: Path) -> dict[str, int]:
    a = cv2.imread(str(left), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=4000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return {"matches": 0, "inliers": 0}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 8:
        return {"matches": len(matches), "inliers": 0}
    pts_a = np.float32([ka[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kb[m.trainIdx].pt for m in matches])
    _, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.5, 0.999)
    return {"matches": len(matches), "inliers": int(mask.sum()) if mask is not None else 0}
