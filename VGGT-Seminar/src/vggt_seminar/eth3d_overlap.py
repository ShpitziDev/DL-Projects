from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .eth3d import ETH3DPose, ETH3DScene, load_scene


PROTOCOL_VERSION = "eth3d-overlap-aware-nested-v1"


@dataclass(frozen=True)
class FrameGeometry:
    index: int
    filename: str
    camera_center: tuple[float, float, float]
    viewing_direction: tuple[float, float, float]


@dataclass(frozen=True)
class PairMetrics:
    index_a: int
    index_b: int
    center_distance: float
    viewing_angle_deg: float
    relative_rotation_deg: float
    keypoints_a: int
    keypoints_b: int
    ratio_matches: int
    normalized_matches: float
    fundamental_inliers: int
    inlier_ratio: float


@dataclass(frozen=True)
class CandidateWindow:
    method: str
    start: int
    stop: int
    score: float
    mean_neighbor_distance: float
    mean_neighbor_angle_deg: float
    mean_neighbor_matches: float
    mean_neighbor_inliers: float

    @property
    def indices(self) -> list[int]:
        return list(range(self.start, self.stop))


def quaternion_wxyz_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("Quaternion norm must be positive")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_center(pose: ETH3DPose) -> np.ndarray:
    rotation = quaternion_wxyz_to_rotation(pose.quaternion_wxyz)
    translation = np.asarray(pose.translation_xyz, dtype=np.float64)
    return -rotation.T @ translation


def viewing_direction(pose: ETH3DPose) -> np.ndarray:
    rotation = quaternion_wxyz_to_rotation(pose.quaternion_wxyz)
    direction = rotation.T @ np.array([0.0, 0.0, 1.0])
    return direction / np.linalg.norm(direction)


def vector_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    first, second = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 0:
        raise ValueError("Cannot measure an angle involving a zero vector")
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def relative_rotation_deg(first: ETH3DPose, second: ETH3DPose) -> float:
    rotation = quaternion_wxyz_to_rotation(second.quaternion_wxyz) @ quaternion_wxyz_to_rotation(first.quaternion_wxyz).T
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def frame_geometry(scene: ETH3DScene) -> list[FrameGeometry]:
    return [FrameGeometry(
        index=index,
        filename=path.name,
        camera_center=tuple(float(value) for value in camera_center(pose)),
        viewing_direction=tuple(float(value) for value in viewing_direction(pose)),
    ) for index, (path, pose) in enumerate(zip(scene.image_paths, scene.poses))]


def _analysis_gray(path: Path, max_dimension: int) -> tuple[np.ndarray, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    scale = min(1.0, max_dimension / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, scale


def compute_orb_features(
    paths: Sequence[Path], max_dimension: int = 960, nfeatures: int = 2500,
) -> tuple[list[list[cv2.KeyPoint]], list[np.ndarray | None], list[float]]:
    cv2.setRNGSeed(0)
    detector = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    keypoints, descriptors, scales = [], [], []
    for path in paths:
        image, scale = _analysis_gray(path, max_dimension)
        points, description = detector.detectAndCompute(image, None)
        keypoints.append(points)
        descriptors.append(description)
        scales.append(scale)
    return keypoints, descriptors, scales


def match_orb_pair(
    keypoints_a: Sequence[cv2.KeyPoint], descriptors_a: np.ndarray | None,
    keypoints_b: Sequence[cv2.KeyPoint], descriptors_b: np.ndarray | None,
    ratio: float = 0.75, ransac_threshold: float = 1.5,
) -> tuple[int, float, int, float]:
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return 0, 0.0, 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = [first for first, second in pairs if first.distance < ratio * second.distance]
    normalized = len(good) / max(1, min(len(keypoints_a), len(keypoints_b)))
    if len(good) < 8:
        return len(good), float(normalized), 0, 0.0
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in good])
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in good])
    cv2.setRNGSeed(0)
    _, mask = cv2.findFundamentalMat(points_a, points_b, cv2.FM_RANSAC, ransac_threshold, 0.999)
    inliers = int(mask.sum()) if mask is not None else 0
    return len(good), float(normalized), inliers, float(inliers / len(good))


def analyze_scene_pairs(
    scene: ETH3DScene, max_pair_gap: int = 12, max_dimension: int = 960,
    nfeatures: int = 2500, ratio: float = 0.75, ransac_threshold: float = 1.5,
) -> tuple[list[FrameGeometry], list[PairMetrics], list[float]]:
    geometry = frame_geometry(scene)
    keypoints, descriptors, scales = compute_orb_features(scene.image_paths, max_dimension, nfeatures)
    pairs = []
    for index_a in range(len(scene.image_paths)):
        for index_b in range(index_a + 1, min(len(scene.image_paths), index_a + max_pair_gap + 1)):
            matches, normalized, inliers, inlier_ratio = match_orb_pair(
                keypoints[index_a], descriptors[index_a], keypoints[index_b], descriptors[index_b], ratio, ransac_threshold
            )
            first, second = scene.poses[index_a], scene.poses[index_b]
            pairs.append(PairMetrics(
                index_a=index_a,
                index_b=index_b,
                center_distance=float(np.linalg.norm(camera_center(first) - camera_center(second))),
                viewing_angle_deg=vector_angle_deg(viewing_direction(first), viewing_direction(second)),
                relative_rotation_deg=relative_rotation_deg(first, second),
                keypoints_a=len(keypoints[index_a]),
                keypoints_b=len(keypoints[index_b]),
                ratio_matches=matches,
                normalized_matches=normalized,
                fundamental_inliers=inliers,
                inlier_ratio=inlier_ratio,
            ))
    return geometry, pairs, scales


def pair_lookup(pairs: Sequence[PairMetrics]) -> dict[tuple[int, int], PairMetrics]:
    return {(pair.index_a, pair.index_b): pair for pair in pairs}


def window_summary(method: str, start: int, length: int, pairs: Sequence[PairMetrics]) -> CandidateWindow:
    lookup = pair_lookup(pairs)
    neighbors = [lookup[(index, index + 1)] for index in range(start, start + length - 1)]
    distances = np.array([pair.center_distance for pair in neighbors])
    angles = np.array([pair.viewing_angle_deg for pair in neighbors])
    matches = np.array([pair.ratio_matches for pair in neighbors])
    inliers = np.array([pair.fundamental_inliers for pair in neighbors])
    all_neighbor_distances = np.array([pair.center_distance for pair in pairs if pair.index_b == pair.index_a + 1])
    distance_scale = max(float(np.median(all_neighbor_distances)), 1e-8)
    pose_continuity = float(np.exp(-np.mean(np.abs(distances / distance_scale - 1.0))))
    feature_quality = float(np.mean(np.log1p(inliers)) / max(math.log1p(max(1, int(inliers.max()))), 1e-8))
    coverage = float(np.sum(distances) / distance_scale / max(length - 1, 1))
    if method == "pose":
        score = pose_continuity + 0.15 * min(coverage, 2.0) - 0.01 * float(np.mean(angles))
    elif method == "feature":
        score = feature_quality + 0.2 * float(np.mean([pair.inlier_ratio for pair in neighbors]))
    elif method == "hybrid":
        score = 0.45 * pose_continuity + 0.45 * feature_quality + 0.1 * min(coverage, 2.0) - 0.005 * float(np.mean(angles))
    elif method == "centered":
        score = 0.0
    else:
        raise ValueError(f"Unknown window method: {method}")
    return CandidateWindow(method, start, start + length, score, float(distances.mean()),
                           float(angles.mean()), float(matches.mean()), float(inliers.mean()))


def rank_windows(frame_count: int, pairs: Sequence[PairMetrics], method: str, length: int = 10) -> list[CandidateWindow]:
    windows = [window_summary(method, start, length, pairs) for start in range(frame_count - length + 1)]
    return sorted(windows, key=lambda window: (-window.score, window.start))


def _pair(lookup: dict[tuple[int, int], PairMetrics], first: int, second: int) -> PairMetrics:
    return lookup[(min(first, second), max(first, second))]


def build_nested_subsets(
    window_indices: Sequence[int], pairs: Sequence[PairMetrics],
    sizes: Sequence[int] = (2, 4, 6, 8, 10), min_gap: int = 2,
    min_inliers: int = 20, min_normalized_matches: float = 0.015,
) -> dict[int, list[int]]:
    candidates = sorted(window_indices)
    if len(candidates) < max(sizes):
        raise ValueError("Candidate window is too small for requested nested subsets")
    lookup = pair_lookup(pairs)
    eligible_pairs = []
    for position, first in enumerate(candidates):
        for second in candidates[position + 1:]:
            gap = second - first
            if gap < min_gap or (first, second) not in lookup:
                continue
            metric = lookup[(first, second)]
            if metric.fundamental_inliers < min_inliers or metric.normalized_matches < min_normalized_matches:
                continue
            diversity = metric.center_distance * (1.0 + metric.viewing_angle_deg / 45.0)
            quality = math.log1p(metric.fundamental_inliers) * (0.5 + metric.inlier_ratio)
            eligible_pairs.append((quality * diversity, first, second))
    if not eligible_pairs:
        raise ValueError("No non-trivial overlapping anchor pair satisfies the thresholds")
    _, first, second = sorted(eligible_pairs, key=lambda item: (-item[0], item[1], item[2]))[0]
    selected = {first, second}
    result = {2: sorted(selected)}
    for target in sizes[1:]:
        while len(selected) < target:
            ranked = []
            for candidate in candidates:
                if candidate in selected:
                    continue
                metrics = [_pair(lookup, candidate, existing) for existing in selected if (min(candidate, existing), max(candidate, existing)) in lookup]
                valid = [metric for metric in metrics if metric.fundamental_inliers >= min_inliers and metric.normalized_matches >= min_normalized_matches]
                if not valid:
                    continue
                index_diversity = min(abs(candidate - existing) for existing in selected)
                overlap = max(math.log1p(metric.fundamental_inliers) * (0.5 + metric.inlier_ratio) for metric in valid)
                ranked.append((index_diversity * 3.0 + overlap, candidate))
            if not ranked:
                raise ValueError(f"Cannot extend nested subset to {target} frames under overlap constraints")
            selected.add(sorted(ranked, key=lambda item: (-item[0], item[1]))[0][1])
        result[target] = sorted(selected)
    return result


def validate_frozen_subsets(subsets: dict[int, Sequence[int]], available_frames: int) -> None:
    expected = (2, 4, 6, 8, 10)
    if tuple(sorted(subsets)) != expected:
        raise ValueError(f"Frozen subsets must contain counts {expected}")
    previous: set[int] = set()
    for count in expected:
        values = list(subsets[count])
        if len(values) != count or values != sorted(values) or len(set(values)) != count:
            raise ValueError(f"Invalid ordered S{count}: {values}")
        if not previous < set(values) and previous:
            raise ValueError(f"S{count} is not a strict superset of the previous subset")
        if any(index < 0 or index >= available_frames for index in values):
            raise ValueError(f"S{count} contains an unavailable index")
        previous = set(values)


def load_frozen_selection(config_path: Path, scene_name: str, frame_count: int) -> dict[str, Any]:
    import yaml
    record = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if record.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unexpected overlap-aware protocol version")
    scenes = record.get("scenes", {})
    if scene_name not in scenes:
        raise KeyError(f"No frozen overlap-aware scene: {scene_name}")
    raw_subsets = scenes[scene_name]["subsets"]
    subsets = {
        int(key.removeprefix("S")): (value["indices"] if isinstance(value, dict) else value)
        for key, value in raw_subsets.items()
    }
    validate_frozen_subsets(subsets, scenes[scene_name]["image_count"])
    if frame_count not in subsets:
        raise KeyError(f"No frozen S{frame_count} subset for {scene_name}")
    subset_record = raw_subsets[f"S{frame_count}"]
    filenames = (
        subset_record["filenames"]
        if isinstance(subset_record, dict)
        else scenes[scene_name]["filenames"][f"S{frame_count}"]
    )
    diagnostics = (
        subset_record.get("stats", {}) if isinstance(subset_record, dict)
        else scenes[scene_name].get("statistics", {}).get(f"S{frame_count}", {})
    )
    return {
        "protocol_version": record["protocol_version"],
        "indices": list(subsets[frame_count]),
        "filenames": list(filenames),
        "diagnostics": diagnostics,
    }


def save_pair_cache(path: Path, scene_name: str, geometry: Sequence[FrameGeometry], pairs: Sequence[PairMetrics], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "scene": scene_name,
        "metadata": metadata,
        "frames": [asdict(item) for item in geometry],
        "pairs": [asdict(item) for item in pairs],
    }, indent=2) + "\n", encoding="utf-8")


def load_pair_cache(path: Path) -> tuple[list[FrameGeometry], list[PairMetrics], dict[str, Any]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Incompatible overlap cache protocol")
    return ([FrameGeometry(**item) for item in record["frames"]],
            [PairMetrics(**item) for item in record["pairs"]], record["metadata"])
