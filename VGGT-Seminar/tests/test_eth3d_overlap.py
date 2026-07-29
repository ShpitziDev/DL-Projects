import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from vggt_seminar.eth3d import ETH3DPose
from vggt_seminar.eth3d_overlap import (
    PROTOCOL_VERSION, PairMetrics, build_nested_subsets, camera_center,
    load_frozen_selection, match_orb_pair, quaternion_wxyz_to_rotation,
    relative_rotation_deg, validate_frozen_subsets, vector_angle_deg,
    viewing_direction, window_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def _pose(index: int, tx: float = 0.0, quaternion=(1.0, 0.0, 0.0, 0.0)) -> ETH3DPose:
    return ETH3DPose(index, quaternion, (tx, 0.0, 0.0), 1, f"{index:03d}.jpg")


def _pair(first: int, second: int, inliers: int = 80, matches: int = 100) -> PairMetrics:
    return PairMetrics(first, second, float(second - first), float((second - first) * 2),
                       float((second - first) * 2), 500, 500, matches, matches / 500,
                       inliers, inliers / matches)


def test_camera_center_direction_distance_and_angles() -> None:
    pose = _pose(1, tx=2.0)
    assert np.allclose(quaternion_wxyz_to_rotation(pose.quaternion_wxyz), np.eye(3))
    assert np.allclose(camera_center(pose), [-2, 0, 0])
    assert np.allclose(viewing_direction(pose), [0, 0, 1])
    assert vector_angle_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90)
    assert relative_rotation_deg(_pose(1), _pose(2)) == pytest.approx(0)


def test_deterministic_orb_matching_on_tiny_fixture() -> None:
    image = np.zeros((240, 320), dtype=np.uint8)
    for index in range(25):
        cv2.circle(image, (20 + (index % 5) * 55, 20 + (index // 5) * 42), 8, 255, 2)
        cv2.putText(image, str(index), (10 + (index % 5) * 55, 35 + (index // 5) * 42), cv2.FONT_HERSHEY_SIMPLEX, .4, 180, 1)
    detector = cv2.ORB_create(nfeatures=500)
    points, descriptors = detector.detectAndCompute(image, None)
    first = match_orb_pair(points, descriptors, points, descriptors)
    second = match_orb_pair(points, descriptors, points, descriptors)
    assert first == second
    assert first[0] > 20 and first[2] > 20


def test_window_scoring_and_nested_construction() -> None:
    pairs = [_pair(first, second) for first in range(10) for second in range(first + 1, min(10, first + 12))]
    assert window_summary("hybrid", 0, 10, pairs).score > 0
    subsets = build_nested_subsets(range(10), pairs)
    validate_frozen_subsets(subsets, 10)
    assert all(values == sorted(values) for values in subsets.values())
    assert set(subsets[2]) < set(subsets[4]) < set(subsets[6]) < set(subsets[8]) < set(subsets[10])
    assert subsets == build_nested_subsets(range(10), pairs)


def test_nested_selection_rejects_low_overlap_and_near_duplicates() -> None:
    low = [_pair(first, second, inliers=2, matches=5) for first in range(10) for second in range(first + 1, 10)]
    with pytest.raises(ValueError):
        build_nested_subsets(range(10), low)
    adjacent_only = [_pair(index, index + 1) for index in range(9)]
    with pytest.raises(ValueError):
        build_nested_subsets(range(10), adjacent_only, min_gap=2)


def test_frozen_config_and_missing_count(tmp_path: Path) -> None:
    subsets = {f"S{count}": list(range(count)) for count in (2, 4, 6, 8, 10)}
    record = {"protocol_version": PROTOCOL_VERSION, "scenes": {"tiny": {
        "image_count": 10, "subsets": subsets,
        "filenames": {key: [f"{i}.jpg" for i in values] for key, values in subsets.items()},
        "statistics": {key: {} for key in subsets},
    }}}
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    assert load_frozen_selection(path, "tiny", 4)["indices"] == [0, 1, 2, 3]
    with pytest.raises(KeyError):
        load_frozen_selection(path, "tiny", 3)


def test_tracked_phase6_1_config_is_strict_and_nested() -> None:
    config_path = ROOT / "configs/experiments/phase6_1_overlap_aware_frames.yaml"
    for scene in ("delivery_area", "courtyard"):
        previous: set[int] = set()
        for count in (2, 4, 6, 8, 10):
            record = load_frozen_selection(config_path, scene, count)
            assert record["protocol_version"] == PROTOCOL_VERSION
            assert len(record["indices"]) == count
            assert len(record["filenames"]) == count
            if previous:
                assert previous < set(record["indices"])
            previous = set(record["indices"])


def test_phase6_1_notebook_uses_frozen_selection_without_fallback() -> None:
    notebook = json.loads((ROOT / "notebooks/02_vggt_multi_input_experiments.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'SELECTED_SELECTION_STRATEGY = "overlap_aware_nested"' in source
    assert "load_frozen_selection(OVERLAP_CONFIG" in source
    assert "Frozen filenames do not match" in source
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
