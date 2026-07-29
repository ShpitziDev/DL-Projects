from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from vggt_seminar.experiments import (
    degrade_image, evenly_spaced_indices, inventory_scenes, load_scene_manifest,
    normalized_point_disagreement, ordered_variant,
)


def _scene(tmp_path: Path) -> Path:
    scene = tmp_path / "controlled_object"
    scene.mkdir()
    for name in ("001.jpg", "002.jpg", "003.jpg"):
        Image.new("RGB", (8, 8), "white").save(scene / name)
    manifest = {
        "scene_id": "test", "title": "Test", "category": "controlled_object",
        "description": "test", "source": "test", "capture_device": "test",
        "scene_behavior": "static", "ordered_images": ["001.jpg", "002.jpg", "003.jpg"],
        "reference_image": "001.jpg", "known_challenges": [], "notes": "",
        "ground_truth": {"available": False, "description": None},
        "redistribution_permission": "private",
    }
    (scene / "scene_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return scene


def test_manifest_inventory_and_frame_selection(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    assert load_scene_manifest(scene)["scene_id"] == "test"
    assert inventory_scenes(tmp_path)[0].valid
    assert evenly_spaced_indices(8, 4) == [0, 2, 5, 7]


def test_order_variants_are_deterministic() -> None:
    items = [Path(str(index)) for index in range(5)]
    assert ordered_variant(items, "reversed") == list(reversed(items))
    assert ordered_variant(items, "shuffled", seed=7) == ordered_variant(items, "shuffled", seed=7)
    with pytest.raises(ValueError):
        ordered_variant(items, "unknown")


def test_degradations_preserve_size() -> None:
    image = Image.new("RGB", (32, 24), "gray")
    for mode in ("blur", "low_light", "low_resolution", "jpeg"):
        assert degrade_image(image, mode).size == image.size


def test_point_disagreement_is_zero_for_identical_maps() -> None:
    points = np.arange(60, dtype=np.float64).reshape(4, 5, 3)
    metrics = normalized_point_disagreement(points, points.copy())
    assert metrics["finite_fraction"] == 1.0
    assert metrics["median_distance_normalized"] == 0.0
