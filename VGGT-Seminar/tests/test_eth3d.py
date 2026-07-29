import json
from pathlib import Path

import pytest
from PIL import Image

from vggt_seminar.eth3d import (
    apply_order,
    build_experiment_configurations,
    discover_scenes,
    load_camera_poses,
    load_dataset_manifest,
    load_scene,
    select_frames,
)


def _fixture(root: Path, scene_name: str = "tiny_scene") -> Path:
    scene = root / scene_name
    images = scene / "images" / "dslr_images_undistorted"
    calibration = scene / "dslr_calibration_undistorted"
    masks = scene / "masks_for_images" / "dslr_images"
    scans = scene / "scan_clean"
    evaluation = scene / "dslr_scan_eval"
    occlusion = scene / "occlusion"
    for directory in (images, calibration, masks, scans, evaluation, occlusion):
        directory.mkdir(parents=True, exist_ok=True)
    names = ["DSC_0003.JPG", "DSC_0001.JPG", "DSC_0002.JPG"]
    for index, name in enumerate(names):
        Image.new("RGB", (12, 8), (index * 40, 80, 120)).save(images / name)
        Image.new("L", (12, 8), 255).save(masks / name)
    (calibration / "cameras.txt").write_text("# cameras\n1 PINHOLE 12 8 10 10 6 4\n", encoding="utf-8")
    pose_rows = []
    for image_id, name in enumerate(names, 1):
        pose_rows.extend([f"{image_id} 1 0 0 0 {image_id} 0 0 1 dslr_images/{name}", "0 0 -1"])
    (calibration / "images.txt").write_text("# images\n" + "\n".join(pose_rows) + "\n", encoding="utf-8")
    (scans / "scan1.ply").write_text("ply\n", encoding="utf-8")
    (evaluation / "scan1.ply").write_text("ply\n", encoding="utf-8")
    (occlusion / "surface_mesh.ply").write_text("ply\n", encoding="utf-8")
    return scene


def test_discovery_scene_loading_and_metadata(tmp_path: Path) -> None:
    _fixture(tmp_path)
    assert discover_scenes(tmp_path) == ["tiny_scene"]
    scene = load_scene(tmp_path, "tiny_scene")
    assert [path.name for path in scene.image_paths] == ["DSC_0001.JPG", "DSC_0002.JPG", "DSC_0003.JPG"]
    assert scene.image_size == (12, 8)
    assert len(scene.cameras) == 1
    assert len(scene.poses) == 3
    assert len(scene.mask_paths) == 3
    assert scene.metadata()["laser_scan_available"]


def test_manifest_parsing(tmp_path: Path) -> None:
    manifest = {"dataset_id": "eth3d_high_res_multi_view_training", "scenes": [{"name": "tiny_scene"}]}
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_dataset_manifest(tmp_path) == manifest
    manifest["dataset_id"] = "wrong"
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset_manifest(tmp_path)


def test_pose_parser_rejects_incomplete_pair(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text("1 1 0 0 0 0 0 0 1 image.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_camera_poses(path)


def test_frame_subset_and_order_generation() -> None:
    images = [Path(str(index)) for index in range(12)]
    assert select_frames(images, 4, "sequential") == images[:4]
    assert select_frames(images, 4, "evenly_spaced") == [images[index] for index in (0, 4, 7, 11)]
    assert apply_order(images[:4], "reversed") == list(reversed(images[:4]))
    assert apply_order(images, "shuffled", 7) == apply_order(images, "shuffled", 7)
    configurations = build_experiment_configurations(8)
    assert {record["frame_count"] for record in configurations} == {2, 4, 6, 8}
    assert {record["selection_strategy"] for record in configurations} == {"sequential", "evenly_spaced"}
    assert {record["order"] for record in configurations} == {"original", "reversed", "shuffled"}


def test_missing_scene_handling(tmp_path: Path) -> None:
    assert discover_scenes(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        load_scene(tmp_path, "missing")
