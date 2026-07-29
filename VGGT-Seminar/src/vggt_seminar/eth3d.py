from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .experiments import evenly_spaced_indices


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ETH3DCamera:
    camera_id: int
    model: str
    width: int
    height: int
    parameters: tuple[float, ...]


@dataclass(frozen=True)
class ETH3DPose:
    image_id: int
    quaternion_wxyz: tuple[float, float, float, float]
    translation_xyz: tuple[float, float, float]
    camera_id: int
    image_name: str


@dataclass(frozen=True)
class ETH3DScene:
    name: str
    root: Path
    image_paths: tuple[Path, ...]
    cameras: dict[int, ETH3DCamera]
    poses: tuple[ETH3DPose, ...]
    mask_paths: tuple[Path, ...]
    clean_scan_paths: tuple[Path, ...]
    evaluation_scan_paths: tuple[Path, ...]
    occlusion_paths: tuple[Path, ...]
    image_size: tuple[int, int]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "image_count": len(self.image_paths),
            "image_size": list(self.image_size),
            "camera_count": len(self.cameras),
            "pose_count": len(self.poses),
            "mask_count": len(self.mask_paths),
            "clean_scan_count": len(self.clean_scan_paths),
            "evaluation_scan_count": len(self.evaluation_scan_paths),
            "occlusion_file_count": len(self.occlusion_paths),
            "calibration_available": bool(self.cameras),
            "poses_available": bool(self.poses),
            "laser_scan_available": bool(self.clean_scan_paths or self.evaluation_scan_paths),
            "evaluation_masks_available": bool(self.mask_paths),
        }


def load_dataset_manifest(dataset_root: Path) -> dict[str, Any]:
    path = Path(dataset_root) / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ETH3D dataset manifest: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("dataset_id") != "eth3d_high_res_multi_view_training":
        raise ValueError("Manifest is not the expected ETH3D high-resolution training subset")
    scenes = record.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("ETH3D manifest must contain a non-empty scenes list")
    return record


def _scene_paths(scene_root: Path) -> tuple[Path, Path, Path]:
    image_dir = scene_root / "images" / "dslr_images_undistorted"
    calibration_dir = scene_root / "dslr_calibration_undistorted"
    mask_dir = scene_root / "masks_for_images" / "dslr_images"
    return image_dir, calibration_dir, mask_dir


def discover_scenes(dataset_root: Path) -> list[str]:
    root = Path(dataset_root)
    if not root.is_dir():
        return []
    result = []
    for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
        image_dir, calibration_dir, _ = _scene_paths(candidate)
        if image_dir.is_dir() and (calibration_dir / "cameras.txt").is_file() and (calibration_dir / "images.txt").is_file():
            result.append(candidate.name)
    return result


def load_cameras(path: Path) -> dict[int, ETH3DCamera]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ETH3D camera calibration: {path}")
    cameras: dict[int, ETH3DCamera] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Malformed camera row: {line}")
        camera_id = int(fields[0])
        cameras[camera_id] = ETH3DCamera(
            camera_id=camera_id,
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            parameters=tuple(float(value) for value in fields[4:]),
        )
    if not cameras:
        raise ValueError(f"No camera calibration rows in {path}")
    return cameras


def load_camera_poses(path: Path) -> list[ETH3DPose]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ETH3D camera poses: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) % 2:
        raise ValueError(f"COLMAP images file must contain pose/observation line pairs: {path}")
    poses = []
    for line in lines[::2]:
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Malformed image pose row: {line}")
        poses.append(ETH3DPose(
            image_id=int(fields[0]),
            quaternion_wxyz=tuple(float(value) for value in fields[1:5]),
            translation_xyz=tuple(float(value) for value in fields[5:8]),
            camera_id=int(fields[8]),
            image_name=" ".join(fields[9:]),
        ))
    return sorted(poses, key=lambda pose: (Path(pose.image_name).name.lower(), pose.image_id))


def load_scene(dataset_root: Path, scene_name: str) -> ETH3DScene:
    scene_root = Path(dataset_root) / scene_name
    if not scene_root.is_dir():
        raise FileNotFoundError(f"ETH3D scene not found: {scene_root}")
    image_dir, calibration_dir, mask_dir = _scene_paths(scene_root)
    cameras = load_cameras(calibration_dir / "cameras.txt")
    poses = load_camera_poses(calibration_dir / "images.txt")
    available = {path.name.lower(): path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    image_paths = []
    for pose in poses:
        name = Path(pose.image_name).name.lower()
        if name not in available:
            raise FileNotFoundError(f"Pose references missing image: {pose.image_name}")
        image_paths.append(available[name])
    if not image_paths:
        raise ValueError(f"No ordered images found for ETH3D scene {scene_name}")
    with Image.open(image_paths[0]) as image:
        image_size = image.size
    mask_paths = tuple(sorted(path for path in mask_dir.glob("*") if path.is_file())) if mask_dir.is_dir() else ()
    clean_scans = tuple(sorted((scene_root / "scan_clean").glob("*.ply")))
    evaluation_scans = tuple(sorted((scene_root / "dslr_scan_eval").glob("*.ply")))
    occlusion_paths = tuple(sorted(path for path in (scene_root / "occlusion").glob("*") if path.is_file()))
    unknown_camera_ids = sorted({pose.camera_id for pose in poses} - cameras.keys())
    if unknown_camera_ids:
        raise ValueError(f"Poses reference unknown camera IDs: {unknown_camera_ids}")
    return ETH3DScene(
        name=scene_name,
        root=scene_root,
        image_paths=tuple(image_paths),
        cameras=cameras,
        poses=tuple(poses),
        mask_paths=mask_paths,
        clean_scan_paths=clean_scans,
        evaluation_scan_paths=evaluation_scans,
        occlusion_paths=occlusion_paths,
        image_size=image_size,
    )


def select_frames(images: Sequence[Path], count: int, strategy: str = "evenly_spaced") -> list[Path]:
    if not 1 <= count <= len(images):
        raise ValueError(f"count must be between 1 and {len(images)}, got {count}")
    if strategy == "sequential":
        return list(images[:count])
    if strategy == "evenly_spaced":
        return [images[index] for index in evenly_spaced_indices(len(images), count)]
    raise ValueError(f"Unknown ETH3D selection strategy: {strategy}")


def apply_order(images: Sequence[Path], order: str, seed: int = 42) -> list[Path]:
    result = list(images)
    if order == "original":
        return result
    if order == "reversed":
        return list(reversed(result))
    if order == "shuffled":
        random.Random(seed).shuffle(result)
        return result
    raise ValueError(f"Unknown ETH3D order: {order}")


def build_experiment_configurations(
    available_frames: int,
    frame_counts: Sequence[int] = (2, 4, 6, 8, 10),
    selection_strategies: Sequence[str] = ("sequential", "evenly_spaced"),
    orders: Sequence[str] = ("original", "reversed", "shuffled"),
    seed: int = 42,
) -> list[dict[str, Any]]:
    configurations = []
    for count in frame_counts:
        if count > available_frames:
            continue
        for strategy in selection_strategies:
            for order in orders:
                configurations.append({
                    "frame_count": count,
                    "selection_strategy": strategy,
                    "order": order,
                    "seed": seed,
                })
    return configurations


def scene_summary(scene: ETH3DScene) -> dict[str, Any]:
    record = scene.metadata()
    record["cameras"] = [asdict(camera) for camera in scene.cameras.values()]
    return record

