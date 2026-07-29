from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from .live_demo import SUPPORTED_IMAGES


REQUIRED_MANIFEST_FIELDS = {
    "scene_id", "title", "category", "description", "source", "capture_device",
    "scene_behavior", "ordered_images", "reference_image", "known_challenges", "notes",
    "ground_truth", "redistribution_permission",
}
SCENE_CATEGORIES = {
    "controlled_object", "indoor_scene", "outdoor_scene", "textureless_scene",
    "reflective_scene", "dynamic_scene",
}


@dataclass(frozen=True)
class SceneStatus:
    name: str
    directory: str
    image_count: int
    manifest_exists: bool
    valid: bool
    issue: str | None


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scene_manifest(scene_dir: Path) -> dict[str, Any]:
    path = scene_dir / "scene_manifest.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing scene manifest: {path}")
    manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if missing:
        raise ValueError(f"Manifest is missing fields: {missing}")
    if manifest["category"] not in SCENE_CATEGORIES:
        raise ValueError(f"Unsupported category: {manifest['category']!r}")
    if manifest["scene_behavior"] not in {"static", "dynamic"}:
        raise ValueError("scene_behavior must be 'static' or 'dynamic'")
    ordered = manifest["ordered_images"]
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("ordered_images must be a non-empty list")
    if len(ordered) != len(set(ordered)):
        raise ValueError("ordered_images contains duplicates")
    if manifest["reference_image"] not in ordered:
        raise ValueError("reference_image must appear in ordered_images")
    paths = [scene_dir / name for name in ordered]
    missing_files = [path.name for path in paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Manifest images are missing: {missing_files}")
    invalid = [path.name for path in paths if path.suffix.lower() not in SUPPORTED_IMAGES]
    if invalid:
        raise ValueError(f"Unsupported image extensions: {invalid}")
    return manifest


def ordered_scene_images(scene_dir: Path, manifest: dict[str, Any] | None = None) -> list[Path]:
    record = manifest or load_scene_manifest(scene_dir)
    return [scene_dir / name for name in record["ordered_images"]]


def inventory_scenes(custom_inputs: Path) -> list[SceneStatus]:
    statuses: list[SceneStatus] = []
    for scene_dir in sorted(path for path in custom_inputs.iterdir() if path.is_dir()):
        image_count = sum(path.suffix.lower() in SUPPORTED_IMAGES for path in scene_dir.iterdir() if path.is_file())
        manifest_exists = (scene_dir / "scene_manifest.yaml").is_file()
        issue = None
        try:
            if image_count == 0:
                raise ValueError("no images")
            load_scene_manifest(scene_dir)
        except (FileNotFoundError, ValueError) as error:
            issue = str(error)
        statuses.append(SceneStatus(scene_dir.name, str(scene_dir), image_count, manifest_exists, issue is None, issue))
    return statuses


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if not 1 <= count <= total:
        raise ValueError(f"count must be between 1 and {total}, got {count}")
    if count == 1:
        return [0]
    indices = np.rint(np.linspace(0, total - 1, count)).astype(int).tolist()
    if len(set(indices)) != count:
        raise RuntimeError("Even frame selection produced duplicate indices")
    return indices


def select_evenly(images: Sequence[Path], count: int) -> list[Path]:
    return [images[index] for index in evenly_spaced_indices(len(images), count)]


def ordered_variant(images: Sequence[Path], mode: str, seed: int = 42) -> list[Path]:
    result = list(images)
    if mode == "original":
        return result
    if mode == "reversed":
        return list(reversed(result))
    if mode == "rotate_first":
        return result[1:] + result[:1]
    if mode == "shuffled":
        generator = np.random.default_rng(seed)
        generator.shuffle(result)
        return result
    raise ValueError(f"Unknown order variant: {mode}")


def degrade_image(image: Image.Image, mode: str, severity: float = 0.5) -> Image.Image:
    if not 0 < severity <= 1:
        raise ValueError("severity must be in (0, 1]")
    rgb = np.asarray(image.convert("RGB"))
    if mode == "none":
        return image.convert("RGB")
    if mode == "blur":
        kernel = int(3 + severity * 18) | 1
        return Image.fromarray(cv2.GaussianBlur(rgb, (kernel, kernel), 0))
    if mode == "low_light":
        factor = 1.0 - 0.8 * severity
        return Image.fromarray(np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8))
    if mode == "low_resolution":
        factor = max(0.1, 1.0 - 0.85 * severity)
        small = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))), Image.Resampling.BICUBIC)
        return small.resize(image.size, Image.Resampling.BICUBIC).convert("RGB")
    if mode == "jpeg":
        encode_quality = max(5, round(95 - 85 * severity))
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
        if not success:
            raise RuntimeError("JPEG encoding failed")
        return Image.fromarray(cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    raise ValueError(f"Unknown degradation mode: {mode}")


def normalized_point_disagreement(direct: np.ndarray, unprojected: np.ndarray) -> dict[str, float]:
    a = np.asarray(direct, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(unprojected, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not valid.any():
        raise ValueError("No mutually finite points")
    a, b = a[valid], b[valid]
    distances = np.linalg.norm(a - b, axis=1)
    scale = np.median(np.linalg.norm(b - np.median(b, axis=0), axis=1))
    scale = max(float(scale), 1e-12)
    return {
        "finite_fraction": float(valid.mean()),
        "median_distance": float(np.median(distances)),
        "p90_distance": float(np.percentile(distances, 90)),
        "median_distance_normalized": float(np.median(distances) / scale),
        "p90_distance_normalized": float(np.percentile(distances, 90) / scale),
    }


def confidence_summary(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float().cpu().numpy().reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("Confidence tensor has no finite values")
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def input_manifest(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    return {"files": [{"path": str(path.relative_to(root)).replace("\\", "/"),
                        "size_bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in paths]}


def statuses_as_dicts(statuses: Sequence[SceneStatus]) -> list[dict[str, Any]]:
    return [asdict(status) for status in statuses]
