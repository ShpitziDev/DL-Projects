from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image

from .external import checkout_status, load_vggt_pin


SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_assets(root: Path) -> dict[str, Any]:
    manifest_path = root / "local_assets/checkpoints/checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = root / manifest["relative_path"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    pin = load_vggt_pin(root)
    checkout = checkout_status(root)
    if not checkout["exists"] or not checkout["matches_pin"] or not checkout["clean"]:
        raise RuntimeError(f"VGGT checkout does not match the clean recorded pin: {checkout}")
    size_matches = checkpoint.stat().st_size == manifest["size_bytes"]
    actual_sha256 = sha256_file(checkpoint)
    hash_matches = actual_sha256 == manifest["sha256"]
    if not size_matches or not hash_matches:
        raise RuntimeError(
            f"Checkpoint verification failed: size_matches={size_matches}, hash_matches={hash_matches}"
        )
    return {
        "manifest": manifest,
        "pin": pin,
        "checkout": checkout,
        "checkpoint": checkpoint,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": actual_sha256,
        "checkpoint_hash_matches": hash_matches,
    }


def discover_images(scene_dir: Path, image_glob: str = "*", max_images: int | None = None) -> list[Path]:
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
    images = sorted(path for path in scene_dir.glob(image_glob) if path.suffix.lower() in SUPPORTED_IMAGES)
    if max_images is not None:
        if max_images < 1:
            raise ValueError("MAX_IMAGES must be None or a positive integer")
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"No supported images matched {image_glob!r} in {scene_dir}")
    return images


def finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def tensor_record(value: torch.Tensor) -> dict[str, object]:
    tensor = value.detach()
    floating = tensor.is_floating_point()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": finite_tensor(tensor),
        "nan_count": int(torch.isnan(tensor).sum().item()) if floating else 0,
        "inf_count": int(torch.isinf(tensor).sum().item()) if floating else 0,
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def prediction_schema(predictions: dict[str, Any], derived: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"schema_version": 1, "outputs": {}, "derived": {}}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            schema["outputs"][key] = tensor_record(value)
        elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
            schema["outputs"][key] = {
                "type": "tensor_list", "length": len(value),
                "items": [tensor_record(item) for item in value],
            }
        else:
            raise TypeError(f"Unexpected prediction type for {key}: {type(value)}")
    for key, value in (derived or {}).items():
        schema["derived"][key] = tensor_record(value)
    return schema


def heatmap_rgb(tensor: torch.Tensor, invert: bool = False) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy().squeeze()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Cannot visualize an array without finite values")
    lo, hi = np.percentile(finite, [1, 99])
    normalized = np.clip((array - lo) / max(float(hi - lo), 1e-8), 0, 1)
    if invert:
        normalized = 1 - normalized
    bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_heatmap(tensor: torch.Tensor, path: Path, invert: bool = False) -> None:
    Image.fromarray(heatmap_rgb(tensor, invert=invert)).save(path)


def save_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> int:
    points = points.reshape(-1, 3)
    colors = np.clip(colors.reshape(-1, 3) * 255, 0, 255).astype(np.uint8)
    valid = np.isfinite(points).all(axis=1)
    points, colors = points[valid], colors[valid]
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors):
            stream.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {color[0]} {color[1]} {color[2]}\n")
    return len(points)


def point_cloud_preview(
    points: np.ndarray, colors: np.ndarray, max_points: int = 40_000, size: int = 700
) -> Image.Image:
    xyz = points.reshape(-1, 3)
    rgb = np.clip(colors.reshape(-1, 3) * 255, 0, 255).astype(np.uint8)
    valid = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[valid], rgb[valid]
    if len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz, rgb = xyz[indices], rgb[indices]
    xy = xyz[:, :2]
    lo, hi = np.percentile(xy, [1, 99], axis=0)
    span = np.maximum(hi - lo, 1e-8)
    pixels = np.clip((xy - lo) / span, 0, 1)
    pixels[:, 1] = 1 - pixels[:, 1]
    pixels = (pixels * (size - 1)).astype(np.int32)
    canvas = np.full((size, size, 3), 248, dtype=np.uint8)
    depth_order = np.argsort(xyz[:, 2])[::-1]
    canvas[pixels[depth_order, 1], pixels[depth_order, 0]] = rgb[depth_order]
    return Image.fromarray(canvas)


def center_or_explicit_query(width: int, height: int, xy: Iterable[float] | None, device: torch.device) -> torch.Tensor:
    x, y = (width / 2.0, height / 2.0) if xy is None else tuple(xy)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Tracking query {(x, y)} lies outside {width}x{height}")
    return torch.tensor([[[float(x), float(y)]]], device=device)
