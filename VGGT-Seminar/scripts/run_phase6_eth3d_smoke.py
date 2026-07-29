from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt_seminar.eth3d import load_scene, select_frames
from vggt_seminar.eth3d_overlap import PROTOCOL_VERSION, load_frozen_selection
from vggt_seminar.live_demo import point_cloud_preview, save_heatmap, save_ply, verify_local_assets


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / os.environ.get("VGGT_SMOKE_CONFIG", "configs/experiments/phase6_eth3d_smoke.yaml")
ETH3D_ROOT = ROOT / "local_assets/datasets/eth3d"
REQUIRED_OUTPUTS = {
    "pose_enc", "depth", "depth_conf", "world_points", "world_points_conf",
    "images", "track", "vis", "conf",
}


def git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    floating = value.is_floating_point()
    finite = torch.isfinite(value) if floating else torch.ones_like(value, dtype=torch.bool)
    finite_values = value[finite].float()
    count = value.numel()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "mean": float(finite_values.mean().item()),
        "median": float(finite_values.median().item()),
        "finite_percentage": float(100.0 * finite.sum().item() / count),
        "nan_count": int(torch.isnan(value).sum().item()) if floating else 0,
        "infinity_count": int(torch.isinf(value).sum().item()) if floating else 0,
    }


def save_contact_sheet(paths: list[Path], original_indices: list[int], output: Path, title: str) -> dict[str, Any]:
    thumb_size = (640, 426)
    sheet = Image.new("RGB", (thumb_size[0] * len(paths), thumb_size[1] + 70), "white")
    draw = ImageDraw.Draw(sheet)
    statistics = []
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            array = np.asarray(rgb)
            statistics.append({
                "filename": path.name,
                "resolution": list(rgb.size),
                "loaded": True,
                "rgb_min": int(array.min()),
                "rgb_max": int(array.max()),
                "rgb_mean": float(array.mean()),
                "rgb_std": float(array.std()),
            })
            rgb.thumbnail(thumb_size)
            sheet.paste(rgb, (index * thumb_size[0], 40))
        draw.text((index * thumb_size[0] + 8, 8), f"ETH3D index {original_indices[index]}: {path.name}", fill="black")
    draw.text((8, thumb_size[1] + 46), title, fill="black")
    sheet.save(output, quality=90)
    return {"selection_calculation": "round(linspace(0, 43, 2)) = [0, 43]", "frames": statistics}


def camera_centers(extrinsics: torch.Tensor) -> np.ndarray:
    matrices = extrinsics.detach().float().cpu().numpy()[0]
    return np.stack([-matrix[:3, :3].T @ matrix[:3, 3] for matrix in matrices])


def save_camera_plot(centers: np.ndarray, output: Path, title: str) -> None:
    size = 700
    margin = 80
    canvas = np.full((size, size, 3), 250, dtype=np.uint8)
    points = centers[:, [0, 2]]
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pixels = margin + (points - lo) / span * (size - 2 * margin)
    pixels[:, 1] = size - pixels[:, 1]
    cv2.line(canvas, tuple(pixels[0].astype(int)), tuple(pixels[1].astype(int)), (80, 80, 80), 2)
    for index, pixel in enumerate(pixels.astype(int)):
        cv2.circle(canvas, tuple(pixel), 10, (20, 80, 220), -1)
        cv2.putText(canvas, str(index), tuple(pixel + np.array([14, -10])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "VGGT camera centers: arbitrary-scale X-Z projection", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(canvas, title, (25, size - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(output)


def add_visualization_title(path: Path, title: str) -> None:
    with Image.open(path) as source:
        image = source.convert("RGB")
    header = 42
    titled = Image.new("RGB", (image.width, image.height + header), "white")
    titled.paste(image, (0, header))
    ImageDraw.Draw(titled).text((10, 12), title, fill="black")
    titled.save(path)


def preflight(config: dict[str, Any]) -> tuple[dict[str, Any], Any, list[Path], list[int]]:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from project root {ROOT}, got {Path.cwd().resolve()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; Phase 6 refuses CPU fallback")
    phase6_input = {
        "source": "eth3d", "scene": "delivery_area", "frame_count": 2,
        "selection_strategy": "evenly_spaced", "order": "original", "preprocessing_mode": "crop",
    }
    phase6_2_input = {**phase6_input, "selection_strategy": "overlap_aware_nested"}
    if config["input"] not in (phase6_input, phase6_2_input):
        raise ValueError(f"Unapproved Phase 6 input configuration: {config['input']}")
    if config["constraints"]["expected_forward_passes"] != 1:
        raise ValueError("Phase 6 permits exactly one forward pass")
    if config["model"]["flash_sdp_enabled"]:
        raise ValueError("Flash SDPA is not approved")
    output = ROOT / config["output"]["directory"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke-test output: {output}")
    assets = verify_local_assets(ROOT)
    scene = load_scene(ETH3D_ROOT, "delivery_area")
    frozen = None
    if config["input"]["selection_strategy"] == "overlap_aware_nested":
        if config["constraints"].get("allow_selection_fallback") is not False:
            raise ValueError("Overlap-aware selection must explicitly forbid fallback")
        protocol_path = ROOT / config["protocol"]["config"]
        frozen = load_frozen_selection(protocol_path, "delivery_area", 2)
        frozen_s4 = load_frozen_selection(protocol_path, "delivery_area", 4)
        if frozen["protocol_version"] != PROTOCOL_VERSION or config["protocol"]["version"] != PROTOCOL_VERSION:
            raise RuntimeError("Frozen protocol version mismatch")
        if not set(frozen["indices"]) < set(frozen_s4["indices"]):
            raise RuntimeError("Frozen S2 is not a strict subset of S4")
        indices = frozen["indices"]
        selected = [scene.image_paths[index] for index in indices]
        if frozen["filenames"] != config["constraints"]["expected_filenames"]:
            raise RuntimeError("Frozen filenames differ from the approved run configuration")
    else:
        selected = select_frames(scene.image_paths, 2, "evenly_spaced")
        indices = [scene.image_paths.index(path) for path in selected]
    if indices != config["constraints"]["expected_original_indices"]:
        raise RuntimeError(f"Unexpected selected indices: {indices}")
    if [path.name for path in selected] != config["constraints"]["expected_filenames"]:
        raise RuntimeError(f"Unexpected selected filenames: {[path.name for path in selected]}")
    if len(set(selected)) != 2 or any(not path.is_file() for path in selected):
        raise RuntimeError("Selected frames are missing or duplicated")
    free, total = torch.cuda.mem_get_info(0)
    preflight_record = {
        "project_root": str(ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_free_bytes": int(free),
        "cuda_total_bytes": int(total),
        "checkpoint": str(assets["checkpoint"].relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_size_bytes": assets["checkpoint_size_bytes"],
        "checkpoint_sha256": assets["checkpoint_sha256"],
        "vggt_commit": assets["pin"]["commit"],
        "scene": scene.metadata(),
        "selected_indices": indices,
        "selected_filenames": [path.name for path in selected],
        "calibration_available": bool(scene.cameras),
        "poses_available": bool(scene.poses),
        "output_directory": str(output.relative_to(ROOT)).replace("\\", "/"),
        "git_head_before_run": git_text("rev-parse", "HEAD"),
        "git_status_before_run": git_text("status", "--short").splitlines(),
        "frozen_protocol": frozen,
    }
    return preflight_record, assets, selected, indices


def main() -> int:
    total_start = time.perf_counter()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    output = ROOT / config["output"]["directory"]
    preflight_record, assets, selected, indices = preflight(config)
    for directory in (output, output / "arrays", output / "visualizations", output / "logs"):
        directory.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "preflight.json").write_text(json.dumps(preflight_record, indent=2) + "\n", encoding="utf-8")
    strategy = config["input"]["selection_strategy"]
    title = f"delivery_area | S2 | {strategy} | indices {indices} | original"
    frame_record = save_contact_sheet(selected, indices, output / "visualizations/contact_sheet.jpg", title)
    frame_record["original_indices"] = indices
    (output / "selected_frames.json").write_text(json.dumps(frame_record, indent=2) + "\n", encoding="utf-8")
    if "protocol" in config:
        (output / "frozen_protocol.json").write_text(json.dumps(config["protocol"], indent=2) + "\n", encoding="utf-8")

    try:
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.enable_flash_sdp(False)
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        init_start = time.perf_counter()
        model = VGGT()
        architecture_init_seconds = time.perf_counter() - init_start

        load_start = time.perf_counter()
        state = torch.load(assets["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True)
        del state
        checkpoint_load_seconds = time.perf_counter() - load_start

        transfer_start = time.perf_counter()
        model = model.to(device).eval()
        torch.cuda.synchronize()
        gpu_transfer_seconds = time.perf_counter() - transfer_start

        preprocessing_start = time.perf_counter()
        images_cpu = load_and_preprocess_images(selected, mode=config["input"]["preprocessing_mode"])
        images = images_cpu.to(device)
        height, width = images.shape[-2:]
        query = torch.tensor([[[width / 2.0, height / 2.0]]], device=device)
        torch.cuda.synchronize()
        preprocessing_seconds = time.perf_counter() - preprocessing_start

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        baseline_allocated = torch.cuda.memory_allocated(0)
        inference_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(images, query_points=query)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        peak_allocated = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)

        post_start = time.perf_counter()
        missing = sorted(REQUIRED_OUTPUTS - predictions.keys())
        if missing:
            raise RuntimeError(f"Required outputs are missing: {missing}")
        if any(value.shape[1] != 2 for value in predictions.values() if isinstance(value, torch.Tensor) and value.ndim >= 2):
            raise RuntimeError("At least one output does not contain exactly two views")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"].float(), images.shape[-2:])
        if extrinsic.shape != (1, 2, 3, 4) or intrinsic.shape != (1, 2, 3, 3):
            raise RuntimeError(f"Malformed decoded cameras: {extrinsic.shape}, {intrinsic.shape}")
        unprojected_np = unproject_depth_map_to_point_map(
            predictions["depth"].float().squeeze(0).cpu().numpy(),
            extrinsic.squeeze(0).cpu().numpy(), intrinsic.squeeze(0).cpu().numpy(),
        )
        unprojected = torch.from_numpy(unprojected_np)
        centers = camera_centers(extrinsic)
        derived = {"extrinsic": extrinsic, "intrinsic": intrinsic, "unprojected_points": unprojected}
        summaries: dict[str, Any] = {}
        for name, value in {**predictions, **derived}.items():
            if isinstance(value, torch.Tensor):
                summaries[name] = tensor_summary(value)
                if summaries[name]["finite_percentage"] < 99.99:
                    raise RuntimeError(f"Excessive invalid values in {name}: {summaries[name]}")
                torch.save(value.detach().cpu(), output / "arrays" / f"{name}.pt")
            elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
                summaries[name] = {"type": "tensor_list", "length": len(value), "items": [tensor_summary(item) for item in value]}
                for index, item in enumerate(value):
                    torch.save(item.detach().cpu(), output / "arrays" / f"{name}_{index}.pt")
            else:
                raise TypeError(f"Unexpected output type for {name}: {type(value)}")
        (output / "tensor_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        camera_record = {
            "convention": "OpenCV world-to-camera [R|t] in VGGT first-frame reference coordinates",
            "camera_centers_formula": "C = -R^T t",
            "arbitrary_scale": True,
            "requires_similarity_alignment_to_eth3d": True,
            "extrinsic": extrinsic.detach().cpu().tolist(),
            "intrinsic": intrinsic.detach().cpu().tolist(),
            "camera_centers": centers.tolist(),
        }
        (output / "camera_parameters.json").write_text(json.dumps(camera_record, indent=2) + "\n", encoding="utf-8")

        colors = predictions["images"][0].permute(0, 2, 3, 1).float().cpu().numpy()
        direct = predictions["world_points"][0].float().cpu().numpy()
        confidence = predictions["world_points_conf"][0].float().cpu().numpy()
        threshold = float(np.median(confidence))
        filtered = direct.copy()
        filtered[confidence <= threshold] = np.nan
        for view in range(2):
            save_heatmap(predictions["depth"][0, view], output / "visualizations" / f"depth_view{view}.png", invert=True)
            save_heatmap(predictions["depth_conf"][0, view], output / "visualizations" / f"depth_confidence_view{view}.png")
            save_heatmap(predictions["world_points_conf"][0, view], output / "visualizations" / f"point_confidence_view{view}.png")
        save_camera_plot(centers, output / "visualizations/camera_centers.png", title)
        point_cloud_preview(direct, colors).save(output / "visualizations/point_cloud_direct_preview.png")
        point_cloud_preview(unprojected_np, colors).save(output / "visualizations/point_cloud_depth_unprojected_preview.png")
        point_cloud_preview(filtered, colors).save(output / "visualizations/point_cloud_confidence_filtered_preview.png")
        for visualization in (
            *(f"depth_view{view}.png" for view in range(2)),
            *(f"depth_confidence_view{view}.png" for view in range(2)),
            *(f"point_confidence_view{view}.png" for view in range(2)),
            "point_cloud_direct_preview.png", "point_cloud_depth_unprojected_preview.png",
            "point_cloud_confidence_filtered_preview.png",
        ):
            add_visualization_title(output / "visualizations" / visualization, title)
        direct_count = save_ply(direct, colors, output / "visualizations/point_cloud_direct.ply")
        unprojected_count = save_ply(unprojected_np, colors, output / "visualizations/point_cloud_depth_unprojected.ply")
        filtered_count = save_ply(filtered, colors, output / "visualizations/point_cloud_confidence_filtered.ply")
        postprocessing_seconds = time.perf_counter() - post_start
        total_elapsed_seconds = time.perf_counter() - total_start
        retained = int(np.isfinite(filtered).all(axis=-1).sum())
        before = int(confidence.size)
        runtime = {
            "status": "passed",
            "forward_pass_count": 1,
            "architecture_init_seconds": architecture_init_seconds,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "gpu_transfer_seconds": gpu_transfer_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "inference_seconds": inference_seconds,
            "postprocessing_seconds": postprocessing_seconds,
            "total_elapsed_seconds": total_elapsed_seconds,
            "baseline_allocated_vram_bytes": int(baseline_allocated),
            "peak_allocated_vram_bytes": int(peak_allocated),
            "peak_reserved_vram_bytes": int(peak_reserved),
            "precision": "torch.bfloat16 autocast",
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "preprocessed_shape": list(images.shape),
            "point_counts": {"direct": direct_count, "depth_unprojected": unprojected_count, "confidence_filtered": filtered_count},
            "confidence_filter": {"rule": "world point confidence > median", "threshold": threshold,
                                  "points_before": before, "points_retained": retained,
                                  "retained_percentage": 100.0 * retained / before},
        }
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
        (output / "logs/execution.log").write_text(
            "Phase 6 delivery_area two-view smoke test passed. Exactly one model forward call executed.\n",
            encoding="utf-8",
        )
        (output / "logs/error.log").write_text("", encoding="utf-8")
        print(json.dumps(runtime, indent=2))
        del predictions, derived, direct, unprojected, unprojected_np, images, images_cpu, model
        gc.collect()
        torch.cuda.empty_cache()
        return 0
    except Exception:
        (output / "logs/error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
