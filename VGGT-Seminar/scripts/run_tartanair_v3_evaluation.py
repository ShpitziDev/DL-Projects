from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_seminar.tartanair import (
    apply_similarity,
    camera_to_world_opencv,
    load_trajectory,
    read_depth,
    resize_square_ground_truth,
    scale_aligned_depth_metrics,
    umeyama_similarity,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / os.environ.get("VGGT_V3_CONFIG", "configs/experiments/v3_tartanair_frozen.yaml")
CHECKPOINT_OVERRIDE = os.environ.get("VGGT_V3_CHECKPOINT")
OUTPUT_OVERRIDE = os.environ.get("VGGT_V3_OUTPUT")
HEADS_OVERRIDE = os.environ.get("VGGT_V3_HEADS")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_metrics(pred_extrinsic: np.ndarray, gt_poses: np.ndarray) -> dict:
    pred_c2w = np.linalg.inv(
        np.concatenate(
            [pred_extrinsic, np.tile(np.array([[[0, 0, 0, 1]]]), (len(pred_extrinsic), 1, 1))],
            axis=1,
        )
    )
    gt_c2w = np.stack([camera_to_world_opencv(p) for p in gt_poses])
    pred_centers, gt_centers = pred_c2w[:, :3, 3], gt_c2w[:, :3, 3]
    alignment = umeyama_similarity(pred_centers, gt_centers)
    aligned_centers = apply_similarity(pred_centers, alignment)
    errors = np.linalg.norm(aligned_centers - gt_centers, axis=1)
    gauge_rotation = gt_c2w[0, :3, :3] @ pred_c2w[0, :3, :3].T
    rotation_errors = []
    for pred, gt in zip(pred_c2w, gt_c2w):
        delta = gt[:3, :3].T @ gauge_rotation @ pred[:3, :3]
        cosine = np.clip((np.trace(delta) - 1) / 2, -1, 1)
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))
    pred_steps = np.linalg.norm(np.diff(aligned_centers, axis=0), axis=1)
    gt_steps = np.linalg.norm(np.diff(gt_centers, axis=0), axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "ate_mean_m": float(errors.mean()),
        "relative_translation_rmse_m": float(np.sqrt(np.mean((pred_steps - gt_steps) ** 2)))
        if len(pred_steps) else 0.0,
        "rotation_mean_deg": float(np.mean(rotation_errors)),
        "rotation_median_deg": float(np.median(rotation_errors)),
        "rotation_errors_deg": rotation_errors,
        "similarity_scale": alignment[0],
        "aligned_centers": aligned_centers.tolist(),
        "gt_centers": gt_centers.tolist(),
    }


def confidence_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, confidence: np.ndarray, scale: float) -> dict:
    valid_error = np.abs(pred[mask] * scale - gt[mask]) / gt[mask]
    valid_conf = confidence[mask]
    correlation = spearmanr(valid_conf, valid_error).statistic
    edges = np.quantile(valid_conf, np.linspace(0, 1, 6))
    bins = []
    for index in range(5):
        include = (valid_conf >= edges[index]) & (
            valid_conf <= edges[index + 1] if index == 4 else valid_conf < edges[index + 1]
        )
        bins.append({
            "bin": index + 1,
            "count": int(include.sum()),
            "confidence_mean": float(valid_conf[include].mean()),
            "abs_rel_mean": float(valid_error[include].mean()),
        })
    return {"spearman_confidence_vs_abs_rel": float(correlation), "quintiles": bins}


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    output = ROOT / (OUTPUT_OVERRIDE or config["execution"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / (CHECKPOINT_OVERRIDE or config["model"]["checkpoint"])
    expected_hash = None if CHECKPOINT_OVERRIDE else config["model"]["checkpoint_sha256"]
    actual_hash = sha256(checkpoint)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError("Checkpoint checksum mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")

    data = ROOT / "local_assets/datasets/tartanair_v2/ArchVizTinyHouseDay/Data_easy"
    trajectory = load_trajectory(data, "P000")
    device = torch.device("cuda:0")
    model = VGGT(enable_camera=True, enable_depth=True, enable_point=False, enable_track=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    incompat = model.load_state_dict(state, strict=False)
    if incompat.missing_keys:
        raise RuntimeError(f"Missing checkpoint keys: {incompat.missing_keys}")
    heads_hash = None
    if HEADS_OVERRIDE:
        heads_path = ROOT / HEADS_OVERRIDE
        heads_hash = sha256(heads_path)
        heads = torch.load(heads_path, map_location="cpu", weights_only=True)
        model.camera_head.load_state_dict(heads["camera_head"], strict=True)
        model.depth_head.load_state_dict(heads["depth_head"], strict=True)
    model.eval().to(device)
    torch.backends.cuda.enable_flash_sdp(False)

    rows = []
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": actual_hash,
        "adapted_heads": HEADS_OVERRIDE,
        "adapted_heads_sha256": heads_hash,
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
    }
    for subset, indices in config["selection"]["subsets"].items():
        subset_dir = output / subset
        if (subset_dir / "metrics.json").is_file() and (subset_dir / "predictions.pt").is_file():
            metrics = json.loads((subset_dir / "metrics.json").read_text())
            if metrics["indices"] != indices or metrics["checkpoint_sha256"] != actual_hash:
                raise RuntimeError(f"Incompatible preserved result: {subset_dir}")
            rows.append({
                "subset": subset,
                "views": len(indices),
                "depth_abs_rel": metrics["depth"]["abs_rel"],
                "depth_rmse": metrics["depth"]["rmse"],
                "depth_delta1": metrics["depth"]["delta1"],
                "camera_ate_rmse_m": metrics["camera"]["ate_rmse_m"],
                "relative_translation_rmse_m": metrics["camera"]["relative_translation_rmse_m"],
                "rotation_mean_deg": metrics["camera"]["rotation_mean_deg"],
                "confidence_error_spearman": metrics["confidence"]["spearman_confidence_vs_abs_rel"],
                "inference_seconds": metrics["inference_seconds"],
                "peak_allocated_gib": metrics["peak_allocated_gib"],
            })
            continue
        if subset_dir.exists():
            raise RuntimeError(f"Partial subset directory requires inspection: {subset_dir}")
        subset_dir.mkdir()
        paths = [trajectory.image_paths[i] for i in indices]
        images = load_and_preprocess_images(paths, mode=config["preprocessing_mode"]).to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"].float(), images.shape[-2:]
        )
        pred_depth = predictions["depth"][0, ..., 0].float().cpu().numpy()
        confidence = predictions["depth_conf"][0].float().cpu().numpy()
        gt_depths, masks = [], []
        for index in indices:
            depth, mask, _ = resize_square_ground_truth(read_depth(trajectory.depth_paths[index]), 518)
            gt_depths.append(depth)
            masks.append(mask)
        gt_depth = np.stack(gt_depths)
        mask = np.stack(masks) & np.isfinite(pred_depth) & (pred_depth > 0)
        depth_metrics = scale_aligned_depth_metrics(pred_depth, gt_depth, mask)
        cameras = camera_metrics(extrinsic[0].float().cpu().numpy(), trajectory.poses_xyzw[indices])
        calibration = confidence_metrics(
            pred_depth, gt_depth, mask, confidence, depth_metrics["scale"]
        )
        metrics = {
            "subset": subset,
            "indices": indices,
            "view_count": len(indices),
            "inference_seconds": elapsed,
            "peak_allocated_gib": peak_allocated / 2**30,
            "peak_reserved_gib": peak_reserved / 2**30,
            "depth": depth_metrics,
            "camera": cameras,
            "confidence": calibration,
            "checkpoint_sha256": actual_hash,
            "adapted_heads_sha256": heads_hash,
        }
        torch.save(
            {
                "pose_enc": predictions["pose_enc"].cpu(),
                "depth": predictions["depth"].cpu(),
                "depth_conf": predictions["depth_conf"].cpu(),
                "extrinsic": extrinsic.cpu(),
                "intrinsic": intrinsic.cpu(),
            },
            subset_dir / "predictions.pt",
        )
        (subset_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (subset_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump({**config, "resolved_subset": subset, "resolved_indices": indices}, sort_keys=False)
        )
        rows.append({
            "subset": subset,
            "views": len(indices),
            "depth_abs_rel": depth_metrics["abs_rel"],
            "depth_rmse": depth_metrics["rmse"],
            "depth_delta1": depth_metrics["delta1"],
            "camera_ate_rmse_m": cameras["ate_rmse_m"],
            "relative_translation_rmse_m": cameras["relative_translation_rmse_m"],
            "rotation_mean_deg": cameras["rotation_mean_deg"],
            "confidence_error_spearman": calibration["spearman_confidence_vs_abs_rel"],
            "inference_seconds": elapsed,
            "peak_allocated_gib": peak_allocated / 2**30,
        })
        del predictions, images
        gc.collect()
        torch.cuda.empty_cache()
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (output / "SUCCESS").write_text("validated\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
