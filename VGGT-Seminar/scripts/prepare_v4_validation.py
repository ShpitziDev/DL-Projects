from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from vggt_seminar.tartanair import (
    apply_similarity,
    load_trajectory,
    read_depth,
    resize_square_ground_truth,
    scale_aligned_depth_metrics,
    umeyama_similarity,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/v4"
BASE = ROOT / "outputs/experiments/v3_tartanair_pretrained_20260729_sdpa_corrected"
ADAPTED = ROOT / "outputs/experiments/v3_tartanair_adapted_step15_20260729"
TRAIN = ROOT / "outputs/experiments/v3_tartanair_finetune_20260729"
DATA = ROOT / "local_assets/datasets/tartanair_v2/ArchVizTinyHouseDay/Data_easy"
CONFIG = ROOT / "configs/experiments/v3_tartanair_frozen.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def depth_bin_analysis(run: Path, label: str, trajectory, subsets: dict) -> list[dict]:
    rows = []
    ranges = [(0.1, 5.0), (5.0, 20.0), (20.0, 100.0)]
    for subset, indices in subsets.items():
        saved = torch.load(run / subset / "predictions.pt", map_location="cpu", weights_only=True)
        pred = saved["depth"][0, ..., 0].float().numpy()
        gt, valid = [], []
        for index in indices:
            depth, mask, _ = resize_square_ground_truth(read_depth(trajectory.depth_paths[index]), 518)
            gt.append(depth)
            valid.append(mask)
        gt = np.stack(gt)
        valid = np.stack(valid) & np.isfinite(pred) & (pred > 0)
        scale = float(np.median(gt[valid]) / np.median(pred[valid]))
        scaled = pred * scale
        for lower, upper in ranges:
            mask = valid & (gt >= lower) & (gt < upper)
            if not mask.any():
                continue
            ratio = np.maximum(scaled[mask] / gt[mask], gt[mask] / np.maximum(scaled[mask], 1e-8))
            rows.append(
                {
                    "model": label,
                    "subset": subset,
                    "depth_range_m": f"{lower:g}-{upper:g}",
                    "valid_pixels": int(mask.sum()),
                    "abs_rel": float(np.mean(np.abs(scaled[mask] - gt[mask]) / gt[mask])),
                    "rmse_m": float(np.sqrt(np.mean((scaled[mask] - gt[mask]) ** 2))),
                    "delta1": float(np.mean(ratio < 1.25)),
                    "condition_global_median_scale": scale,
                }
            )
    return rows


def main() -> None:
    for folder in ("figures", "tables", "data", "supplementary", "build", "validation"):
        (REPORT / folder).mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    trajectory = load_trajectory(DATA, "P000")
    base_rows = read_rows(BASE / "metrics.csv")
    adapted_rows = read_rows(ADAPTED / "metrics.csv")
    if [row["subset"] for row in base_rows] != list(config["selection"]["subsets"]):
        raise AssertionError("Baseline subset order differs from frozen protocol")
    if [row["subset"] for row in adapted_rows] != list(config["selection"]["subsets"]):
        raise AssertionError("Adapted subset order differs from frozen protocol")

    numerical_checks = []
    for run, canonical, label in ((BASE, base_rows, "pretrained"), (ADAPTED, adapted_rows, "adapted_step15")):
        for row in canonical:
            subset = row["subset"]
            indices = config["selection"]["subsets"][subset]
            saved = torch.load(run / subset / "predictions.pt", map_location="cpu", weights_only=True)
            pred = saved["depth"][0, ..., 0].float().numpy()
            conf = saved["depth_conf"][0].float().numpy()
            gt, valid = [], []
            for index in indices:
                depth, mask, _ = resize_square_ground_truth(read_depth(trajectory.depth_paths[index]), 518)
                gt.append(depth)
                valid.append(mask)
            gt = np.stack(gt)
            valid = np.stack(valid) & np.isfinite(pred) & (pred > 0)
            recomputed = scale_aligned_depth_metrics(pred, gt, valid)
            error = np.abs(pred[valid] * recomputed["scale"] - gt[valid]) / gt[valid]
            rho = float(spearmanr(conf[valid], error).statistic)
            checks = {
                "abs_rel": abs(recomputed["abs_rel"] - float(row["depth_abs_rel"])) < 1e-7,
                "rmse": abs(recomputed["rmse"] - float(row["depth_rmse"])) < 1e-6,
                "delta1": abs(recomputed["delta1"] - float(row["depth_delta1"])) < 1e-7,
                "confidence_rho": abs(rho - float(row["confidence_error_spearman"])) < 1e-7,
                "valid_mask": "finite prediction, prediction > 0, and GT 0.1-100 m",
                "scale_scope": "one global median scale jointly across all valid frames in this condition",
            }
            numerical_checks.append({"model": label, "subset": subset, **checks})
            if not all(value is True for key, value in checks.items() if key in {"abs_rel", "rmse", "delta1", "confidence_rho"}):
                raise AssertionError(f"Canonical metric mismatch: {label}/{subset}")

    bins = depth_bin_analysis(BASE, "pretrained", trajectory, config["selection"]["subsets"])
    bins += depth_bin_analysis(ADAPTED, "adapted_step15", trajectory, config["selection"]["subsets"])
    write_rows(REPORT / "tables/depth_range_analysis.csv", bins)
    write_rows(REPORT / "tables/pretrained_p000.csv", base_rows)
    write_rows(REPORT / "tables/adapted_p000_step15.csv", adapted_rows)

    train_config = yaml.safe_load(
        (ROOT / "configs/experiments/v3_tartanair_finetune.yaml").read_text(encoding="utf-8")
    )
    parameters = json.loads((TRAIN / "parameters.json").read_text(encoding="utf-8"))
    probe = json.loads((TRAIN / "memory_probe.json").read_text(encoding="utf-8"))
    runtime = json.loads((TRAIN / "runtime.json").read_text(encoding="utf-8"))
    best_heads = torch.load(TRAIN / "best_heads.pt", map_location="cpu", weights_only=True)
    history = read_rows(TRAIN / "history.csv")
    validation_rows = [row for row in history if row.get("validation_objective")]
    selected = min(validation_rows, key=lambda row: float(row["validation_objective"]))
    if int(selected["step"]) != 15 or int(best_heads["step"]) != 15:
        raise AssertionError("P006 checkpoint selection is not step 15")
    if set(train_config["split"]["test"]) & (
        set(train_config["split"]["train"]) | set(train_config["split"]["validation"])
    ):
        raise AssertionError("P000 test leakage")
    adapted_hashes = {
        json.loads((ADAPTED / subset / "metrics.json").read_text())["adapted_heads_sha256"]
        for subset in config["selection"]["subsets"]
    }
    if adapted_hashes != {sha256(TRAIN / "best_heads.pt")}:
        raise AssertionError("Adapted results do not all use selected step-15 heads")

    recommendation = {
        "status": "not_executed_no_independent_environment_local",
        "new_forward_passes": 0,
        "candidate": {
            "environment": "Office",
            "difficulty": "easy",
            "camera": "lcam_front",
            "required_archives": [
                {"name": "Office/Data_easy/image_lcam_front.zip", "size_gb": 2.764395089},
                {"name": "Office/Data_easy/depth_lcam_front.zip", "size_gb": 0.939556411},
            ],
            "compressed_total_gb": 3.7039515,
            "official_source": "https://huggingface.co/datasets/theairlabcmu/tartanair2",
            "reason": "Distinct official TartanAir V2 environment with RGB, depth, and bundled poses.",
        },
    }
    (REPORT / "data/independent_environment_recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    validation = {
        "source_commit_before_v4": source_commit,
        "canonical_runs": {
            "pretrained": str(BASE.relative_to(ROOT)).replace("\\", "/"),
            "adapted": str(ADAPTED.relative_to(ROOT)).replace("\\", "/"),
            "adaptation": str(TRAIN.relative_to(ROOT)).replace("\\", "/"),
        },
        "numerical_checks": numerical_checks,
        "metric_definitions_checked": {
            "abs_rel": "mean(abs(scaled_prediction - ground_truth) / ground_truth)",
            "delta1": "mean(max(pred/gt, gt/pred) < 1.25)",
            "rmse_unit": "metres after condition-level median scaling",
            "valid_depth_m": [0.1, 100.0],
            "similarity": "Umeyama least-squares Sim(3), independently per view-count condition",
            "rotation": "first-camera orientation gauge; mean geodesic SO(3) angle in degrees",
            "confidence": "Spearman rho between predicted confidence and per-pixel AbsRel",
        },
        "split_validation": {
            "training": train_config["split"]["train"],
            "validation": train_config["split"]["validation"],
            "test": train_config["split"]["test"],
            "p000_never_training_or_selection": True,
            "p006_alone_selected_step15": True,
            "adapted_head_hash": next(iter(adapted_hashes)),
        },
        "adaptation": {
            "parameters": parameters,
            "config": train_config,
            "memory_probe": probe,
            "runtime": runtime,
            "training_samples": 30,
            "batch": "one sequence of two adjacent frames",
            "gradient_accumulation": 1,
            "validation_frequency_steps": 5,
            "selected_step": 15,
            "additional_adaptation_for_v4": False,
        },
        "independent_environment": recommendation,
    }
    (REPORT / "validation/numerical_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checks": len(numerical_checks), "depth_bin_rows": len(bins), **recommendation}, indent=2))


if __name__ == "__main__":
    main()
