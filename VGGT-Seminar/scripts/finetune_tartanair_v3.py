from __future__ import annotations

import csv
import gc
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import yaml
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import extri_intri_to_pose_encoding

from vggt_seminar.tartanair import (
    camera_to_world_opencv,
    load_trajectory,
    read_depth,
    resize_square_ground_truth,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/v3_tartanair_finetune.yaml"
DATA = ROOT / "local_assets/datasets/tartanair_v2/ArchVizTinyHouseDay/Data_easy"


def ground_truth(trajectory, indices, device):
    depths, masks, intrinsics = [], [], []
    for index in indices:
        depth, mask, intrinsic = resize_square_ground_truth(
            read_depth(trajectory.depth_paths[index]), 518
        )
        depths.append(depth)
        masks.append(mask)
        intrinsics.append(intrinsic)
    depth = np.stack(depths)
    mask = np.stack(masks)
    scale = float(np.mean(depth[mask]))
    c2w = np.stack([camera_to_world_opencv(trajectory.poses_xyzw[i]) for i in indices])
    first_inverse = np.linalg.inv(c2w[0])
    relative_c2w = np.stack([first_inverse @ pose for pose in c2w])
    extrinsics = np.linalg.inv(relative_c2w)[:, :3]
    extrinsics[:, :3, 3] /= scale
    depth /= scale
    return {
        "depth": torch.from_numpy(depth).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "extrinsic": torch.from_numpy(extrinsics.astype(np.float32)).to(device),
        "intrinsic": torch.from_numpy(np.stack(intrinsics)).to(device),
    }


def losses(predictions, gt, camera_weight, depth_weight, confidence_regularizer):
    target_pose = extri_intri_to_pose_encoding(
        gt["extrinsic"][None], gt["intrinsic"][None], (518, 518)
    )
    camera = torch.nn.functional.smooth_l1_loss(predictions["pose_enc"], target_pose)
    pred_depth = predictions["depth"][0, ..., 0].clamp_min(1e-5)
    conf = predictions["depth_conf"][0].clamp_min(1.00001)
    mask = gt["mask"]
    error = (torch.log(pred_depth[mask]) - torch.log(gt["depth"][mask])).abs()
    confidence = (error * conf[mask] - confidence_regularizer * torch.log(conf[mask])).mean()
    regular = error.mean()
    depth = confidence + regular
    objective = camera_weight * camera + depth_weight * depth
    return {"objective": objective, "camera": camera, "depth": depth, "depth_log_l1": regular}


def evaluate(model, pairs, trajectories, device, config):
    values = []
    model.camera_head.eval()
    model.depth_head.eval()
    with torch.no_grad():
        for name, start in pairs:
            trajectory = trajectories[name]
            indices = [start, start + 1]
            images = load_and_preprocess_images(
                [trajectory.image_paths[i] for i in indices], mode="crop"
            ).to(device)
            gt = ground_truth(trajectory, indices, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(images)
                loss = losses(
                    prediction,
                    gt,
                    config["optimization"]["camera_loss_weight"],
                    config["optimization"]["depth_loss_weight"],
                    config["optimization"]["confidence_regularizer"],
                )
            values.append({key: float(value.detach()) for key, value in loss.items()})
    model.camera_head.train()
    model.depth_head.train()
    return {key: float(np.mean([item[key] for item in values])) for key in values[0]}


def main():
    config = yaml.safe_load(CONFIG.read_text())
    output = ROOT / config["execution"]["output_root"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.enable_flash_sdp(False)
    trajectories = {
        name: load_trajectory(DATA, name)
        for name in config["split"]["train"] + config["split"]["validation"]
    }
    model = VGGT(enable_camera=True, enable_depth=True, enable_point=False, enable_track=False)
    state = torch.load(
        ROOT / config["base_checkpoint"], map_location="cpu", weights_only=True, mmap=True
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(incompatible.missing_keys)
    del state
    for parameter in model.aggregator.parameters():
        parameter.requires_grad_(False)
    model.aggregator.eval()
    model.camera_head.train()
    model.depth_head.train()
    model.to(device)
    trainable = list(model.camera_head.parameters()) + list(model.depth_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["optimization"]["learning_rate"],
        weight_decay=config["optimization"]["weight_decay"],
    )
    parameter_record = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_parameters": sum(p.numel() for p in model.aggregator.parameters()),
    }
    (output / "parameters.json").write_text(json.dumps(parameter_record, indent=2) + "\n")
    pairs = [(str(name), int(start)) for name, start in config["validation"]["pairs"]]

    # No-step feasibility probe: forward + backward with the actual trainable heads.
    probe_trajectory = trajectories[config["split"]["train"][0]]
    probe_indices = [20, 21]
    probe_images = load_and_preprocess_images(
        [probe_trajectory.image_paths[i] for i in probe_indices], mode="crop"
    ).to(device)
    probe_gt = ground_truth(probe_trajectory, probe_indices, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        probe_prediction = model(probe_images)
        probe_loss = losses(
            probe_prediction,
            probe_gt,
            config["optimization"]["camera_loss_weight"],
            config["optimization"]["depth_loss_weight"],
            config["optimization"]["confidence_regularizer"],
        )
    probe_loss["objective"].backward()
    torch.cuda.synchronize()
    probe = {
        "status": "passed",
        "objective": float(probe_loss["objective"].detach()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "optimizer_step_performed": False,
    }
    optimizer.zero_grad(set_to_none=True)
    del probe_prediction, probe_images, probe_gt, probe_loss
    gc.collect()
    torch.cuda.empty_cache()
    (output / "memory_probe.json").write_text(json.dumps(probe, indent=2) + "\n")

    history = []
    best_validation = float("inf")
    started = time.perf_counter()
    for step in range(config["optimization"]["steps"]):
        name = config["split"]["train"][step % len(config["split"]["train"])]
        trajectory = trajectories[name]
        span = config["sampling"]["start_max"] - config["sampling"]["start_min"] + 1
        start = config["sampling"]["start_min"] + (step * 17) % span
        indices = [start, start + config["sampling"]["adjacent_stride"]]
        images = load_and_preprocess_images(
            [trajectory.image_paths[i] for i in indices], mode="crop"
        ).to(device)
        gt = ground_truth(trajectory, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(images)
            loss = losses(
                prediction,
                gt,
                config["optimization"]["camera_loss_weight"],
                config["optimization"]["depth_loss_weight"],
                config["optimization"]["confidence_regularizer"],
            )
        loss["objective"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, config["optimization"]["gradient_clip_norm"]
        )
        optimizer.step()
        row = {
            "step": step + 1,
            "trajectory": name,
            "indices": indices,
            **{key: float(value.detach()) for key, value in loss.items()},
            "gradient_norm": float(gradient_norm),
        }
        if (step + 1) % 5 == 0 or step == 0:
            validation = evaluate(model, pairs, trajectories, device, config)
            row.update({f"validation_{key}": value for key, value in validation.items()})
            if validation["objective"] < best_validation:
                best_validation = validation["objective"]
                torch.save(
                    {
                        "camera_head": model.camera_head.state_dict(),
                        "depth_head": model.depth_head.state_dict(),
                        "step": step + 1,
                        "validation": validation,
                    },
                    output / "best_heads.pt",
                )
        history.append(row)
        del prediction, images, gt, loss
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(row))
    torch.save(
        {
            "camera_head": model.camera_head.state_dict(),
            "depth_head": model.depth_head.state_dict(),
            "step": config["optimization"]["steps"],
        },
        output / "final_heads.pt",
    )
    fields = sorted({key for row in history for key in row})
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "status": "passed",
        "elapsed_seconds": time.perf_counter() - started,
        "best_validation_objective": best_validation,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "split": config["split"],
    }
    (output / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "SUCCESS").write_text("validated\n")
    print(json.dumps({"probe": probe, "runtime": metadata}, indent=2))


if __name__ == "__main__":
    main()
