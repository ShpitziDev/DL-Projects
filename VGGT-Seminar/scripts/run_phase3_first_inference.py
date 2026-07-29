from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import yaml
from PIL import Image

from vggt_seminar.live_demo import finite_tensor, save_heatmap, save_ply, tensor_record

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/phase3_first_run.yaml"
PIN_PATH = ROOT / "external/VGGT_PIN.json"
MANIFEST_PATH = ROOT / "local_assets/checkpoints/checkpoint_manifest.json"


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; Phase 3 refuses CPU fallback")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    output = ROOT / config["output"]["directory"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output}")
    for name in ("raw", "visualizations", "logs"):
        (output / name).mkdir(parents=True, exist_ok=False if name == "raw" else True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    checkpoint = ROOT / manifest["relative_path"]
    images_path = [ROOT / item for item in config["input"]["files"]]

    init_start = time.perf_counter()
    model = VGGT()
    architecture_init_seconds = time.perf_counter() - init_start
    load_start = time.perf_counter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    checkpoint_load_seconds = time.perf_counter() - load_start
    transfer_start = time.perf_counter()
    model = model.to(device).eval()
    torch.cuda.synchronize()
    gpu_transfer_seconds = time.perf_counter() - transfer_start

    images = load_and_preprocess_images(images_path, mode=config["input"]["preprocessing_mode"]).to(device)
    height, width = images.shape[-2:]
    query = torch.tensor([[[width / 2.0, height / 2.0]]], device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    baseline_memory = torch.cuda.memory_allocated(0)

    inference_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(images, query_points=query)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    peak_memory = torch.cuda.max_memory_allocated(0)

    pose = predictions["pose_enc"].float()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose, images.shape[-2:])
    unprojected = torch.from_numpy(unproject_depth_map_to_point_map(
        predictions["depth"].float().squeeze(0).cpu().numpy(),
        extrinsic.squeeze(0).cpu().numpy(), intrinsic.squeeze(0).cpu().numpy(),
    ))
    derived = {"extrinsic": extrinsic, "intrinsic": intrinsic, "unprojected_points": unprojected}

    schema: dict[str, object] = {"schema_version": 1, "outputs": {}, "derived": {}}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            if not finite_tensor(value):
                raise RuntimeError(f"Non-finite output detected: {key}")
            torch.save(value.detach().cpu(), output / "raw" / f"{key}.pt")
            schema["outputs"][key] = tensor_record(value)
        elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
            for index, item in enumerate(value):
                if not finite_tensor(item):
                    raise RuntimeError(f"Non-finite output detected: {key}[{index}]")
                torch.save(item.detach().cpu(), output / "raw" / f"{key}_{index}.pt")
            schema["outputs"][key] = {"type": "tensor_list", "length": len(value),
                                       "items": [tensor_record(item) for item in value]}
        else:
            raise TypeError(f"Unexpected prediction type for {key}: {type(value)}")
    for key, value in derived.items():
        if not finite_tensor(value):
            raise RuntimeError(f"Non-finite derived output detected: {key}")
        torch.save(value.detach().cpu(), output / "raw" / f"{key}.pt")
        schema["derived"][key] = tensor_record(value)
    (output / "output_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    input_image = predictions["images"][0, 0].permute(1, 2, 0).float().cpu().numpy()
    Image.fromarray((np.clip(input_image, 0, 1) * 255).astype(np.uint8)).save(output / "visualizations/input.png")
    save_heatmap(predictions["depth"][0, 0], output / "visualizations/depth.png", invert=True)
    save_heatmap(predictions["depth_conf"][0, 0], output / "visualizations/depth_confidence.png")
    save_heatmap(predictions["world_points_conf"][0, 0], output / "visualizations/point_confidence.png")
    points = unprojected[0].numpy()
    save_ply(points, input_image, output / "visualizations/point_cloud_depth_unprojected.ply")
    save_ply(predictions["world_points"][0, 0].float().cpu().numpy(), input_image,
              output / "visualizations/point_cloud_direct.ply")

    runtime = {
        "schema_version": 1, "status": "passed", "offline": True,
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "device": str(device), "dtype": "torch.bfloat16", "cuda_verified": True,
        "architecture_init_seconds": architecture_init_seconds,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "gpu_transfer_seconds": gpu_transfer_seconds,
        "inference_seconds": inference_seconds,
        "baseline_gpu_memory_bytes": int(baseline_memory),
        "peak_gpu_memory_bytes": int(peak_memory),
        "peak_inference_increment_bytes": int(peak_memory - baseline_memory),
        "image_count": int(images.shape[0]), "batch_count": 1,
        "input_resolution": [int(height), int(width)],
        "checkpoint_sha256": manifest["sha256"], "official_code_commit": pin["commit"],
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    input_manifest = {"files": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                  "size_bytes": path.stat().st_size} for path in images_path]}
    (output / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n", encoding="utf-8")
    (output / "logs/run.log").write_text(
        "Phase 3 first inference passed. Offline local checkpoint loading was enforced.\n", encoding="utf-8"
    )
    print(json.dumps(runtime, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
