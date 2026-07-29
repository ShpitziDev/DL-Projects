from __future__ import annotations

import csv
import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from itertools import combinations
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
from vggt_seminar.eth3d import load_scene
from vggt_seminar.eth3d_overlap import PROTOCOL_VERSION, load_frozen_selection
from vggt_seminar.live_demo import point_cloud_preview, save_heatmap, save_ply, verify_local_assets
try:
    from scripts.run_phase6_eth3d_smoke import add_visualization_title, camera_centers, git_text, save_camera_plot, save_contact_sheet, tensor_summary
except ModuleNotFoundError:  # Direct `python scripts/...` execution places scripts/ on sys.path.
    from run_phase6_eth3d_smoke import add_visualization_title, camera_centers, git_text, save_camera_plot, save_contact_sheet, tensor_summary

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / os.environ.get("VGGT_PILOT_CONFIG", "configs/experiments/phase7_delivery_area_view_count.yaml")
ETH3D_ROOT = ROOT / "local_assets/datasets/eth3d"
REQUIRED_OUTPUTS = {"pose_enc", "depth", "depth_conf", "world_points", "world_points_conf", "images", "track", "vis", "conf"}
SUBSETS = ("S2", "S4", "S6", "S8", "S10")
METRIC_COLUMNS = (
    "scene", "subset", "view_count", "protocol_version", "indices", "inference_seconds",
    "preprocessing_seconds", "postprocessing_seconds", "total_seconds", "peak_allocated_gib",
    "peak_reserved_gib", "depth_confidence_mean", "depth_confidence_median",
    "point_confidence_mean", "point_confidence_median", "retained_points",
    "retained_percentage", "camera_path_length", "max_camera_separation",
    "checkpoint_sha256", "git_commit", "reused_existing_result",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_reusable_s2(path: Path, assets: dict[str, Any], frozen: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    required = ["runtime.json", "preflight.json", "tensor_summaries.json", "camera_parameters.json",
                "frozen_protocol.json", "selected_frames.json", "arrays/depth.pt", "arrays/world_points.pt",
                "visualizations/contact_sheet.jpg", "visualizations/point_cloud_direct.ply",
                "visualizations/point_cloud_depth_unprojected.ply", "visualizations/point_cloud_confidence_filtered.ply"]
    errors.extend(f"missing {item}" for item in required if not (path / item).is_file())
    if errors:
        return False, errors
    runtime, preflight, protocol, summaries = (read_json(path / name) for name in
        ("runtime.json", "preflight.json", "frozen_protocol.json", "tensor_summaries.json"))
    checks = [
        (runtime.get("status") == "passed", "S2 status is not passed"),
        (runtime.get("forward_pass_count") == 1, "S2 did not record exactly one forward pass"),
        (runtime.get("precision") == "torch.bfloat16 autocast", "S2 precision mismatch"),
        (runtime.get("flash_sdp_enabled") is False, "S2 Flash SDPA mismatch"),
        (preflight.get("selected_indices") == frozen["indices"], "S2 indices mismatch"),
        (preflight.get("selected_filenames") == frozen["filenames"], "S2 filenames mismatch"),
        (preflight.get("checkpoint_sha256") == assets["checkpoint_sha256"], "S2 checkpoint mismatch"),
        (protocol.get("version") == PROTOCOL_VERSION, "S2 protocol mismatch"),
        (all(v.get("finite_percentage", 100.0) >= 99.99 for v in summaries.values() if isinstance(v, dict)), "S2 invalid tensor values"),
    ]
    errors.extend(message for passed, message in checks if not passed)
    return not errors, errors


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from {ROOT}")
    scene_name = config["scene"]
    if scene_name != config["constraints"]["approved_scene"] or config["protocol"]["subsets"] != list(SUBSETS):
        raise ValueError("Scene or subset matrix differs from the explicitly approved configuration")
    if config["constraints"]["allow_fallback"] is not False or config["order"] != "original":
        raise ValueError("Fallback or non-original ordering is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    assets = verify_local_assets(ROOT)
    scene = load_scene(ETH3D_ROOT, scene_name)
    protocol_path = ROOT / config["protocol"]["config"]
    frozen = {subset: load_frozen_selection(protocol_path, scene_name, int(subset[1:])) for subset in SUBSETS}
    if any(item["protocol_version"] != PROTOCOL_VERSION for item in frozen.values()):
        raise RuntimeError("Protocol version mismatch")
    for left, right in zip(SUBSETS, SUBSETS[1:]):
        if not set(frozen[left]["indices"]) < set(frozen[right]["indices"]):
            raise RuntimeError(f"{left} is not a strict subset of {right}")
    for subset, item in frozen.items():
        paths = [scene.image_paths[index] for index in item["indices"]]
        if [path.name for path in paths] != item["filenames"] or any(not path.is_file() for path in paths):
            raise RuntimeError(f"Frozen files invalid for {subset}")
    run_root, aggregate_root = ROOT / config["run_root"], ROOT / config["aggregate_root"]
    if aggregate_root.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate {aggregate_root}")
    free, total = torch.cuda.mem_get_info(0)
    reuse_path = ROOT / config["s2_reuse"] if config.get("s2_reuse") else None
    reuse_ok, reuse_errors = validate_reusable_s2(reuse_path, assets, frozen["S2"]) if reuse_path else (False, ["reuse not configured"])
    return {"assets": assets, "scene": scene, "frozen": frozen, "run_root": run_root,
            "aggregate_root": aggregate_root, "reuse_path": reuse_path, "reuse_ok": reuse_ok,
            "reuse_errors": reuse_errors, "git_commit": git_text("rev-parse", "HEAD"),
            "git_status": git_text("status", "--short").splitlines(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "gpu_free_bytes": int(free), "gpu_total_bytes": int(total)}


def geometry_metrics(points: np.ndarray, centers: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(points).all(axis=-1)
    finite_points = points[valid]
    extents = (finite_points.max(axis=0) - finite_points.min(axis=0)).tolist()
    adjacent = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    maximum = max((float(np.linalg.norm(centers[a] - centers[b])) for a, b in combinations(range(len(centers)), 2)), default=0.0)
    return {"raw_point_count": int(valid.size), "valid_point_count": int(valid.sum()), "bounding_box_extents": extents,
            "camera_path_length": float(adjacent.sum()), "max_camera_separation": maximum,
            "adjacent_camera_distances": adjacent.tolist(), "coordinate_scale": "VGGT arbitrary units; unaligned"}


def completed_metrics(final: Path, expected_indices: list[int], checkpoint_sha256: str) -> dict[str, Any] | None:
    if (final / "SUCCESS").is_file():
        metrics = read_json(final / "metrics.json")
        if metrics["indices"] != expected_indices or metrics["checkpoint_sha256"] != checkpoint_sha256:
            raise RuntimeError(f"Completed result is incompatible: {final}")
        return {**metrics, "resumed_existing": True}
    if final.exists():
        raise RuntimeError(f"Partial result exists; preserve and inspect before resume: {final}")
    return None


def run_subset(subset: str, context: dict[str, Any], config: dict[str, Any], model: VGGT, device: torch.device) -> dict[str, Any]:
    item = context["frozen"][subset]
    paths = [context["scene"].image_paths[index] for index in item["indices"]]
    final = context["run_root"] / f"{subset}_overlap_aware_nested_original"
    completed = completed_metrics(final, item["indices"], context["assets"]["checkpoint_sha256"])
    if completed is not None:
        return completed
    temporary = final.with_name(final.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Temporary failed run exists: {temporary}")
    for directory in (temporary, temporary / "arrays", temporary / "visualizations", temporary / "logs"):
        directory.mkdir(parents=True, exist_ok=False)
    scene_name = config["scene"]
    title = f"{scene_name} | {subset} | overlap_aware_nested | indices {item['indices']} | original"
    started = time.perf_counter()
    try:
        preprocess_started = time.perf_counter()
        images_cpu = load_and_preprocess_images(paths, mode=config["preprocessing_mode"])
        images = images_cpu.to(device)
        height, width = images.shape[-2:]
        query = torch.tensor([[[width / 2.0, height / 2.0]]], device=device)
        torch.cuda.synchronize()
        preprocessing_seconds = time.perf_counter() - preprocess_started
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
        inference_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(images, query_points=query)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0)
        post_started = time.perf_counter()
        missing = REQUIRED_OUTPUTS - predictions.keys()
        if missing:
            raise RuntimeError(f"Missing outputs: {sorted(missing)}")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"].float(), images.shape[-2:])
        unprojected_np = unproject_depth_map_to_point_map(predictions["depth"].float().squeeze(0).cpu().numpy(),
                                                          extrinsic.squeeze(0).cpu().numpy(), intrinsic.squeeze(0).cpu().numpy())
        derived = {"extrinsic": extrinsic, "intrinsic": intrinsic, "unprojected_points": torch.from_numpy(unprojected_np)}
        summaries: dict[str, Any] = {}
        for name, value in {**predictions, **derived}.items():
            if isinstance(value, torch.Tensor):
                summaries[name] = tensor_summary(value)
                if summaries[name]["finite_percentage"] < 99.99:
                    raise RuntimeError(f"Invalid values in {name}")
                torch.save(value.detach().cpu(), temporary / "arrays" / f"{name}.pt")
            elif isinstance(value, list):
                summaries[name] = {"type": "tensor_list", "length": len(value), "items": [tensor_summary(v) for v in value]}
                for index, value_item in enumerate(value):
                    torch.save(value_item.detach().cpu(), temporary / "arrays" / f"{name}_{index}.pt")
        centers = camera_centers(extrinsic)
        colors = predictions["images"][0].permute(0, 2, 3, 1).float().cpu().numpy()
        direct = predictions["world_points"][0].float().cpu().numpy()
        confidence = predictions["world_points_conf"][0].float().cpu().numpy()
        threshold = float(np.median(confidence)); filtered = direct.copy(); filtered[confidence <= threshold] = np.nan
        frame_record = save_contact_sheet(paths, item["indices"], temporary / "visualizations/contact_sheet.jpg", title)
        frame_record.update({"indices": item["indices"], "filenames": item["filenames"], "protocol_version": PROTOCOL_VERSION})
        for view in range(len(paths)):
            save_heatmap(predictions["depth"][0, view], temporary / "visualizations" / f"depth_view{view}.png", invert=True)
            save_heatmap(predictions["depth_conf"][0, view], temporary / "visualizations" / f"depth_confidence_view{view}.png")
            save_heatmap(predictions["world_points_conf"][0, view], temporary / "visualizations" / f"point_confidence_view{view}.png")
            for prefix in ("depth", "depth_confidence", "point_confidence"):
                add_visualization_title(temporary / "visualizations" / f"{prefix}_view{view}.png", title)
        save_camera_plot(centers, temporary / "visualizations/camera_centers.png", title)
        previews = {"point_cloud_direct_preview.png": direct, "point_cloud_depth_unprojected_preview.png": unprojected_np,
                    "point_cloud_confidence_filtered_preview.png": filtered}
        for name, points in previews.items():
            point_cloud_preview(points, colors).save(temporary / "visualizations" / name)
            add_visualization_title(temporary / "visualizations" / name, title)
        point_counts = {
            "direct": save_ply(direct, colors, temporary / "visualizations/point_cloud_direct.ply"),
            "depth_unprojected": save_ply(unprojected_np, colors, temporary / "visualizations/point_cloud_depth_unprojected.ply"),
            "confidence_filtered": save_ply(filtered, colors, temporary / "visualizations/point_cloud_confidence_filtered.ply")}
        geometry = geometry_metrics(direct, centers)
        postprocessing_seconds = time.perf_counter() - post_started
        metrics = {"scene": scene_name, "subset": subset, "view_count": len(paths), "protocol_version": PROTOCOL_VERSION,
                   "indices": item["indices"], "filenames": item["filenames"], "checkpoint_sha256": context["assets"]["checkpoint_sha256"],
                   "git_commit": context["git_commit"], "precision": "torch.bfloat16 autocast", "device": str(device),
                   "preprocessing_seconds": preprocessing_seconds, "inference_seconds": inference_seconds,
                   "postprocessing_seconds": postprocessing_seconds, "total_seconds": time.perf_counter() - started,
                   "peak_allocated_vram_bytes": int(peak_allocated), "peak_reserved_vram_bytes": int(peak_reserved),
                   "tensor_summaries": summaries, "depth_confidence": summaries["depth_conf"],
                   "point_confidence": summaries["world_points_conf"], "confidence_threshold": threshold,
                   "points_before_filtering": int(confidence.size), "retained_points": point_counts["confidence_filtered"],
                   "retained_percentage": 100.0 * point_counts["confidence_filtered"] / confidence.size,
                   "point_counts": point_counts, "geometry": geometry, "reused_existing_result": False,
                   "forward_pass_count": 1, "resumed_existing": False,
                   "artifact_directory": str(final.relative_to(ROOT)).replace("\\", "/")}
        (temporary / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        (temporary / "tensor_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        (temporary / "selected_frames.json").write_text(json.dumps(frame_record, indent=2) + "\n", encoding="utf-8")
        (temporary / "camera_parameters.json").write_text(json.dumps({"camera_centers": centers.tolist(), "extrinsic": extrinsic.cpu().tolist(),
            "intrinsic": intrinsic.cpu().tolist(), "scale": "arbitrary; unaligned"}, indent=2) + "\n", encoding="utf-8")
        (temporary / "resolved_config.yaml").write_text(yaml.safe_dump({**config, "resolved_subset": subset, "resolved_indices": item["indices"]}, sort_keys=False), encoding="utf-8")
        (temporary / "protocol.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
        (temporary / "logs/execution.log").write_text(f"{subset} passed; exactly one forward pass.\n", encoding="utf-8")
        (temporary / "logs/error.log").write_text("", encoding="utf-8")
        (temporary / "SUCCESS").write_text("validated\n", encoding="utf-8")
        temporary.rename(final)
        del predictions, derived, images, images_cpu, direct, unprojected_np, filtered
        gc.collect(); torch.cuda.empty_cache()
        return metrics
    except Exception:
        (temporary / "logs/error.log").write_text(traceback.format_exc(), encoding="utf-8")
        (temporary / "FAILED").write_text(subset + "\n", encoding="utf-8")
        raise


def metrics_from_reused_s2(path: Path, context: dict[str, Any]) -> dict[str, Any]:
    runtime, summaries, cameras = (read_json(path / name) for name in ("runtime.json", "tensor_summaries.json", "camera_parameters.json"))
    centers = np.asarray(cameras["camera_centers"]); geometry = geometry_metrics(torch.load(path / "arrays/world_points.pt", map_location="cpu", weights_only=True)[0].numpy(), centers)
    return {"scene": context["scene"].name, "subset": "S2", "view_count": 2, "protocol_version": PROTOCOL_VERSION,
            "indices": context["frozen"]["S2"]["indices"], "filenames": context["frozen"]["S2"]["filenames"],
            "checkpoint_sha256": context["assets"]["checkpoint_sha256"], "git_commit": read_json(path / "preflight.json")["git_head_before_run"],
            "precision": runtime["precision"], "device": "cuda:0", "preprocessing_seconds": runtime["preprocessing_seconds"],
            "inference_seconds": runtime["inference_seconds"], "postprocessing_seconds": runtime["postprocessing_seconds"],
            "total_seconds": runtime["preprocessing_seconds"] + runtime["inference_seconds"] + runtime["postprocessing_seconds"],
            "peak_allocated_vram_bytes": runtime["peak_allocated_vram_bytes"],
            "peak_reserved_vram_bytes": runtime["peak_reserved_vram_bytes"], "tensor_summaries": summaries,
            "depth_confidence": summaries["depth_conf"], "point_confidence": summaries["world_points_conf"],
            "confidence_threshold": runtime["confidence_filter"]["threshold"], "points_before_filtering": runtime["confidence_filter"]["points_before"],
            "retained_points": runtime["confidence_filter"]["points_retained"], "retained_percentage": runtime["confidence_filter"]["retained_percentage"],
            "point_counts": runtime["point_counts"], "geometry": geometry, "reused_existing_result": True, "forward_pass_count": 0,
            "resumed_existing": False, "artifact_directory": str(path.relative_to(ROOT)).replace("\\", "/")}


def row(metric: dict[str, Any]) -> dict[str, Any]:
    return {"scene": metric["scene"], "subset": metric["subset"], "view_count": metric["view_count"],
            "protocol_version": metric["protocol_version"], "indices": json.dumps(metric["indices"]),
            "inference_seconds": metric["inference_seconds"], "preprocessing_seconds": metric["preprocessing_seconds"],
            "postprocessing_seconds": metric["postprocessing_seconds"], "total_seconds": metric["total_seconds"],
            "peak_allocated_gib": metric["peak_allocated_vram_bytes"] / 2**30, "peak_reserved_gib": metric["peak_reserved_vram_bytes"] / 2**30,
            "depth_confidence_mean": metric["depth_confidence"]["mean"], "depth_confidence_median": metric["depth_confidence"]["median"],
            "point_confidence_mean": metric["point_confidence"]["mean"], "point_confidence_median": metric["point_confidence"]["median"],
            "retained_points": metric["retained_points"], "retained_percentage": metric["retained_percentage"],
            "camera_path_length": metric["geometry"]["camera_path_length"], "max_camera_separation": metric["geometry"]["max_camera_separation"],
            "checkpoint_sha256": metric["checkpoint_sha256"], "git_commit": metric["git_commit"],
            "reused_existing_result": metric["reused_existing_result"]}


def plot_series(rows: list[dict[str, Any]], keys: list[str], title: str, ylabel: str, output: Path) -> None:
    width, height, margin = 900, 560, 80
    canvas = np.full((height, width, 3), 255, np.uint8); values = [float(r[k]) for r in rows for k in keys]
    lo, hi = min(0.0, min(values)), max(values); span = max(hi - lo, 1e-9)
    cv2.line(canvas, (margin, height-margin), (width-margin, height-margin), (0,0,0), 2); cv2.line(canvas, (margin, margin), (margin, height-margin), (0,0,0), 2)
    colors = [(210,80,30), (30,120,210)]
    for key_index, key in enumerate(keys):
        points=[]
        for index, item in enumerate(rows):
            x=margin+index*(width-2*margin)/(len(rows)-1); y=height-margin-(float(item[key])-lo)/span*(height-2*margin); points.append((int(x),int(y)))
            cv2.circle(canvas, points[-1], 6, colors[key_index], -1); cv2.putText(canvas, str(item["view_count"]), (int(x)-8,height-margin+25), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        cv2.polylines(canvas,[np.array(points)],False,colors[key_index],2); cv2.putText(canvas,key,(margin+key_index*330,height-25),cv2.FONT_HERSHEY_SIMPLEX,.48,colors[key_index],1)
    cv2.putText(canvas,title,(25,35),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,0,0),2); cv2.putText(canvas,"Input views",(width//2-50,height-45),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
    cv2.putText(canvas,ylabel,(10,65),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1); cv2.putText(canvas,"Source: summary.csv | one run/condition; no error bars",(25,height-5),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
    Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)).save(output)


def gallery(paths: list[Path], labels: list[str], output: Path, title: str) -> None:
    thumbs=[]
    for path in paths:
        with Image.open(path) as source: image=source.convert("RGB"); image.thumbnail((360,300)); thumbs.append(image.copy())
    canvas=Image.new("RGB",(360*len(thumbs),350),"white"); draw=ImageDraw.Draw(canvas)
    for index,image in enumerate(thumbs): canvas.paste(image,(index*360,45)); draw.text((index*360+8,20),labels[index],fill="black")
    draw.text((8,332),title,fill="black"); canvas.save(output)


def contact_gallery(paths: list[Path], labels: list[str], output: Path, title: str) -> None:
    panels=[]
    for path in paths:
        with Image.open(path) as source: image=source.convert("RGB"); image.thumbnail((1600,420)); panels.append(image.copy())
    canvas=Image.new("RGB",(1620,sum(image.height+35 for image in panels)+35),"white"); draw=ImageDraw.Draw(canvas); y=8
    for label,image in zip(labels,panels):
        draw.text((8,y),label,fill="black"); y+=24; canvas.paste(image,(8,y)); y+=image.height+11
    draw.text((8,canvas.height-22),title,fill="black"); canvas.save(output)


def cross_scene_comparison(rows: list[dict[str, Any]], config: dict[str, Any], root: Path) -> None:
    comparison_path = config.get("comparison_aggregate")
    if not comparison_path:
        return
    other_root = ROOT / comparison_path
    if not (other_root / "summary.json").is_file():
        raise FileNotFoundError(f"Saved comparison aggregate missing: {other_root}")
    other = json.loads((other_root / "summary.json").read_text(encoding="utf-8"))
    if [int(item["view_count"]) for item in other] != [int(item["view_count"]) for item in rows]:
        raise RuntimeError("Cross-scene view-count axes differ")
    fields = ["inference_seconds", "total_seconds", "peak_allocated_gib", "peak_reserved_gib",
              "depth_confidence_mean", "depth_confidence_median", "point_confidence_mean", "point_confidence_median",
              "retained_points", "retained_percentage", "camera_path_length", "max_camera_separation"]
    merged=[]
    for delivery, courtyard in zip(other, rows):
        record={"view_count":int(courtyard["view_count"]), "delivery_area_subset":delivery["subset"], "courtyard_subset":courtyard["subset"]}
        for field in fields:
            record[f"delivery_area_{field}"]=delivery[field]; record[f"courtyard_{field}"]=courtyard[field]
        merged.append(record)
    comparison=root/"comparisons"; (comparison/"delivery_area_vs_courtyard.json").write_text(json.dumps(merged,indent=2)+"\n",encoding="utf-8")
    columns=list(merged[0])
    with (comparison/"delivery_area_vs_courtyard.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader(); writer.writerows(merged)
    plot_specs=[("inference_seconds","Inference time"),("total_seconds","Total processing time"),("peak_allocated_gib","Allocated VRAM"),
                ("depth_confidence_mean","Mean depth confidence"),("point_confidence_mean","Mean point confidence")]
    for index,(field,title) in enumerate(plot_specs,1):
        plot_series(merged,[f"delivery_area_{field}",f"courtyard_{field}"],f"Two-scene descriptive comparison | {title}",field,comparison/f"cross_{index:02d}_{field}.png")


def aggregate(metrics: list[dict[str, Any]], context: dict[str, Any], config: dict[str, Any], model_timing: dict[str, float], executed: int) -> None:
    root=context["aggregate_root"]
    for directory in (root,root/"plots",root/"comparisons",root/"contact_sheets",root/"point_cloud_previews",root/"run_logs",root/"tables"):
        directory.mkdir(parents=True,exist_ok=False)
    rows=[row(item) for item in metrics]
    with (root/"summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=METRIC_COLUMNS); writer.writeheader(); writer.writerows(rows)
    (root/"summary.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    scene_name=config["scene"]
    manifest={"protocol_version":PROTOCOL_VERSION,"scene":scene_name,"subsets":SUBSETS,"s2_reused":metrics[0]["reused_existing_result"],
              "new_forward_pass_count":sum(m["forward_pass_count"] for m in metrics),
              "forward_passes_this_invocation":executed,"represented_configurations":len(metrics),"model_timing":model_timing,
              "source_config":str(CONFIG_PATH.relative_to(ROOT)).replace("\\","/"),"runs":[{"subset":m["subset"],"path":m["artifact_directory"]} for m in metrics]}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    specifications=[(["inference_seconds"],"Inference time vs views","Seconds","01_inference_time.png"),(["total_seconds"],"Total processing time vs views","Seconds","02_total_time.png"),
        (["peak_allocated_gib"],"Peak allocated VRAM vs views","GiB","03_peak_allocated.png"),(["peak_reserved_gib"],"Peak reserved VRAM vs views","GiB","04_peak_reserved.png"),
        (["depth_confidence_mean"],"Mean depth confidence vs views","Model confidence","05_depth_conf_mean.png"),(["point_confidence_mean"],"Mean point confidence vs views","Model confidence","06_point_conf_mean.png"),
        (["depth_confidence_median","point_confidence_median"],"Median confidence vs views","Model confidence","07_median_confidence.png"),(["retained_points"],"Filtered point count vs views","Points","08_retained_points.png"),
        (["retained_percentage"],"Retained point percentage vs views","Percent","09_retained_percentage.png"),(["camera_path_length"],"Predicted camera path vs views","VGGT arbitrary units","10_camera_path.png"),
        (["max_camera_separation"],"Maximum camera separation vs views","VGGT arbitrary units","11_max_camera_separation.png")]
    for keys,title,ylabel,name in specifications: plot_series(rows,keys,f"{scene_name} | {title} | {PROTOCOL_VERSION}",ylabel,root/"plots"/name)
    run_paths=[ROOT/m["artifact_directory"] for m in metrics]; labels=[m["subset"] for m in metrics]
    visuals=[("contact_sheet.jpg",root/"contact_sheets/all_subsets.jpg"), ("depth_view0.png",root/"comparisons/depth_view0.jpg"),
             ("depth_confidence_view0.png",root/"comparisons/depth_confidence_view0.jpg"),("point_confidence_view0.png",root/"comparisons/point_confidence_view0.jpg"),
             ("camera_centers.png",root/"comparisons/camera_trajectories.jpg"),("point_cloud_direct_preview.png",root/"point_cloud_previews/direct.jpg"),
             ("point_cloud_confidence_filtered_preview.png",root/"point_cloud_previews/confidence_filtered.jpg")]
    for filename,target in visuals:
        maker=contact_gallery if filename=="contact_sheet.jpg" else gallery
        maker([path/"visualizations"/filename for path in run_paths],labels,target,f"{scene_name} | {PROTOCOL_VERSION} | deterministic preview; arbitrary unaligned scale")
    changes=[]
    for before,after in zip(rows,rows[1:]):
        changes.append({"from":before["subset"],"to":after["subset"],**{key:{"absolute":float(after[key])-float(before[key]),"percent":100*(float(after[key])-float(before[key]))/float(before[key]) if float(before[key]) else None} for key in
            ("inference_seconds","peak_allocated_gib","depth_confidence_mean","point_confidence_mean","retained_points","camera_path_length")}})
    depth_directions=["up" if b["depth_confidence_mean"]>a["depth_confidence_mean"] else "down" for a,b in zip(rows,rows[1:])]
    point_directions=["up" if b["point_confidence_mean"]>a["point_confidence_mean"] else "down" for a,b in zip(rows,rows[1:])]
    confidence_dips=[f"{after['subset']} mean confidence is lower than {before['subset']}" for before,after in zip(rows,rows[1:])
                     if after["depth_confidence_mean"]<before["depth_confidence_mean"] or after["point_confidence_mean"]<before["point_confidence_mean"]]
    camera_dips=[f"{after['subset']} camera path length is lower than {before['subset']}" for before,after in zip(rows,rows[1:])
                 if after["camera_path_length"]<before["camera_path_length"]]
    trends={"consecutive_changes":changes,"largest_runtime_increase":max(changes,key=lambda x:x["inference_seconds"]["absolute"]),
            "largest_vram_increase":max(changes,key=lambda x:x["peak_allocated_gib"]["absolute"]),
            "retained_points_monotonic":all(rows[i]["retained_points"]<=rows[i+1]["retained_points"] for i in range(len(rows)-1)),
            "depth_confidence_directions":depth_directions,"point_confidence_directions":point_directions,
            "confidence_appears_stabilized":abs(changes[-1]["depth_confidence_mean"]["percent"])<5 and abs(changes[-1]["point_confidence_mean"]["percent"])<5,
            "descriptive_anomalies":confidence_dips+camera_dips,
            "note":"Descriptive single-run analysis only; no causal or statistical claim."}
    (root/"comparisons/trend_analysis.json").write_text(json.dumps(trends,indent=2)+"\n",encoding="utf-8")
    cross_scene_comparison(rows,config,root)


def main() -> int:
    config=yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")); context=preflight(config)
    metrics=[]; executed=0
    if context["reuse_ok"]: metrics.append(metrics_from_reused_s2(context["reuse_path"],context))
    subsets_to_run=SUBSETS[1:]
    if not context["reuse_ok"]: subsets_to_run=SUBSETS
    torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"]); torch.backends.cuda.enable_flash_sdp(False); device=torch.device("cuda:0")
    init=time.perf_counter(); model=VGGT(); architecture=time.perf_counter()-init
    load=time.perf_counter(); state=torch.load(context["assets"]["checkpoint"],map_location="cpu",weights_only=True,mmap=True); model.load_state_dict(state,strict=True); del state; checkpoint=time.perf_counter()-load
    transfer=time.perf_counter(); model=model.to(device).eval(); torch.cuda.synchronize(); gpu_transfer=time.perf_counter()-transfer
    try:
        for subset in subsets_to_run:
            metric=run_subset(subset,context,config,model,device); metrics.append(metric); executed += 0 if metric.get("resumed_existing") else 1
        metrics.sort(key=lambda item:item["view_count"])
        expected=4 if context["reuse_ok"] else 5
        represented_forward_passes=sum(item["forward_pass_count"] for item in metrics)
        if represented_forward_passes != expected or len(metrics)!=5:
            raise RuntimeError(f"Forward-count contract failed: represented={represented_forward_passes}, configs={len(metrics)}")
        aggregate(metrics,context,config,{"architecture_init_seconds":architecture,"checkpoint_load_seconds":checkpoint,"gpu_transfer_seconds":gpu_transfer},executed)
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"status":"passed","s2_reused":context["reuse_ok"],"new_forward_pass_count":executed,"represented":len(metrics)},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
