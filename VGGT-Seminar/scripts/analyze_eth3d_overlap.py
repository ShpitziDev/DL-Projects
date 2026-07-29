from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vggt_seminar.eth3d import load_scene
from vggt_seminar.eth3d_overlap import (
    PROTOCOL_VERSION, build_nested_subsets, analyze_scene_pairs, pair_lookup,
    rank_windows, save_pair_cache, window_summary,
)


ROOT = Path(__file__).resolve().parents[1]
ETH3D_ROOT = ROOT / "local_assets/datasets/eth3d"
OUTPUT_ROOT = ROOT / "outputs/analysis/eth3d_overlap"
SCENES = ("delivery_area", "courtyard")
MAX_DIMENSION = 960
NFEATURES = 2500
MAX_PAIR_GAP = 12
RATIO = 0.75
RANSAC_THRESHOLD = 1.5


def line_plot(values: list[float], path: Path, title: str, y_label: str) -> None:
    width, height, margin = 1000, 420, 65
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    maximum = max(max(values), 1e-8)
    points = []
    for index, value in enumerate(values):
        x = margin + index / max(1, len(values) - 1) * (width - 2 * margin)
        y = height - margin - value / maximum * (height - 2 * margin)
        points.append((int(x), int(y)))
    cv2.polylines(canvas, [np.array(points)], False, (30, 90, 210), 2)
    cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (60, 60, 60), 1)
    cv2.line(canvas, (margin, margin), (margin, height - margin), (60, 60, 60), 1)
    cv2.putText(canvas, title, (margin, 32), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2)
    cv2.putText(canvas, y_label, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.putText(canvas, "ordered neighbor index", (width // 2 - 100, height - 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.putText(canvas, f"max={maximum:.3g}", (width - 180, 32), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


def trajectory_plot(centers: np.ndarray, path: Path, title: str) -> None:
    width = height = 760
    margin = 75
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    points = centers[:, [0, 2]]
    lo, hi = points.min(axis=0), points.max(axis=0)
    pixels = margin + (points - lo) / np.maximum(hi - lo, 1e-9) * (width - 2 * margin)
    pixels[:, 1] = height - pixels[:, 1]
    cv2.polylines(canvas, [pixels.astype(np.int32)], False, (90, 90, 90), 2)
    for index, point in enumerate(pixels.astype(int)):
        color = (10, int(220 * index / max(1, len(points) - 1)), 230)
        cv2.circle(canvas, tuple(point), 5, color, -1)
        if index % 5 == 0 or index == len(points) - 1:
            cv2.putText(canvas, str(index), tuple(point + np.array([7, -7])), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    cv2.putText(canvas, title, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
    cv2.putText(canvas, "camera-center X-Z projection (ETH3D coordinates)", (30, height - 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


def overlap_heatmap(frame_count: int, pairs, path: Path, title: str) -> None:
    values = np.full((frame_count, frame_count), np.nan, dtype=np.float32)
    np.fill_diagonal(values, 1.0)
    for pair in pairs:
        values[pair.index_a, pair.index_b] = values[pair.index_b, pair.index_a] = pair.inlier_ratio
    image = np.zeros((frame_count, frame_count, 3), dtype=np.uint8)
    known = np.isfinite(values)
    image[:] = (210, 210, 210)
    image[known] = cv2.applyColorMap((np.nan_to_num(values) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[known]
    image = cv2.resize(image, (700, 700), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((780, 760, 3), 250, dtype=np.uint8)
    canvas[55:755, 40:740] = image
    cv2.putText(canvas, title, (40, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
    cv2.putText(canvas, "color = fundamental-matrix inlier ratio; gray = pair not analyzed", (40, 775), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


def contact_sheet(scene, indices: list[int], path: Path, title: str) -> None:
    thumb = (280, 186)
    columns = 5
    rows = (len(indices) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb[0], rows * (thumb[1] + 32) + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for position, index in enumerate(indices):
        with Image.open(scene.image_paths[index]) as source:
            image = source.convert("RGB")
            image.thumbnail(thumb)
        x, y = (position % columns) * thumb[0], 40 + (position // columns) * (thumb[1] + 32)
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + thumb[1] + 4), f"{index}: {scene.image_paths[index].name}", fill="black")
    sheet.save(path, quality=88)


def subset_statistics(indices: list[int], lookup) -> dict:
    adjacent = [lookup[(first, second)] for first, second in zip(indices, indices[1:])]
    centers = [pair.center_distance for pair in adjacent]
    angles = [pair.viewing_angle_deg for pair in adjacent]
    matches = [pair.ratio_matches for pair in adjacent]
    inliers = [pair.fundamental_inliers for pair in adjacent]
    return {
        "adjacent_pairs": [[pair.index_a, pair.index_b] for pair in adjacent],
        "camera_center_spacing": centers,
        "viewing_angle_deg": angles,
        "ratio_match_counts": matches,
        "fundamental_inlier_counts": inliers,
        "mean_center_spacing": float(np.mean(centers)),
        "mean_viewing_angle_deg": float(np.mean(angles)),
        "mean_ratio_matches": float(np.mean(matches)),
        "min_ratio_matches": int(min(matches)),
        "mean_fundamental_inliers": float(np.mean(inliers)),
        "min_fundamental_inliers": int(min(inliers)),
        "index_span": int(indices[-1] - indices[0]),
    }


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    started = time.perf_counter()
    final = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_parameters": {
            "max_image_dimension": MAX_DIMENSION,
            "orb_nfeatures": NFEATURES,
            "max_pair_gap": MAX_PAIR_GAP,
            "lowe_ratio": RATIO,
            "fundamental_ransac_threshold_px": RANSAC_THRESHOLD,
            "fundamental_ransac_confidence": 0.999,
            "nested_sizes": [2, 4, 6, 8, 10],
            "anchor_min_index_gap": 2,
            "min_fundamental_inliers": 20,
            "min_normalized_matches": 0.015,
            "tie_break": "lowest original indices",
        },
        "scenes": {},
    }
    for scene_name in SCENES:
        scene_started = time.perf_counter()
        scene_dir = OUTPUT_ROOT / scene_name
        scene_dir.mkdir()
        scene = load_scene(ETH3D_ROOT, scene_name)
        geometry, pairs, scales = analyze_scene_pairs(
            scene, MAX_PAIR_GAP, MAX_DIMENSION, NFEATURES, RATIO, RANSAC_THRESHOLD
        )
        metadata = {
            "image_count": len(scene.image_paths),
            "analysis_scales": scales,
            "image_resolutions": sorted({tuple(Image.open(path).size) for path in scene.image_paths}),
        }
        save_pair_cache(scene_dir / "pair_metrics.json", scene_name, geometry, pairs, metadata)
        lookup = pair_lookup(pairs)
        neighbor_pairs = [lookup[(index, index + 1)] for index in range(len(scene.image_paths) - 1)]
        centers = np.asarray([frame.camera_center for frame in geometry])
        trajectory_plot(centers, scene_dir / "camera_trajectory.png", f"{scene_name}: ordered ETH3D camera trajectory")
        line_plot([pair.center_distance for pair in neighbor_pairs], scene_dir / "neighbor_displacement.png", f"{scene_name}: neighboring camera displacement", "ETH3D coordinate distance")
        line_plot([pair.viewing_angle_deg for pair in neighbor_pairs], scene_dir / "neighbor_viewing_angle.png", f"{scene_name}: neighboring viewing-direction change", "degrees")
        line_plot([pair.ratio_matches for pair in neighbor_pairs], scene_dir / "neighbor_feature_matches.png", f"{scene_name}: neighboring ORB ratio matches", "match count")
        line_plot([pair.fundamental_inliers for pair in neighbor_pairs], scene_dir / "neighbor_feature_inliers.png", f"{scene_name}: neighboring fundamental-matrix inliers", "inlier count")
        overlap_heatmap(len(scene.image_paths), pairs, scene_dir / "local_overlap_heatmap.png", f"{scene_name}: local feature-overlap proxy")

        centered_start = (len(scene.image_paths) - 10) // 2
        methods = {
            "A_centered_contiguous": window_summary("centered", centered_start, 10, pairs),
            "B_pose_constrained": rank_windows(len(scene.image_paths), pairs, "pose")[0],
            "C_feature_constrained": rank_windows(len(scene.image_paths), pairs, "feature")[0],
            "D_hybrid_pose_feature": rank_windows(len(scene.image_paths), pairs, "hybrid")[0],
        }
        comparisons = []
        for method_name, window in methods.items():
            indices = window.indices
            nested = build_nested_subsets(indices, pairs)
            contact_sheet(scene, indices, scene_dir / f"candidate_{method_name}.jpg", f"{scene_name} | {method_name} | S10 candidate")
            comparisons.append({**asdict(window), "method": method_name, "indices": indices,
                                "nested_subsets": {f"S{count}": values for count, values in nested.items()}})
        with (scene_dir / "candidate_windows.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["method", "start", "stop", "score", "mean_neighbor_distance", "mean_neighbor_angle_deg", "mean_neighbor_matches", "mean_neighbor_inliers", "indices"])
            writer.writeheader()
            for row in comparisons:
                writer.writerow({key: row[key] for key in writer.fieldnames})

        chosen = methods["D_hybrid_pose_feature"]
        nested = build_nested_subsets(chosen.indices, pairs)
        statistics, filenames = {}, {}
        for count, indices in nested.items():
            key = f"S{count}"
            statistics[key] = subset_statistics(indices, lookup)
            filenames[key] = [scene.image_paths[index].name for index in indices]
            contact_sheet(scene, indices, scene_dir / f"final_{key}.jpg", f"{scene_name} | {PROTOCOL_VERSION} | {key}")
        jumps = [{"after_index": index, "distance": pair.center_distance, "angle_deg": pair.viewing_angle_deg,
                  "inliers": pair.fundamental_inliers} for index, pair in enumerate(neighbor_pairs)
                 if pair.center_distance > np.median([item.center_distance for item in neighbor_pairs]) * 2.5
                 or pair.viewing_angle_deg > 25 or pair.fundamental_inliers < 20]
        final["scenes"][scene_name] = {
            "image_count": len(scene.image_paths),
            "filenames_in_order": [path.name for path in scene.image_paths],
            "candidate_methods": comparisons,
            "selected_method": "D_hybrid_pose_feature",
            "selected_window": asdict(chosen),
            "subsets": {f"S{count}": indices for count, indices in nested.items()},
            "filenames": filenames,
            "statistics": statistics,
            "ordering_diagnostics": {
                "lexical_matches_pose_order": True,
                "timestamps_available": False,
                "neighbor_distance_median": float(np.median([pair.center_distance for pair in neighbor_pairs])),
                "neighbor_angle_median_deg": float(np.median([pair.viewing_angle_deg for pair in neighbor_pairs])),
                "potential_jumps": jumps,
            },
            "analysis_seconds": time.perf_counter() - scene_started,
        }
    final["total_analysis_seconds"] = time.perf_counter() - started
    (OUTPUT_ROOT / "selection_recommendations.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_analysis_seconds": final["total_analysis_seconds"],
                      "selections": {scene: record["subsets"] for scene, record in final["scenes"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

