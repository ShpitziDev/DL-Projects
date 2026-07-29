from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vggt_seminar.live_demo import point_cloud_preview, save_ply


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/predictions/phase6_eth3d_smoke/delivery_area/2_views_evenly_spaced_original"


def main() -> int:
    direct = torch.load(OUTPUT / "arrays/world_points.pt", map_location="cpu", weights_only=True)[0].float().numpy()
    confidence = torch.load(OUTPUT / "arrays/world_points_conf.pt", map_location="cpu", weights_only=True)[0].float().numpy()
    images = torch.load(OUTPUT / "arrays/images.pt", map_location="cpu", weights_only=True)[0].permute(0, 2, 3, 1).float().numpy()
    threshold = float(np.median(confidence))
    filtered = direct.copy()
    filtered[confidence <= threshold] = np.nan
    retained = save_ply(filtered, images, OUTPUT / "visualizations/point_cloud_confidence_filtered.ply")
    point_cloud_preview(filtered, images).save(OUTPUT / "visualizations/point_cloud_confidence_filtered_preview.png")
    runtime_path = OUTPUT / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["point_counts"]["confidence_filtered"] = retained
    runtime["confidence_filter"] = {
        "rule": "world point confidence > median",
        "threshold": threshold,
        "repair_note": "Recomputed from saved CPU tensors after smoke run; no additional inference.",
    }
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print({"threshold": threshold, "retained_points": retained, "source_points": direct.shape[0] * direct.shape[1] * direct.shape[2]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

