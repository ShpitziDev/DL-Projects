from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vggt_seminar.tartanair import load_trajectory, orb_pair_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local_assets/datasets/tartanair_v2/ArchVizTinyHouseDay/Data_easy"
OUT = ROOT / "outputs/analysis/v3_tartanair_audit_20260729"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    trajectory = load_trajectory(DATA, "P000")
    candidates = []
    for start in range(0, len(trajectory) - 10):
        indices = list(range(start, start + 10))
        pairs = [orb_pair_score(trajectory.image_paths[a], trajectory.image_paths[b])
                 for a, b in zip(indices, indices[1:])]
        centers = trajectory.poses_xyzw[indices, :3]
        path_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
        candidates.append({
            "start": start,
            "indices": indices,
            "min_adjacent_inliers": min(p["inliers"] for p in pairs),
            "median_adjacent_inliers": float(np.median([p["inliers"] for p in pairs])),
            "path_length_m": path_length,
        })
    eligible = [c for c in candidates if c["min_adjacent_inliers"] >= 100]
    selected = max(eligible, key=lambda c: (c["path_length_m"], c["median_adjacent_inliers"]))
    indices = selected["indices"]
    subsets = {f"S{count}": indices[:count] for count in (2, 4, 6, 8, 10)}
    selected["subsets"] = subsets
    (OUT / "audit.json").write_text(json.dumps({"selected": selected, "candidates": candidates}, indent=2) + "\n")

    thumbs = []
    for index in indices:
        image = Image.open(trajectory.image_paths[index]).convert("RGB").resize((256, 256))
        canvas = Image.new("RGB", (256, 286), "white")
        canvas.paste(image, (0, 30))
        ImageDraw.Draw(canvas).text((8, 8), f"P000 frame {index:06d}", fill="black")
        thumbs.append(canvas)
    sheet = Image.new("RGB", (256 * 5, 286 * 2), "white")
    for i, image in enumerate(thumbs):
        sheet.paste(image, ((i % 5) * 256, (i // 5) * 286))
    sheet.save(OUT / "selected_contact_sheet.jpg", quality=94)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
