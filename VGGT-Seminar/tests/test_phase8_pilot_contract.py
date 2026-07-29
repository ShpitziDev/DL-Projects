import csv
import json
from pathlib import Path

import pytest
import yaml

from scripts.run_eth3d_view_count_pilot import METRIC_COLUMNS, SUBSETS
from vggt_seminar.eth3d_overlap import PROTOCOL_VERSION, load_frozen_selection

ROOT=Path(__file__).resolve().parents[1]


def test_phase8_exact_frozen_courtyard_matrix() -> None:
    config=yaml.safe_load((ROOT/"configs/experiments/phase8_courtyard_view_count.yaml").read_text(encoding="utf-8"))
    records=[load_frozen_selection(ROOT/config["protocol"]["config"],"courtyard",int(s[1:])) for s in SUBSETS]
    assert [r["indices"] for r in records] == [[0,9],[0,4,6,9],[0,2,3,4,6,9],[0,2,3,4,5,6,7,9],list(range(10))]
    assert [r["filenames"] for r in records] == [
        ["DSC_0286.JPG","DSC_0295.JPG"],
        ["DSC_0286.JPG","DSC_0290.JPG","DSC_0292.JPG","DSC_0295.JPG"],
        ["DSC_0286.JPG","DSC_0288.JPG","DSC_0289.JPG","DSC_0290.JPG","DSC_0292.JPG","DSC_0295.JPG"],
        ["DSC_0286.JPG","DSC_0288.JPG","DSC_0289.JPG","DSC_0290.JPG","DSC_0291.JPG","DSC_0292.JPG","DSC_0293.JPG","DSC_0295.JPG"],
        [f"DSC_{n:04d}.JPG" for n in range(286,296)]]
    assert all(set(a["indices"]) < set(b["indices"]) for a,b in zip(records,records[1:]))
    assert config["scene"]==config["constraints"]["approved_scene"]=="courtyard"
    assert config["s2_reuse"] is None and config["constraints"]["expected_new_forward_passes"]==5
    assert config["order"]=="original" and config["constraints"]["allow_fallback"] is False


def test_phase8_uses_phase7_compatible_schema_and_config_driven_scene() -> None:
    assert tuple(METRIC_COLUMNS)==tuple(__import__('scripts.run_eth3d_view_count_pilot',fromlist=['METRIC_COLUMNS']).METRIC_COLUMNS)
    source=(ROOT/"scripts/run_eth3d_view_count_pilot.py").read_text(encoding="utf-8")
    assert source.count("predictions = model(")==1 and "from_pretrained(" not in source
    assert 'scene_name = config["scene"]' in source and 'load_scene(ETH3D_ROOT, scene_name)' in source
    assert "cross_scene_comparison" in source and "allow_fallback" in source


def test_phase8_source_notebook_unexecuted() -> None:
    notebook=json.loads((ROOT/"notebooks/02_vggt_multi_input_experiments.ipynb").read_text(encoding="utf-8"))
    assert all(c.get("execution_count") is None and c.get("outputs")==[] for c in notebook["cells"] if c["cell_type"]=="code")
    source="\n".join("".join(c.get("source",[])) for c in notebook["cells"])
    assert "delivery_area_vs_courtyard.csv" in source and "Phase 8 courtyard" in source


def test_local_phase8_aggregate_and_cross_merge_when_present() -> None:
    root=ROOT/"outputs/experiments/phase8_courtyard_view_count"
    if not root.is_dir(): pytest.skip("Ignored local Phase 8 results absent")
    with (root/"summary.csv").open(encoding="utf-8",newline="") as handle: csv_rows=list(csv.DictReader(handle))
    json_rows=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    cross=json.loads((root/"comparisons/delivery_area_vs_courtyard.json").read_text(encoding="utf-8"))
    assert [r["subset"] for r in csv_rows]==[r["subset"] for r in json_rows]==list(SUBSETS)
    assert manifest["scene"]=="courtyard" and manifest["new_forward_pass_count"]==5
    assert [r["view_count"] for r in cross]==[2,4,6,8,10]
    assert all("delivery_area_inference_seconds" in r and "courtyard_inference_seconds" in r for r in cross)
    assert len(list((root/"plots").glob("*.png")))==11
    assert len(list((root/"comparisons").glob("cross_*.png")))==5
