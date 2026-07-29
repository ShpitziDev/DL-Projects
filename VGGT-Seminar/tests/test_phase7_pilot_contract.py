import json
from pathlib import Path

import pytest
import yaml

from scripts.run_eth3d_view_count_pilot import METRIC_COLUMNS, SUBSETS, completed_metrics, plot_series, validate_reusable_s2
from vggt_seminar.eth3d_overlap import PROTOCOL_VERSION, load_frozen_selection

ROOT=Path(__file__).resolve().parents[1]


def test_phase7_exact_frozen_nested_original_matrix() -> None:
    config=yaml.safe_load((ROOT/"configs/experiments/phase7_delivery_area_view_count.yaml").read_text(encoding="utf-8"))
    frozen=ROOT/config["protocol"]["config"]
    records=[load_frozen_selection(frozen,"delivery_area",int(name[1:])) for name in SUBSETS]
    assert [record["indices"] for record in records] == [[0,6],[0,3,6,9],[0,3,4,5,6,9],[0,1,2,3,4,5,6,9],list(range(10))]
    assert all(set(a["indices"]) < set(b["indices"]) for a,b in zip(records,records[1:]))
    assert all(record["indices"] == sorted(record["indices"]) for record in records)
    assert config["order"] == "original" and config["constraints"]["allow_fallback"] is False


def test_invalid_s2_reuse_is_rejected(tmp_path: Path) -> None:
    valid, errors=validate_reusable_s2(tmp_path,{"checkpoint_sha256":"x"},{"indices":[0,6],"filenames":["a","b"]})
    assert not valid and errors


def test_valid_s2_reuse_fixture(tmp_path: Path) -> None:
    required=["runtime.json","preflight.json","tensor_summaries.json","camera_parameters.json","frozen_protocol.json","selected_frames.json",
              "arrays/depth.pt","arrays/world_points.pt","visualizations/contact_sheet.jpg","visualizations/point_cloud_direct.ply",
              "visualizations/point_cloud_depth_unprojected.ply","visualizations/point_cloud_confidence_filtered.ply"]
    for name in required: path=tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b"x")
    (tmp_path/"runtime.json").write_text(json.dumps({"status":"passed","forward_pass_count":1,"precision":"torch.bfloat16 autocast","flash_sdp_enabled":False}))
    (tmp_path/"preflight.json").write_text(json.dumps({"selected_indices":[0,6],"selected_filenames":["a","b"],"checkpoint_sha256":"x"}))
    (tmp_path/"frozen_protocol.json").write_text(json.dumps({"version":PROTOCOL_VERSION}))
    (tmp_path/"tensor_summaries.json").write_text(json.dumps({"depth":{"finite_percentage":100.0}}))
    valid,errors=validate_reusable_s2(tmp_path,{"checkpoint_sha256":"x"},{"indices":[0,6],"filenames":["a","b"]})
    assert valid and not errors


def test_metrics_schema_and_plot_generation(tmp_path: Path) -> None:
    assert {"subset","view_count","indices","inference_seconds","retained_points","camera_path_length","reused_existing_result"} <= set(METRIC_COLUMNS)
    rows=[{"view_count":count,"inference_seconds":float(count)} for count in (2,4,6,8,10)]
    target=tmp_path/"plot.png"; plot_series(rows,["inference_seconds"],"test","seconds",target); assert target.is_file()


def test_completed_detection_and_partial_recovery(tmp_path: Path) -> None:
    result=tmp_path/"S4"
    assert completed_metrics(result,[0,3,6,9],"hash") is None
    result.mkdir()
    with pytest.raises(RuntimeError,match="Partial result"):
        completed_metrics(result,[0,3,6,9],"hash")
    (result/"metrics.json").write_text(json.dumps({"indices":[0,3,6,9],"checkpoint_sha256":"hash"}))
    (result/"SUCCESS").write_text("validated\n")
    assert completed_metrics(result,[0,3,6,9],"hash")["resumed_existing"] is True


def test_runner_forward_scope_resume_and_notebook_contract() -> None:
    source=(ROOT/"scripts/run_eth3d_view_count_pilot.py").read_text(encoding="utf-8")
    assert source.count("predictions = model(") == 1
    assert "from_pretrained(" not in source and "load_scene(ETH3D_ROOT, scene_name)" in source
    assert "constraints\"][\"approved_scene" in source and "SUCCESS" in source and "resumed_existing" in source
    notebook=json.loads((ROOT/"notebooks/02_vggt_multi_input_experiments.ipynb").read_text(encoding="utf-8"))
    assert all(c.get("execution_count") is None and c.get("outputs")==[] for c in notebook["cells"] if c["cell_type"]=="code")


def test_local_aggregate_consistency_when_present() -> None:
    root=ROOT/"outputs/experiments/phase7_delivery_area_view_count"
    if not root.is_dir(): pytest.skip("Ignored local Phase 7 aggregate is absent")
    import csv
    with (root/"summary.csv").open(encoding="utf-8",newline="") as handle: csv_rows=list(csv.DictReader(handle))
    json_rows=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    assert [row["subset"] for row in csv_rows] == [row["subset"] for row in json_rows] == list(SUBSETS)
    assert manifest["s2_reused"] is True and manifest["new_forward_pass_count"] == 4
    assert len(list((root/"plots").glob("*.png"))) == 11
    for subset in SUBSETS[1:]:
        assert (ROOT/f"outputs/predictions/phase7_eth3d_view_count/delivery_area/{subset}_overlap_aware_nested_original/SUCCESS").is_file()
