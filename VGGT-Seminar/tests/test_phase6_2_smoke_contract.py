import json
from pathlib import Path

import yaml
import pytest

from vggt_seminar.eth3d_overlap import PROTOCOL_VERSION, load_frozen_selection

ROOT = Path(__file__).resolve().parents[1]


def test_phase6_2_is_exactly_the_frozen_s2_condition() -> None:
    config = yaml.safe_load((ROOT / "configs/experiments/phase6_2_overlap_smoke.yaml").read_text(encoding="utf-8"))
    frozen_path = ROOT / config["protocol"]["config"]
    s2 = load_frozen_selection(frozen_path, "delivery_area", 2)
    s4 = load_frozen_selection(frozen_path, "delivery_area", 4)
    assert config["protocol"]["version"] == PROTOCOL_VERSION == s2["protocol_version"]
    assert config["input"] == {"source": "eth3d", "scene": "delivery_area", "frame_count": 2,
                               "selection_strategy": "overlap_aware_nested", "order": "original",
                               "preprocessing_mode": "crop"}
    assert s2["indices"] == config["constraints"]["expected_original_indices"] == [0, 6]
    assert s2["filenames"] == config["constraints"]["expected_filenames"]
    assert set(s2["indices"]) < set(s4["indices"])
    assert config["constraints"]["expected_forward_passes"] == 1
    assert config["constraints"]["allow_selection_fallback"] is False


def test_runner_is_offline_one_forward_and_fail_closed() -> None:
    source = (ROOT / "scripts/run_phase6_eth3d_smoke.py").read_text(encoding="utf-8")
    assert source.count("predictions = model(") == 1
    assert "from_pretrained(" not in source
    assert "load_frozen_selection(protocol_path" in source
    assert "allow_selection_fallback" in source
    assert "overlap_aware_nested" in source


def test_source_notebook_remains_unexecuted() -> None:
    notebook = json.loads((ROOT / "notebooks/02_vggt_multi_input_experiments.ipynb").read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is None and cell.get("outputs") == [] for cell in code)


def test_local_phase6_2_artifacts_match_frozen_protocol_when_present() -> None:
    output = ROOT / "outputs/predictions/phase6_2_eth3d_overlap_smoke/delivery_area/S2_overlap_aware_nested_original"
    if not output.is_dir():
        pytest.skip("Ignored local Phase 6.2 artifacts are not present")
    runtime = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert runtime["forward_pass_count"] == 1
    assert preflight["selected_indices"] == [0, 6]
    assert preflight["selected_filenames"] == ["DSC_0675.JPG", "DSC_0681.JPG"]
    assert protocol["version"] == PROTOCOL_VERSION
    assert runtime["confidence_filter"]["rule"] == "world point confidence > median"
    assert runtime["confidence_filter"]["points_retained"] == runtime["point_counts"]["confidence_filtered"]
