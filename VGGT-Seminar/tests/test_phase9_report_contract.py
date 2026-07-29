import csv
import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
DATA = REPORT / "data"


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_unified_dataset_has_the_frozen_ten_conditions():
    rows = _rows(DATA / "view_count_results.csv")
    assert len(rows) == 10
    assert {(r["scene"], r["subset"], int(r["view_count"])) for r in rows} == {
        (scene, f"S{count}", count)
        for scene in ("delivery_area", "courtyard")
        for count in (2, 4, 6, 8, 10)
    }
    assert {r["protocol_version"] for r in rows} == {"eth3d-overlap-aware-nested-v1"}
    assert len([r for r in rows if r["scene"] == "delivery_area"]) == 5
    assert len([r for r in rows if r["scene"] == "courtyard"]) == 5


def test_json_csv_and_cross_scene_are_consistent():
    csv_rows = _rows(DATA / "view_count_results.csv")
    json_rows = json.loads((DATA / "view_count_results.json").read_text(encoding="utf-8"))
    assert len(json_rows) == len(csv_rows)
    assert [(r["scene"], r["subset"]) for r in json_rows] == [
        (r["scene"], r["subset"]) for r in csv_rows
    ]
    assert len(_rows(DATA / "cross_scene_results.csv")) == 5


def test_provenance_hashes_and_checkpoint_are_valid():
    provenance = json.loads((DATA / "report_provenance.json").read_text(encoding="utf-8"))
    assert provenance["generated_from_saved_outputs_only"] is True
    assert set(provenance["source_commits"]) == {"5ef68d3", "ccb5487"}
    for item in provenance["sources"]:
        path = ROOT / item["path"]
        if not path.is_file():
            pytest.skip("canonical generated experiment outputs are intentionally not committed")
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_all_tables_and_figures_exist_and_are_referenced():
    markdown = (REPORT / "vggt_seminar_report.md").read_text(encoding="utf-8")
    for number in range(1, 9):
        stem = f"table{number:02d}_"
        assert len(list((REPORT / "tables").glob(stem + "*.csv"))) == 1
        assert len(list((REPORT / "tables").glob(stem + "*.md"))) == 1
        assert f"Table {number}." in markdown
    for number in range(1, 11):
        figures = list((REPORT / "figures").glob(f"fig{number:02d}_*.png"))
        assert len(figures) == 1
        assert f"Figure {number}." in markdown
        assert figures[0].name in markdown


def test_markdown_local_links_resolve():
    markdown = (REPORT / "vggt_seminar_report.md").read_text(encoding="utf-8")
    for target in re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", markdown):
        if "://" not in target and not target.startswith("#"):
            assert (REPORT / target).resolve().is_file(), target


def test_notebook_is_unexecuted_and_has_saved_only_synthesis():
    notebook = json.loads((ROOT / "notebooks/02_vggt_multi_input_experiments.ipynb").read_text(encoding="utf-8"))
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"][-2:])
    assert "Phase 9 report synthesis" in source
    assert "view_count_results.csv" in source
    assert "model(" not in source and "load_state_dict" not in source


def test_report_generation_has_no_model_execution_path():
    source = (ROOT / "scripts/build_phase9_report.py").read_text(encoding="utf-8").lower()
    assert "import torch" not in source
    assert "from vggt" not in source
    assert "load_state_dict" not in source


def test_report_contains_boundaries_without_accuracy_overclaims():
    text = (REPORT / "vggt_seminar_report.md").read_text(encoding="utf-8").lower()
    for required in (
        "eth3d-overlap-aware-nested-v1", "delivery_area", "courtyard",
        "one run per condition", "does not claim statistical significance", "no similarity alignment",
        "confidence is not accuracy", "arbitrary", "order sensitivity",
        "degradation", "fine-tuning",
    ):
        assert required in text
    for forbidden in (
        "statistically significant improvement", "proves that", "superior reconstruction",
        "better reconstruction accuracy", "ground-truth agreement was", "metric-scale reconstruction",
    ):
        assert forbidden not in text
    assert (REPORT / "vggt_seminar_report.docx").stat().st_size > 100_000
