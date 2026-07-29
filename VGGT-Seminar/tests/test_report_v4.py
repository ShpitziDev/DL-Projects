import csv
import json
import math
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "v4"
MD = OUT / "seminar_report_working.md"


def docx_text(path):
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    return re.sub(r"<[^>]+>", " ", xml)


def test_required_artifacts_exist():
    for name in (
        "seminar_report_working.md",
        "seminar_report_working.docx",
        "seminar_report_working.pdf",
    ):
        assert (OUT / name).is_file()
    assert len(list((OUT / "figures").glob("*.png"))) >= 9


def test_visible_submission_language():
    text = MD.read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in (
        "version 4",
        "version 3",
        "experimental draft",
        "assignment brief",
        "את התרגיל",
        "checklist",
    ):
        assert forbidden not in lowered
    assert "Reproducing and Evaluating VGGT:" in text
    assert "A controlled study on ETH3D and TartanAir" in text
    assert "Peleg Shpitzer" in text and "Razi Mreeh" in text


def test_required_scientific_content():
    text = MD.read_text(encoding="utf-8")
    for required in (
        "RQ1",
        "RQ2",
        "RQ3",
        "Reproduction statement",
        "Umeyama",
        "AbsRel",
        "RMSE",
        "Spearman",
        "P001–P005",
        "P006",
        "P000",
        "zero new VGGT forward passes",
        "only report content pending author confirmation",
    ):
        assert required in text
    assert "MULTI-VIEW RGB IMAGES" in (ROOT / "scripts" / "build_v4_figures.py").read_text()


def test_saved_output_validation_passed():
    validation = json.loads((OUT / "validation" / "numerical_validation.json").read_text())
    assert len(validation["numerical_checks"]) == 10
    assert all(
        check[key]
        for check in validation["numerical_checks"]
        for key in ("abs_rel", "rmse", "delta1", "confidence_rho")
    )
    assert validation["independent_environment"]["new_forward_passes"] == 0
    assert validation["adaptation"]["additional_adaptation_for_v4"] is False
    recommendation = json.loads((OUT / "data" / "independent_environment_recommendation.json").read_text())
    assert recommendation["new_forward_passes"] == 0
    assert recommendation["status"].startswith("not_executed")


def test_canonical_table_values():
    with (OUT / "tables" / "pretrained_p000.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["views"]) for row in rows] == [2, 4, 6, 8, 10]
    assert math.isclose(float(rows[0]["depth_abs_rel"]), 0.03142585, abs_tol=1e-8)
    assert math.isclose(float(rows[-1]["depth_abs_rel"]), 0.0689439, abs_tol=3e-8)


def test_docx_contains_no_forbidden_labels():
    text = docx_text(OUT / "seminar_report_working.docx").lower()
    for forbidden in ("version 4", "version 3", "experimental draft", "את התרגיל"):
        assert forbidden not in text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} report checks passed")
