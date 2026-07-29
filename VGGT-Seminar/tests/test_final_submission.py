from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "final_submission"
STEM = "Peleg_Shpitzer_Razi_Mreeh_Final_Report"
MD = OUT / f"{STEM}.md"
DOCX = OUT / f"{STEM}.docx"
PDF = OUT / f"{STEM}.pdf"
VALIDATION = OUT / "validation" / "final_validation_report.json"
COURSE = "Seminar in Deep Learning for Solving Computer Vision Problems"
HEADER = "SEMINAR IN DEEP LEARNING FOR COMPUTER VISION | VGGT EVALUATION"


def pdf_text() -> tuple[list[str], str]:
    pages = [(page.extract_text() or "") for page in PdfReader(str(PDF)).pages]
    return pages, "\n".join(pages)


def test_required_artifacts_and_page_count():
    for path in (MD, DOCX, PDF, VALIDATION, OUT / "validation" / "final_page_contact_sheet.png"):
        assert path.exists() and path.stat().st_size > 0
    assert len(PdfReader(str(PDF)).pages) == 17


def test_course_identity_and_english_only():
    pages, text = pdf_text()
    md = MD.read_text(encoding="utf-8")
    docx_text = "\n".join(p.text for p in Document(DOCX).paragraphs)
    combined = "\n".join((md, docx_text, text))
    assert COURSE in combined
    assert HEADER not in pages[0]
    assert all(HEADER in page for page in pages[1:])
    assert not re.search(r"VGGT\s+SEMINAR\s+REPORT", combined, re.IGNORECASE)
    assert not re.search(r"\bVGGT\s+seminar\b|seminar\s+on\s+VGGT", combined, re.IGNORECASE)
    assert not re.search(r"[\u0590-\u05ff]", combined)


def test_scientific_values_and_split_roles_are_preserved():
    text = MD.read_text(encoding="utf-8")
    required = (
        "0.031", "0.069", "0.77°", "0.1279", "step 15", "P001–P005",
        "P006", "P000", "25.85 s", "5.947 GiB", "6.523 GiB",
        "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0",
    )
    for value in required:
        assert value in text


def test_submission_exclusions_and_rendering_tokens():
    _, text = pdf_text()
    lowered = text.lower()
    for forbidden in (
        "university of", "instructor:", "course number", "author contribution",
        "contribution statement", "draft report", "report version",
    ):
        assert forbidden not in lowered
    assert not re.search(r"\$[^$]+\$|\\(?:frac|delta|hat|rho|sqrt)\b", text)
    assert "References" in text


def test_validation_report_passes():
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))
    assert report["final_page_count"] == 17
    assert report["english_only"] is True
    assert report["course_title_correct"] is True
    assert report["all_vggt_seminar_report_instances_removed"] is True
    assert report["numerical_integrity_confirmed"] is True
    assert report["figure_table_integrity_confirmed"] is True
    assert report["visual_inspection_confirmed"] is True
    assert report["zero_inference"] is True
    assert report["zero_training"] is True
    assert report["zero_downloads"] is True
    assert report["remaining_issues"] == "none"
