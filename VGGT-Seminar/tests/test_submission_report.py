import csv
import json
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "submission"
STEM = "Peleg_Shpitzer_Razi_Mreeh_VGGT_Seminar_Report"
PDF = OUT / f"{STEM}.pdf"
DOCX = OUT / f"{STEM}.docx"
MD = OUT / f"{STEM}.md"


def pdf_text():
    return "\n".join(page.extract_text() or "" for page in PdfReader(PDF).pages)


def docx_text():
    with zipfile.ZipFile(DOCX) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    return re.sub(r"<[^>]+>", " ", xml)


def test_artifacts_and_page_count():
    assert PDF.is_file() and DOCX.is_file() and MD.is_file()
    assert len(PdfReader(PDF).pages) == 16
    assert (OUT / "validation" / "final_page_contact_sheet.png").is_file()


def test_visible_language_contract():
    combined = (pdf_text() + "\n" + docx_text()).lower()
    forbidden = (
        "university of haifa",
        "course:",
        "instructor:",
        "author contribution",
        "contribution split",
        "pending confirmation",
        "placeholder",
        "version",
        "working",
        "draft",
    )
    for value in forbidden:
        assert value not in combined
    assert "peleg shpitzer" in combined and "razi mreeh" in combined


def test_no_raw_latex_in_pdf():
    text = pdf_text()
    for token in (
        r"\(",
        r"\)",
        r"\[",
        r"\]",
        r"\frac",
        r"\mathrm",
        r"\hat",
        r"\Delta",
        r"\top",
        r"\qquad",
        r"\left",
        r"\right",
        r"\mathbf",
    ):
        assert token not in text


def test_required_figures_and_eth3d_pages():
    figures = list((OUT / "figures").glob("*.png"))
    assert len(figures) >= 21
    assert (OUT / "figures" / "eth3d_delivery_area_evidence.png").is_file()
    assert (OUT / "figures" / "eth3d_courtyard_evidence.png").is_file()
    text = pdf_text()
    assert "7. ETH3D delivery_area" in text
    assert "8. ETH3D courtyard" in text


def test_references_are_separate_and_numbered():
    text = pdf_text()
    for number in range(1, 10):
        assert re.search(rf"(?m)^\s*{number}\.\s", text)


def test_canonical_boundaries_and_values():
    text = pdf_text()
    assert "P001–P005" in text
    assert "P006" in text and "selected step 15" in text
    assert "P000" in text and "never used" in text
    for value in ("0.0314", "0.0689", "0.0401", "0.1279", "0.9778", "0.8485"):
        assert value in text
    validation = json.loads((ROOT / "report" / "v4" / "validation" / "numerical_validation.json").read_text())
    assert validation["independent_environment"]["new_forward_passes"] == 0
    assert validation["adaptation"]["additional_adaptation_for_v4"] is False


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} submission checks passed")
