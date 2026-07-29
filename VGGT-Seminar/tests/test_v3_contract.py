from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_v3_split_has_no_leakage():
    config = yaml.safe_load(
        (ROOT / "configs/experiments/v3_tartanair_finetune.yaml").read_text()
    )
    train = set(config["split"]["train"])
    validation = set(config["split"]["validation"])
    test = set(config["split"]["test"])
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert test == {"P000"}


def test_v3_nested_evaluation_protocol():
    config = yaml.safe_load(
        (ROOT / "configs/experiments/v3_tartanair_frozen.yaml").read_text()
    )
    subsets = config["selection"]["subsets"]
    assert list(subsets) == ["S2", "S4", "S6", "S8", "S10"]
    for left, right in zip(subsets.values(), list(subsets.values())[1:]):
        assert set(left) < set(right)


def test_v3_report_deliverables_exist():
    report = ROOT / "report/v3"
    assert (report / "vggt_seminar_report_v3.docx").stat().st_size > 100_000
    assert (report / "vggt_seminar_report_v3.pdf").stat().st_size > 100_000
    assert "Razi Mreeh" in (report / "README.md").read_text()
