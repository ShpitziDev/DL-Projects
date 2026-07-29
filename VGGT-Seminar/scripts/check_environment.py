from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vggt_seminar.environment import collect_environment  # noqa: E402
from vggt_seminar.paths import ensure_project_directories  # noqa: E402


def main() -> int:
    ensure_project_directories(ROOT)
    report = collect_environment(ROOT)
    destination = ROOT / "outputs/environment/environment_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
