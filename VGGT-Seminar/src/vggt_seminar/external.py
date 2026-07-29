from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .paths import find_project_root


def load_vggt_pin(root: Path | None = None) -> dict[str, Any]:
    base = root or find_project_root()
    path = base / "external/VGGT_PIN.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    commit = data.get("commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("VGGT pin must contain a full lowercase 40-character Git hash")
    return data


def checkout_status(root: Path | None = None) -> dict[str, Any]:
    base = root or find_project_root()
    checkout = base / load_vggt_pin(base)["checkout_path"]
    if not (checkout / ".git").exists():
        return {"exists": False, "matches_pin": False, "clean": None, "commit": None}
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {"exists": True, "matches_pin": commit == load_vggt_pin(base)["commit"],
            "clean": not bool(dirty), "commit": commit}
