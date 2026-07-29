from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


OPTIONAL_ACCELERATION = ("xformers", "flash_attn", "triton")


def collect_environment(project_root: Path, torch_module: Any | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    if torch_module is None and importlib.util.find_spec("torch") is not None:
        import torch as torch_module  # type: ignore[no-redef]
    report: dict[str, Any] = {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "disk": {},
        "torch": {"installed": torch_module is not None},
        "optional_acceleration": {},
        "warnings": warnings,
    }
    disk = shutil.disk_usage(project_root)
    report["disk"] = {"total_bytes": disk.total, "free_bytes": disk.free}
    for package in OPTIONAL_ACCELERATION:
        report["optional_acceleration"][package] = importlib.util.find_spec(package) is not None
    if torch_module is None:
        warnings.append("PyTorch is not installed; CUDA cannot be validated through PyTorch.")
        return report
    cuda = torch_module.cuda
    available = bool(cuda.is_available())
    report["torch"].update({"version": str(torch_module.__version__), "cuda_available": available,
                            "cuda_runtime": getattr(torch_module.version, "cuda", None), "gpu_count": int(cuda.device_count()) if available else 0})
    if not available:
        warnings.append("PyTorch cannot access CUDA; GPU inference is unavailable in this environment.")
        return report
    devices = []
    for index in range(cuda.device_count()):
        props = cuda.get_device_properties(index)
        devices.append({"index": index, "name": cuda.get_device_name(index),
                        "total_memory_bytes": int(props.total_memory),
                        "compute_capability": list(cuda.get_device_capability(index))})
    report["torch"]["devices"] = devices
    report["torch"]["bfloat16_supported"] = bool(cuda.is_bf16_supported())
    return report
