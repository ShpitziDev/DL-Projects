from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/environment/pytorch_cuda_validation.json"


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA validation failed: torch.cuda.is_available() is False")

    device = torch.device("cuda:0")
    device_index = 0
    torch.cuda.set_device(device_index)
    warmup = torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device_index)
    before = torch.cuda.memory_allocated(device)
    a = torch.randn((256, 256), device=device)
    b = torch.randn((256, 256), device=device)
    c = a @ b
    torch.cuda.synchronize(device)
    if c.device.type != "cuda" or not torch.isfinite(c).all().item():
        raise RuntimeError("CUDA validation failed: matrix result is invalid or not on CUDA")

    bf16_supported = bool(torch.cuda.is_bf16_supported())
    bf16_ok = False
    if bf16_supported:
        x = torch.randn((64, 64), device=device, dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize(device)
        bf16_ok = y.device.type == "cuda" and y.dtype == torch.bfloat16
        del x, y

    peak = torch.cuda.max_memory_allocated(device_index)
    del warmup, a, b, c
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    after_cleanup = torch.cuda.memory_allocated(device)
    props = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": 1,
        "status": "passed",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(props.total_memory),
        "matrix_multiply_passed": True,
        "bfloat16_supported": bf16_supported,
        "bfloat16_operation_passed": bf16_ok,
        "memory_allocated_before_bytes": int(before),
        "peak_memory_allocated_bytes": int(peak),
        "memory_allocated_after_cleanup_bytes": int(after_cleanup),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
