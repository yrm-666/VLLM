"""Read-only preflight check to run immediately after renting a GPU."""

from __future__ import annotations

import importlib
import json
import platform
import sys


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
        "cuda": {"available": False},
    }
    for package, import_name in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("decord2", "decord"),
        ("accelerate", "accelerate"),
    ):
        try:
            module = importlib.import_module(import_name)
            report["packages"][package] = getattr(module, "__version__", "installed")
        except Exception as error:
            report["packages"][package] = f"ERROR: {type(error).__name__}: {error}"

    try:
        import torch

        cuda: dict[str, object] = {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
        }
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            cuda.update(
                {
                    "device": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "vram_gb": round(properties.total_memory / (1024**3), 2),
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
        report["cuda"] = cuda
    except Exception:
        pass

    print(json.dumps(report, ensure_ascii=False, indent=2))
    python_ok = sys.version_info[:2] == (3, 11)
    cuda_ok = bool(report["cuda"].get("available"))
    return 0 if python_ok and cuda_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

