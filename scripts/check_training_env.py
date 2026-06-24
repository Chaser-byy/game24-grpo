#!/usr/bin/env python3
"""Fail early with actionable dependency diagnostics for GPU training scripts."""

from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "peft": "peft",
    "trl": "trl",
    "accelerate": "accelerate",
    "onnxscript": "onnxscript",
}


def main() -> None:
    missing = []
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)

    if missing:
        packages = " ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing Python packages for training: "
            f"{packages}\nInstall them with:\n"
            f"  python -m pip install -U {packages}"
        )

    import torch
    import transformers
    import trl
    import peft

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this training pipeline requires one NVIDIA GPU")

    device = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"Transformers: {transformers.__version__}; TRL: {trl.__version__}; PEFT: {peft.__version__}")
    print(f"GPU: {device} ({memory_gb:.1f} GB), bf16={torch.cuda.is_bf16_supported()}")


if __name__ == "__main__":
    main()
