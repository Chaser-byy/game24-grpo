#!/usr/bin/env python3
"""Fail early with actionable dependency diagnostics for GPU training scripts."""

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

STABLE_INSTALL_COMMAND = (
    'python -m pip install -U --force-reinstall '
    '"numpy==1.26.4" '
    '"accelerate==1.2.1" '
    '"datasets==3.2.0" '
    '"transformers==4.46.3" '
    '"peft==0.14.0" '
    '"trl==0.15.2" '
    "onnx onnxscript"
)


def main() -> None:
    missing = []
    broken = []
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)
        except Exception as error:  # noqa: BLE001 - this is a diagnostic script.
            broken.append((module_name, type(error).__name__, str(error)))

    if missing or broken:
        messages = []
        if missing:
            messages.append("Missing Python packages: " + " ".join(sorted(set(missing))))
        if broken:
            messages.append("Packages that are installed but fail to import:")
            messages.extend(
                f"  - {module}: {error_type}: {message}"
                for module, error_type, message in broken
            )
        raise SystemExit(
            "\n".join(messages)
            + "\n\nRecommended stable reinstall command for this project:\n"
            + f"  {STABLE_INSTALL_COMMAND}"
        )

    import numpy as np
    import peft
    import torch
    import transformers
    import trl

    major = int(np.__version__.split(".", 1)[0])
    if major >= 2:
        raise SystemExit(
            f"Installed numpy is {np.__version__}, but this project pins numpy<2 for "
            "the current Transformers/PEFT/TRL stack.\n"
            "Recommended stable reinstall command:\n"
            f"  {STABLE_INSTALL_COMMAND}"
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this training pipeline requires one NVIDIA GPU")

    device = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(
        f"Transformers: {transformers.__version__}; "
        f"TRL: {trl.__version__}; PEFT: {peft.__version__}"
    )
    print(f"GPU: {device} ({memory_gb:.1f} GB), bf16={torch.cuda.is_bf16_supported()}")


if __name__ == "__main__":
    main()
