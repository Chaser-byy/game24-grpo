import importlib
import argparse
import platform
import sys
import tempfile
from pathlib import Path


def version_of(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return None, exc
    return getattr(module, "__version__", "unknown"), None


def fail(message, fixes):
    print(f"[FAIL] {message}")
    print("Suggested fix:")
    for line in fixes:
        print(f"  {line}")
    raise SystemExit(1)


def warn(message):
    print(f"[WARN] {message}")


def ok(message):
    print(f"[ OK ] {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-vllm", action="store_true")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    if sys.version_info >= (3, 12):
        message = "Python 3.12 works for some packages, but TinyZero/vLLM is much steadier on Python 3.9 or 3.10."
        if args.strict_vllm:
            fail(message, ["bash scripts/setup_autodl_vllm_env.sh"])
        warn(message)

    versions = {}
    for name in ["numpy", "torch", "pandas", "pyarrow", "transformers", "vllm", "ray"]:
        version, error = version_of(name)
        if error is not None:
            fail(f"Cannot import {name}: {error}", ["pip install -r requirements-autodl.txt"])
        versions[name] = version
        ok(f"{name}=={version}")

    import numpy as np
    import torch

    numpy_major = int(str(versions["numpy"]).split(".", 1)[0])
    if numpy_major >= 2:
        fail(
            f"NumPy {versions['numpy']} is too new for this TinyZero/vLLM stack.",
            [
                'pip install --force-reinstall "numpy==1.26.4" "pandas==2.2.2" "pyarrow==16.1.0"',
            ],
        )

    try:
        tensor = torch.from_numpy(np.array([1, 2, 3], dtype=np.int64))
        assert tensor.tolist() == [1, 2, 3]
        ok("torch.from_numpy smoke test")
    except Exception as exc:
        fail(
            f"torch/numpy ABI smoke test failed: {exc}",
            [
                "conda create -n game24-grpo-clean python=3.10 -y",
                "conda activate game24-grpo-clean",
                "pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121",
                "pip install -r requirements-autodl.txt",
                "pip install -e . --no-build-isolation",
            ],
        )

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "smoke.parquet"
            pq.write_table(pa.Table.from_pylist([{"x": 1}, {"x": 2}]), path)
            rows = pq.read_table(path).to_pylist()
            assert rows == [{"x": 1}, {"x": 2}]
            frame = pd.read_parquet(path)
            assert frame["x"].tolist() == [1, 2]
        ok("pyarrow/pandas parquet smoke test")
    except Exception as exc:
        fail(
            f"parquet smoke test failed: {exc}",
            ['pip install --force-reinstall "numpy==1.26.4" "pandas==2.2.2" "pyarrow==16.1.0"'],
        )

    _, flash_error = version_of("flash_attn")
    if flash_error is None:
        ok("flash_attn is installed; vLLM + flash_attention_2 path is available if desired")
    else:
        warn("flash_attn is not installed; use ROLLOUT_NAME=hf or ATTN_IMPLEMENTATION=sdpa")

    print("Environment preflight passed.")


if __name__ == "__main__":
    main()
