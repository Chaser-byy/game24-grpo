#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-game24-vllm}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Run this script inside the AutoDL image that provides conda."
    exit 2
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env ${ENV_NAME} already exists. Reusing it."
else
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel -i "${PIP_INDEX_URL}"
python -m pip install "torch==2.4.0" --index-url "${TORCH_INDEX_URL}"

# Keep NumPy below 2.x. The TinyZero/vLLM 0.6.3 stack is built against the 1.x ABI.
python -m pip install --force-reinstall \
    "numpy==1.26.4" \
    "pandas==2.2.2" \
    "pyarrow==16.1.0" \
    -i "${PIP_INDEX_URL}"

python -m pip install -r requirements-autodl.txt -i "${PIP_INDEX_URL}"
python -m pip install -e . --no-build-isolation -i "${PIP_INDEX_URL}"

python scripts/check_autodl_env.py --strict-vllm

echo
echo "Environment ${ENV_NAME} is ready. Activate it with:"
echo "  conda activate ${ENV_NAME}"
