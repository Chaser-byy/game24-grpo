#!/usr/bin/env bash
# Build a clean Python 3.10 conda environment for A800 training.
#
# This avoids the Python 3.12 + ONNX/ONNXScript import problems seen in
# some AutoDL base images.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENV_NAME="${ENV_NAME:-game24-a800}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_MAIN_CHANNEL="${CONDA_MAIN_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
CONDA_EXTRA_CHANNEL="${CONDA_EXTRA_CHANNEL:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://mirrors.aliyun.com/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. AutoDL images usually provide /root/miniconda3/bin/conda." >&2
  exit 1
fi

echo "=== Setting up conda env: ${ENV_NAME} ==="
echo "Python: ${PYTHON_VERSION}"
if [[ -n "${CONDA_FREE_CHANNEL:-}" ]]; then
  echo "Ignoring legacy CONDA_FREE_CHANNEL=${CONDA_FREE_CHANNEL}; pkgs/free is unreliable on current mirrors."
fi
if [[ -n "${CONDA_EXTRA_CHANNEL}" ]]; then
  echo "Conda channels: ${CONDA_MAIN_CHANNEL}, ${CONDA_EXTRA_CHANNEL}"
else
  echo "Conda channels: ${CONDA_MAIN_CHANNEL}"
fi
echo "Pip index: ${PIP_INDEX_URL}"
echo "PyTorch index: ${PYTORCH_INDEX_URL}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env already exists: ${ENV_NAME}"
else
  CONDA_CHANNEL_ARGS=(--override-channels -c "${CONDA_MAIN_CHANNEL}")
  if [[ -n "${CONDA_EXTRA_CHANNEL}" ]]; then
    CONDA_CHANNEL_ARGS+=(-c "${CONDA_EXTRA_CHANNEL}")
  fi
  conda create -y -n "${ENV_NAME}" \
    "${CONDA_CHANNEL_ARGS[@]}" \
    "python=${PYTHON_VERSION}" \
    pip
fi

run_in_env() {
  conda run --no-capture-output -n "${ENV_NAME}" "$@"
}

run_in_env python -m pip install -U pip setuptools wheel \
  -i "${PIP_INDEX_URL}" \
  --trusted-host "${PIP_TRUSTED_HOST}"

run_in_env python -m pip install -U \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==2.4.0+cu121"

run_in_env python -m pip install -U \
  "numpy==1.26.4" \
  "accelerate==1.2.1" \
  "datasets==3.2.0" \
  "transformers==4.46.3" \
  "peft==0.14.0" \
  "trl==0.15.2" \
  "onnx==1.16.2" \
  "onnxscript==0.1.0" \
  "matplotlib>=3.7,<4" \
  -i "${PIP_INDEX_URL}" \
  --trusted-host "${PIP_TRUSTED_HOST}"

run_in_env python scripts/check_training_env.py

echo
echo "Environment is ready."
echo "Start training with:"
echo "  SESSION=game24_a800 CONDA_ENV=${ENV_NAME} bash scripts/run_a800_tmux.sh"
