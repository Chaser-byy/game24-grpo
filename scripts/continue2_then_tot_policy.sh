#!/usr/bin/env bash
# Wait for the existing continue2 GRPO run, evaluate it, then start ToT policy SFT.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

WAIT_PATTERN="${WAIT_PATTERN:-outputs/sft_trace_grpo_continue2}"
MODEL_DIR_FOR_EVAL="${MODEL_DIR_FOR_EVAL:-outputs/sft_trace_grpo_continue2}"
DATA_FILE="${DATA_FILE:-data/processed/test_hard.jsonl}"
BEST64_OUTPUT="${BEST64_OUTPUT:-outputs/eval_continue2_best64.jsonl}"
BEST128_OUTPUT="${BEST128_OUTPUT:-outputs/eval_continue2_best128.jsonl}"
TEMPERATURE="${TEMPERATURE:-0.95}"
TOP_P="${TOP_P:-0.97}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

mkdir -p outputs
LOG_FILE="${LOG_FILE:-outputs/continue2_then_tot_policy.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== continue2 -> ToT policy queue started at $(date -Is) ==="
echo "Repo: $(pwd)"
echo "Wait pattern: ${WAIT_PATTERN}"

find_train_pid() {
  ps -ww -eo pid=,args= \
    | awk -v pattern="${WAIT_PATTERN}" \
      '$0 ~ /[t]rain_grpo.py/ && index($0, pattern) {print $1; exit}'
}

pid="$(find_train_pid || true)"
if [[ -n "${pid}" ]]; then
  echo "Waiting for current GRPO PID: ${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    echo "$(date -Is) still training..."
    sleep "${WAIT_SECONDS}"
  done
  echo "Current GRPO process ended at $(date -Is)"
else
  echo "No matching train_grpo.py process found; assuming continue2 is already finished."
fi

if [[ ! -d "${MODEL_DIR_FOR_EVAL}" ]]; then
  echo "Missing model/output directory: ${MODEL_DIR_FOR_EVAL}" >&2
  exit 1
fi
if [[ ! -f "${DATA_FILE}" ]]; then
  echo "Missing eval data: ${DATA_FILE}" >&2
  exit 1
fi

echo
echo "===== Evaluate continue2 best64 ====="
python scripts/evaluate_verified_decoding.py \
  --model "${MODEL_DIR_FOR_EVAL}" \
  --data "${DATA_FILE}" \
  --output "${BEST64_OUTPUT}" \
  --num-samples 64 \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

echo
echo "===== Evaluate continue2 best128 ====="
python scripts/evaluate_verified_decoding.py \
  --model "${MODEL_DIR_FOR_EVAL}" \
  --data "${DATA_FILE}" \
  --output "${BEST128_OUTPUT}" \
  --num-samples 128 \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

echo
echo "===== Switch to ToT policy branch ====="
git fetch origin
if git show-ref --verify --quiet refs/heads/feature/tot-policy-sft; then
  git switch feature/tot-policy-sft
else
  git switch -c feature/tot-policy-sft --track origin/feature/tot-policy-sft
fi
git pull --ff-only

echo
echo "===== Start ToT policy training ====="
bash scripts/tot_policy_4090.sh

echo "=== continue2 -> ToT policy queue finished at $(date -Is) ==="
