#!/usr/bin/env bash
# Train and evaluate a ToT next-operation policy on one RTX 4090.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/root/autodl-tmp/data}"
TRAINING_FILE="${TRAINING_FILE:-${RAW_DATA_DIR}/nlile_24_game.jsonl}"
RANKED_FILE="${RANKED_FILE:-${RAW_DATA_DIR}/game24.csv}"
RUN_NAME="${RUN_NAME:-tot_policy_4090_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/${RUN_NAME}}"
DATA_FILE="${DATA_FILE:-data/processed/train.jsonl}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LENGTH="${MAX_LENGTH:-768}"
CANDIDATES_PER_STATE="${CANDIDATES_PER_STATE:-4}"
MAX_STATES_PER_EXAMPLE="${MAX_STATES_PER_EXAMPLE:-16}"
MAX_ACTIONS_PER_STATE="${MAX_ACTIONS_PER_STATE:-4}"
BEAM_SIZE="${BEAM_SIZE:-5}"
BRANCH_SAMPLES="${BRANCH_SAMPLES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
FALLBACK_CANDIDATES="${FALLBACK_CANDIDATES:-0}"

SFT_DIR="${RUN_ROOT}/tot_policy_sft"
MERGED_DIR="${RUN_ROOT}/tot_policy_sft_merged"
EVAL_DIR="${RUN_ROOT}/eval"

mkdir -p "${RUN_ROOT}" "${EVAL_DIR}"
LOG_FILE="${RUN_ROOT}/tot_policy.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Game24 ToT policy SFT run: ${RUN_NAME} ==="
echo "Started at: $(date -Is)"
echo "Repo: $(pwd)"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Model: ${MODEL_DIR}"
echo "Run root: ${RUN_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required file or directory: ${path}" >&2
    exit 1
  fi
}

run_step() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "Command: $*"
  "$@"
}

run_step_if_missing() {
  local artifact="$1"
  local name="$2"
  shift 2
  if [[ -e "${artifact}" ]]; then
    echo
    echo "===== ${name} ====="
    echo "Skipping because artifact exists: ${artifact}"
  else
    run_step "${name}" "$@"
  fi
}

require_file "${MODEL_DIR}"

if [[ ! -f data/processed/train.jsonl || ! -f data/processed/test_hard.jsonl ]]; then
  require_file "${TRAINING_FILE}"
  require_file "${RANKED_FILE}"
  run_step "Prepare leakage-safe Game24 data" \
    python scripts/prepare_experiment_data.py \
      --training-file "${TRAINING_FILE}" \
      --ranked-file "${RANKED_FILE}" \
      --output-dir data/processed \
      --test-start 900 \
      --test-end 1000 \
      --validation-size 100 \
      --unsolvable-size 100 \
      --seed 42
else
  echo "Reusing existing data/processed train and hard test splits"
fi

require_file "${DATA_FILE}"
require_file data/processed/test_hard.jsonl

run_step "Environment check" \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'bf16=', torch.cuda.is_bf16_supported())"

run_step_if_missing "${MERGED_DIR}/config.json" \
  "Train ToT policy SFT" \
  python scripts/train_tot_policy_sft.py \
    --model "${MODEL_DIR}" \
    --data "${DATA_FILE}" \
    --output "${SFT_DIR}" \
    --merged-output "${MERGED_DIR}" \
    --limit "${TRAIN_LIMIT}" \
    --epochs "${SFT_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --max-length "${MAX_LENGTH}" \
    --precision bf16 \
    --candidates-per-state "${CANDIDATES_PER_STATE}" \
    --max-states-per-example "${MAX_STATES_PER_EXAMPLE}" \
    --max-actions-per-state "${MAX_ACTIONS_PER_STATE}"

EVAL_ARGS=(
  python scripts/evaluate_model_tot.py
  --model "${MERGED_DIR}"
  --data data/processed/test_hard.jsonl
  --output "${EVAL_DIR}/model_tot_test_hard.jsonl"
  --beam-size "${BEAM_SIZE}"
  --candidates-per-state "${CANDIDATES_PER_STATE}"
  --branch-samples "${BRANCH_SAMPLES}"
  --fallback-candidates "${FALLBACK_CANDIDATES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
)
if [[ "${EVAL_LIMIT}" -gt 0 ]]; then
  EVAL_ARGS+=(--limit "${EVAL_LIMIT}")
fi

run_step_if_missing "${EVAL_DIR}/model_tot_test_hard.summary.json" \
  "Evaluate ToT policy on hard test" \
  "${EVAL_ARGS[@]}"

run_step "Write report" \
  python - "${RUN_ROOT}" "${MODEL_DIR}" "${DATA_FILE}" "${MERGED_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
model = sys.argv[2]
data_file = sys.argv[3]
merged = sys.argv[4]
config_path = root / "tot_policy_sft/tot_policy_sft_run_config.json"
summary_path = root / "eval/model_tot_test_hard.summary.json"

config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

lines = [
    "# ToT Policy SFT Report",
    "",
    f"- Base model: `{model}`",
    f"- Train data: `{data_file}`",
    f"- Merged model: `{merged}`",
    f"- Policy samples: `{config.get('policy_labeled_samples')}`",
    f"- Tokenized samples: `{config.get('tokenized_examples')}`",
    f"- Hard test total: `{summary.get('total')}`",
    f"- Hard test accuracy: `{summary.get('accuracy')}`",
    f"- ToT found rate: `{summary.get('found_rate')}`",
    f"- Avg model calls: `{summary.get('avg_model_calls')}`",
    f"- Avg valid proposals: `{summary.get('avg_valid_proposals')}`",
    f"- Avg invalid proposals: `{summary.get('avg_invalid_proposals')}`",
    "",
    "## Artifacts",
    "",
    f"- Adapter: `{root / 'tot_policy_sft'}`",
    f"- Evaluation JSONL: `{root / 'eval/model_tot_test_hard.jsonl'}`",
    f"- Summary JSON: `{summary_path}`",
    f"- Log: `{root / 'tot_policy.log'}`",
]
(root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(root / "REPORT.md")
PY

echo
echo "Completed at: $(date -Is)"
echo "Report: ${RUN_ROOT}/REPORT.md"
