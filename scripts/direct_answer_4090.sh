#!/usr/bin/env bash
# Train a direct-answer model with internal ToT-style traces, then evaluate answers.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/root/autodl-tmp/data}"
TRAINING_FILE="${TRAINING_FILE:-${RAW_DATA_DIR}/nlile_24_game.jsonl}"
RANKED_FILE="${RANKED_FILE:-${RAW_DATA_DIR}/game24.csv}"
RUN_NAME="${RUN_NAME:-direct_answer_4090_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/${RUN_NAME}}"
DATA_FILE="${DATA_FILE:-data/processed/train_full.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/processed/test_hard.jsonl}"
VALIDATION_DATA="${VALIDATION_DATA:-data/processed/validation_id.jsonl}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_SOLUTIONS_PER_EXAMPLE="${SFT_SOLUTIONS_PER_EXAMPLE:-4}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-1024}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
BEST_OF_SAMPLES="${BEST_OF_SAMPLES:-64 128}"
BEST_OF_TEMPERATURE="${BEST_OF_TEMPERATURE:-0.95}"
BEST_OF_TOP_P="${BEST_OF_TOP_P:-0.97}"
RUN_GRPO="${RUN_GRPO:-1}"
GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-3e-6}"
GRPO_BETA="${GRPO_BETA:-0.02}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-16}"
GRPO_COMPLETION_LENGTH="${GRPO_COMPLETION_LENGTH:-192}"
GRPO_EVAL_LIMIT="${GRPO_EVAL_LIMIT:-32}"

SFT_DIR="${RUN_ROOT}/direct_tot_sft"
SFT_MERGED="${RUN_ROOT}/direct_tot_sft_merged"
GRPO_DIR="${RUN_ROOT}/direct_tot_grpo"
EVAL_DIR="${RUN_ROOT}/eval"

mkdir -p "${RUN_ROOT}" "${EVAL_DIR}"
LOG_FILE="${RUN_ROOT}/direct_answer.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Game24 direct-answer run: ${RUN_NAME} ==="
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

evaluate_direct_model() {
  local model_dir="$1"
  local prefix="$2"

  run_step_if_missing "${EVAL_DIR}/${prefix}_greedy.summary.json" \
    "Evaluate ${prefix} greedy answer accuracy" \
    python scripts/evaluate_baseline.py \
      --model "${model_dir}" \
      --data "${EVAL_DATA}" \
      --output "${EVAL_DIR}/${prefix}_greedy.jsonl" \
      --num-samples 1 \
      --max-new-tokens "${EVAL_MAX_NEW_TOKENS}"

  for samples in ${BEST_OF_SAMPLES}; do
    run_step_if_missing "${EVAL_DIR}/${prefix}_best${samples}.summary.json" \
      "Evaluate ${prefix} best-of-${samples} answer accuracy" \
      python scripts/evaluate_verified_decoding.py \
        --model "${model_dir}" \
        --data "${EVAL_DATA}" \
        --output "${EVAL_DIR}/${prefix}_best${samples}.jsonl" \
        --num-samples "${samples}" \
        --temperature "${BEST_OF_TEMPERATURE}" \
        --top-p "${BEST_OF_TOP_P}" \
        --max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
  done
}

require_file "${MODEL_DIR}"

if [[ ! -f data/processed/train_full.jsonl || ! -f data/processed/test_hard.jsonl ]]; then
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
  echo "Reusing existing data/processed splits"
fi

require_file "${DATA_FILE}"
require_file "${EVAL_DATA}"
require_file "${VALIDATION_DATA}"

run_step "Environment check" \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'bf16=', torch.cuda.is_bf16_supported())"

run_step_if_missing "${SFT_MERGED}/config.json" \
  "Direct ToT answer SFT" \
  python scripts/train_sft_warmup.py \
    --model "${MODEL_DIR}" \
    --data "${DATA_FILE}" \
    --output "${SFT_DIR}" \
    --merged-output "${SFT_MERGED}" \
    --epochs "${SFT_EPOCHS}" \
    --precision bf16 \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --max-length "${SFT_MAX_LENGTH}" \
    --label-style direct_tot \
    --solutions-per-example "${SFT_SOLUTIONS_PER_EXAMPLE}" \
    --learning-rate "${SFT_LEARNING_RATE}"

evaluate_direct_model "${SFT_MERGED}" "sft"

if [[ "${RUN_GRPO}" == "1" ]]; then
  run_step_if_missing "${GRPO_DIR}/adapter_config.json" \
    "Accuracy GRPO from direct ToT SFT" \
    python scripts/train_grpo.py \
      --config configs/rtx4090_grpo.json \
      --model "${SFT_MERGED}" \
      --data "${DATA_FILE}" \
      --eval-data "${VALIDATION_DATA}" \
      --output "${GRPO_DIR}" \
      --reward-mode accuracy \
      --epochs "${GRPO_EPOCHS}" \
      --num-generations "${GRPO_NUM_GENERATIONS}" \
      --prompts-per-batch 1 \
      --gradient-accumulation-steps 2 \
      --max-completion-length "${GRPO_COMPLETION_LENGTH}" \
      --precision bf16 \
      --learning-rate "${GRPO_LEARNING_RATE}" \
      --beta "${GRPO_BETA}" \
      --eval-limit "${GRPO_EVAL_LIMIT}" \
      --eval-steps 100 \
      --save-steps 100 \
      --completion-log-steps 100

  evaluate_direct_model "${GRPO_DIR}" "grpo"
else
  echo "Skipping GRPO because RUN_GRPO=${RUN_GRPO}"
fi

run_step "Write report" \
  python - "${RUN_ROOT}" "${DATA_FILE}" "${EVAL_DATA}" "${RUN_GRPO}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
data = sys.argv[2]
eval_data = sys.argv[3]
run_grpo = sys.argv[4]
items = [
    ("sft_greedy", root / "eval/sft_greedy.summary.json", "accuracy"),
    ("sft_best64", root / "eval/sft_best64.summary.json", "selected_accuracy"),
    ("sft_best128", root / "eval/sft_best128.summary.json", "selected_accuracy"),
]
if run_grpo == "1":
    items.extend(
        [
            ("grpo_greedy", root / "eval/grpo_greedy.summary.json", "accuracy"),
            ("grpo_best64", root / "eval/grpo_best64.summary.json", "selected_accuracy"),
            ("grpo_best128", root / "eval/grpo_best128.summary.json", "selected_accuracy"),
        ]
    )

lines = [
    "# Direct Answer Report",
    "",
    f"- Train data: `{data}`",
    f"- Eval data: `{eval_data}`",
    "",
    "| run | metric | value | total |",
    "| --- | --- | --- | --- |",
]
for label, path, metric in items:
    if not path.exists():
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    lines.append(f"| {label} | {metric} | {summary.get(metric)} | {summary.get('total')} |")

lines.extend(
    [
        "",
        "## Artifacts",
        "",
        f"- SFT merged model: `{root / 'direct_tot_sft_merged'}`",
        f"- GRPO adapter/model: `{root / 'direct_tot_grpo'}`",
        f"- Log: `{root / 'direct_answer.log'}`",
    ]
)
(root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(root / "REPORT.md")
PY

echo
echo "Completed at: $(date -Is)"
echo "Report: ${RUN_ROOT}/REPORT.md"
