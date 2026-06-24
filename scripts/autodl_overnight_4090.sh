#!/usr/bin/env bash
# Run an unattended RTX 4090 overnight experiment:
# data prep -> unsolvable SFT warmup -> two GRPO variants -> evaluations -> summary.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/root/autodl-tmp/data}"
TRAINING_FILE="${TRAINING_FILE:-${RAW_DATA_DIR}/nlile_24_game.jsonl}"
RANKED_FILE="${RANKED_FILE:-${RAW_DATA_DIR}/game24.csv}"
RUN_NAME="${RUN_NAME:-overnight_4090_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/${RUN_NAME}}"
TRAIN_UNSOLVABLE_SIZE="${TRAIN_UNSOLVABLE_SIZE:-200}"
EVAL_LIMIT="${EVAL_LIMIT:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"

mkdir -p "${RUN_ROOT}"
LOG_FILE="${RUN_ROOT}/overnight.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Game24 overnight run: ${RUN_NAME} ==="
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

run_step_optional() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "Command: $*"
  if ! "$@"; then
    echo "WARNING: optional step failed, continuing: ${name}" >&2
  fi
}

evaluate_all() {
  local model_dir="$1"
  local eval_dir="$2"
  mkdir -p "${eval_dir}"

  run_step_if_missing "${eval_dir}/hard_eval.summary.json" \
    "Evaluate hard accuracy@1: ${eval_dir}" \
    python scripts/evaluate_baseline.py \
      --model "${model_dir}" \
      --data data/processed/test_hard.jsonl \
      --output "${eval_dir}/hard_eval.jsonl" \
      --num-samples 1 \
      --max-new-tokens "${MAX_NEW_TOKENS}"

  run_step_if_missing "${eval_dir}/hard_eval_pass16.summary.json" \
    "Evaluate hard pass@16: ${eval_dir}" \
    python scripts/evaluate_baseline.py \
      --model "${model_dir}" \
      --data data/processed/test_hard.jsonl \
      --output "${eval_dir}/hard_eval_pass16.jsonl" \
      --num-samples 16 \
      --sample \
      --temperature 0.9 \
      --top-p 0.95 \
      --max-new-tokens "${MAX_NEW_TOKENS}"

  run_step_if_missing "${eval_dir}/unsolvable_eval.summary.json" \
    "Evaluate unsolvable: ${eval_dir}" \
    python scripts/evaluate_baseline.py \
      --model "${model_dir}" \
      --data data/processed/test_unsolvable.jsonl \
      --output "${eval_dir}/unsolvable_eval.jsonl" \
      --num-samples 1 \
      --max-new-tokens "${MAX_NEW_TOKENS}"
}

plot_if_possible() {
  local train_metrics="$1"
  local grpo_summary="$2"
  local plot_dir="$3"
  if [[ -f "${train_metrics}" && -f "${grpo_summary}" && -f outputs/baseline_hard.summary.json ]]; then
    run_step_optional "Plot ${plot_dir}" \
      python scripts/plot_results.py \
        --train-metrics "${train_metrics}" \
        --baseline-summary outputs/baseline_hard.summary.json \
        --grpo-summary "${grpo_summary}" \
        --output-dir "${plot_dir}"
  else
    echo "Skipping plot: missing metrics, GRPO summary, or baseline summary"
  fi
}

require_file "${MODEL_DIR}"
require_file "${TRAINING_FILE}"
require_file "${RANKED_FILE}"

run_step "Environment check" \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'bf16=', torch.cuda.is_bf16_supported())"

run_step "Validate 4090 GRPO config" \
  python -m json.tool configs/rtx4090_grpo.json

run_step "Prepare leakage-safe data with auxiliary unsolvable train split" \
  python scripts/prepare_experiment_data.py \
    --training-file "${TRAINING_FILE}" \
    --ranked-file "${RANKED_FILE}" \
    --output-dir data/processed \
    --test-start 900 \
    --test-end 1000 \
    --validation-size 100 \
    --unsolvable-size 100 \
    --train-unsolvable-size "${TRAIN_UNSOLVABLE_SIZE}" \
    --seed 42

if [[ ! -f outputs/baseline_hard.summary.json ]]; then
  run_step "Baseline hard evaluation" \
    python scripts/evaluate_baseline.py \
      --model "${MODEL_DIR}" \
      --data data/processed/test_hard.jsonl \
      --output outputs/baseline_hard.jsonl \
      --num-samples 1 \
      --max-new-tokens "${MAX_NEW_TOKENS}"
else
  echo "Reusing existing outputs/baseline_hard.summary.json"
fi

SFT_DIR="${RUN_ROOT}/sft_warmup_unsolv"
SFT_MERGED="${RUN_ROOT}/sft_warmup_unsolv_merged"
run_step_if_missing "${SFT_MERGED}/config.json" \
  "SFT warmup with solvable + auxiliary unsolvable labels" \
  python scripts/train_sft_warmup.py \
    --model "${MODEL_DIR}" \
    --data data/processed/train.jsonl data/processed/train_unsolvable.jsonl \
    --include-unsolvable \
    --output "${SFT_DIR}" \
    --merged-output "${SFT_MERGED}" \
    --epochs "${SFT_EPOCHS}" \
    --precision bf16 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5

evaluate_all "${SFT_MERGED}" "${RUN_ROOT}/eval_sft"

GRPO_A="${RUN_ROOT}/grpo_a_stable_g12"
run_step_if_missing "${GRPO_A}/adapter_config.json" \
  "GRPO variant A: stable g12, two epochs" \
  python scripts/train_grpo.py \
    --config configs/rtx4090_grpo.json \
    --model "${SFT_MERGED}" \
    --data data/processed/train.jsonl data/processed/train_unsolvable.jsonl \
    --include-unsolvable \
    --eval-data data/processed/validation_id.jsonl \
    --output "${GRPO_A}" \
    --epochs 2 \
    --num-generations 12 \
    --prompts-per-batch 1 \
    --gradient-accumulation-steps 2 \
    --max-completion-length 128 \
    --eval-limit "${EVAL_LIMIT}" \
    --eval-steps 100 \
    --save-steps 100 \
    --completion-log-steps 100

evaluate_all "${GRPO_A}" "${GRPO_A}/eval"
plot_if_possible \
  "${GRPO_A}/train_metrics.jsonl" \
  "${GRPO_A}/eval/hard_eval.summary.json" \
  "${GRPO_A}/plots"

GRPO_B="${RUN_ROOT}/grpo_b_explore_g16"
run_step_if_missing "${GRPO_B}/adapter_config.json" \
  "GRPO variant B: higher exploration g16, one epoch" \
  python scripts/train_grpo.py \
    --config configs/rtx4090_grpo.json \
    --model "${SFT_MERGED}" \
    --data data/processed/train.jsonl data/processed/train_unsolvable.jsonl \
    --include-unsolvable \
    --eval-data data/processed/validation_id.jsonl \
    --output "${GRPO_B}" \
    --epochs 1 \
    --num-generations 16 \
    --prompts-per-batch 1 \
    --gradient-accumulation-steps 2 \
    --max-completion-length 128 \
    --temperature 1.2 \
    --beta 0.03 \
    --learning-rate 5e-6 \
    --eval-limit "${EVAL_LIMIT}" \
    --eval-steps 100 \
    --save-steps 100 \
    --completion-log-steps 100

evaluate_all "${GRPO_B}" "${GRPO_B}/eval"
plot_if_possible \
  "${GRPO_B}/train_metrics.jsonl" \
  "${GRPO_B}/eval/hard_eval.summary.json" \
  "${GRPO_B}/plots"

run_step "Aggregate summary table" \
  python scripts/summarize_experiments.py \
    --summary baseline_hard=outputs/baseline_hard.summary.json \
    --summary sft_hard="${RUN_ROOT}/eval_sft/hard_eval.summary.json" \
    --summary sft_unsolvable="${RUN_ROOT}/eval_sft/unsolvable_eval.summary.json" \
    --summary grpo_a_hard="${GRPO_A}/eval/hard_eval.summary.json" \
    --summary grpo_a_unsolvable="${GRPO_A}/eval/unsolvable_eval.summary.json" \
    --summary grpo_b_hard="${GRPO_B}/eval/hard_eval.summary.json" \
    --summary grpo_b_unsolvable="${GRPO_B}/eval/unsolvable_eval.summary.json" \
    --output "${RUN_ROOT}/summary.csv"

run_step "Write compact report" \
  python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
items = [
    ("baseline_hard", Path("outputs/baseline_hard.summary.json")),
    ("sft_hard", root / "eval_sft/hard_eval.summary.json"),
    ("sft_pass16", root / "eval_sft/hard_eval_pass16.summary.json"),
    ("sft_unsolvable", root / "eval_sft/unsolvable_eval.summary.json"),
    ("grpo_a_hard", root / "grpo_a_stable_g12/eval/hard_eval.summary.json"),
    ("grpo_a_pass16", root / "grpo_a_stable_g12/eval/hard_eval_pass16.summary.json"),
    ("grpo_a_unsolvable", root / "grpo_a_stable_g12/eval/unsolvable_eval.summary.json"),
    ("grpo_b_hard", root / "grpo_b_explore_g16/eval/hard_eval.summary.json"),
    ("grpo_b_pass16", root / "grpo_b_explore_g16/eval/hard_eval_pass16.summary.json"),
    ("grpo_b_unsolvable", root / "grpo_b_explore_g16/eval/unsolvable_eval.summary.json"),
]
lines = ["# Overnight Game24 Summary", ""]
for label, path in items:
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    pass_keys = sorted(k for k in data if k.startswith("pass_at_") and not k.endswith("_ci95"))
    pass_text = ", ".join(f"{k}={data[k]}" for k in pass_keys)
    lines.append(
        f"- {label}: acc@1={data.get('accuracy_at_1')}, correct={data.get('correct')}/"
        f"{data.get('total')}, {pass_text}, strict={data.get('strict_format_rate')}, "
        f"legal={data.get('legal_number_rate')}, abstain={data.get('correct_abstention_rate')}, "
        f"false_claim={data.get('false_claim_rate')}"
    )
report = "\n".join(lines) + "\n"
(root / "REPORT.md").write_text(report, encoding="utf-8")
print(report)
PY

echo
echo "Finished at: $(date -Is)"
echo "All outputs are under: ${RUN_ROOT}"
echo "Compact report: ${RUN_ROOT}/REPORT.md"
