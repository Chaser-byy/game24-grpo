#!/usr/bin/env bash
# Continue from a format-correct model and optimize only for solving 24-point tasks.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SOURCE_MODEL="${SOURCE_MODEL:-outputs/overnight_4090_20260624_014838/grpo_b_explore_g16}"
RUN_NAME="${RUN_NAME:-accuracy_first_4090_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/${RUN_NAME}}"
PRECISION="${PRECISION:-bf16}"

mkdir -p "${RUN_ROOT}"
LOG_FILE="${RUN_ROOT}/accuracy_first.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Game24 accuracy-first run: ${RUN_NAME} ==="
echo "Started at: $(date -Is)"
echo "Source model: ${SOURCE_MODEL}"
echo "Run root: ${RUN_ROOT}"

run_step() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "Command: $*"
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required file or directory: ${path}" >&2
    exit 1
  fi
}

require_file "${SOURCE_MODEL}"
require_file data/processed/train_full.jsonl
require_file data/processed/test_hard.jsonl

BASE_FOR_REFRESH="${SOURCE_MODEL}"
if [[ -f "${SOURCE_MODEL}/adapter_config.json" ]]; then
  BASE_FOR_REFRESH="${RUN_ROOT}/source_merged"
  if [[ ! -f "${BASE_FOR_REFRESH}/config.json" ]]; then
    run_step "Merge source adapter" \
      python scripts/merge_adapter.py \
        --adapter "${SOURCE_MODEL}" \
        --output "${BASE_FOR_REFRESH}" \
        --dtype "${PRECISION}"
  else
    echo "Reusing merged source: ${BASE_FOR_REFRESH}"
  fi
fi

SFT_DIR="${RUN_ROOT}/sft_accuracy_refresh"
SFT_MERGED="${RUN_ROOT}/sft_accuracy_refresh_merged"
if [[ ! -f "${SFT_MERGED}/config.json" ]]; then
  run_step "Solver SFT refresh on solvable train_full only" \
    python scripts/train_sft_warmup.py \
      --model "${BASE_FOR_REFRESH}" \
      --data data/processed/train_full.jsonl \
      --output "${SFT_DIR}" \
      --merged-output "${SFT_MERGED}" \
      --epochs 2 \
      --precision "${PRECISION}" \
      --batch-size 4 \
      --gradient-accumulation-steps 4 \
      --learning-rate 2e-5
else
  echo "Reusing SFT refresh model: ${SFT_MERGED}"
fi

GRPO_ACCURACY="${RUN_ROOT}/grpo_accuracy_reward"
if [[ ! -f "${GRPO_ACCURACY}/adapter_config.json" ]]; then
  run_step "GRPO accuracy mode: no format reward, no unsolvable rows" \
    python scripts/train_grpo.py \
      --config configs/rtx4090_grpo.json \
      --model "${SFT_MERGED}" \
      --data data/processed/train_full.jsonl \
      --eval-data data/processed/validation_id.jsonl \
      --output "${GRPO_ACCURACY}" \
      --reward-mode accuracy \
      --epochs 2 \
      --num-generations 16 \
      --prompts-per-batch 1 \
      --gradient-accumulation-steps 2 \
      --max-completion-length 128 \
      --temperature 1.15 \
      --beta 0.02 \
      --learning-rate 8e-6 \
      --eval-limit 32 \
      --eval-steps 100 \
      --save-steps 100 \
      --completion-log-steps 100
else
  echo "Reusing GRPO accuracy model: ${GRPO_ACCURACY}"
fi

run_step "Evaluate SFT refresh hard accuracy@1" \
  python scripts/evaluate_baseline.py \
    --model "${SFT_MERGED}" \
    --data data/processed/test_hard.jsonl \
    --output "${RUN_ROOT}/sft_accuracy_refresh_hard.jsonl" \
    --num-samples 1 \
    --max-new-tokens 192

run_step "Evaluate GRPO accuracy hard accuracy@1" \
  python scripts/evaluate_baseline.py \
    --model "${GRPO_ACCURACY}" \
    --data data/processed/test_hard.jsonl \
    --output "${RUN_ROOT}/grpo_accuracy_hard.jsonl" \
    --num-samples 1 \
    --max-new-tokens 192

run_step "Evaluate GRPO accuracy pass@32" \
  python scripts/evaluate_baseline.py \
    --model "${GRPO_ACCURACY}" \
    --data data/processed/test_hard.jsonl \
    --output "${RUN_ROOT}/grpo_accuracy_hard_pass32.jsonl" \
    --num-samples 32 \
    --sample \
    --temperature 0.9 \
    --top-p 0.95 \
    --max-new-tokens 192

run_step "Write compact accuracy report" \
  python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
items = [
    ("sft_accuracy_refresh", root / "sft_accuracy_refresh_hard.summary.json"),
    ("grpo_accuracy", root / "grpo_accuracy_hard.summary.json"),
    ("grpo_accuracy_pass32", root / "grpo_accuracy_hard_pass32.summary.json"),
]
lines = ["# Accuracy-First Game24 Summary", ""]
for label, path in items:
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    pass_keys = sorted(k for k in data if k.startswith("pass_at_") and not k.endswith("_ci95"))
    pass_text = ", ".join(f"{key}={data[key]}" for key in pass_keys)
    lines.append(
        f"- {label}: acc@1={data.get('accuracy_at_1')}, correct={data.get('correct')}/"
        f"{data.get('total')}, {pass_text}, strict={data.get('strict_format_rate')}, "
        f"legal={data.get('legal_number_rate')}"
    )
report = "\n".join(lines) + "\n"
(root / "REPORT.md").write_text(report, encoding="utf-8")
print(report)
PY

echo "Finished at: $(date -Is)"
echo "Report: ${RUN_ROOT}/REPORT.md"
