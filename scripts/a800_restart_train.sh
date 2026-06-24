#!/usr/bin/env bash
# Fresh A800 run for accuracy-first Game24 experiments.
#
# Pipeline:
#   data prep -> trajectory SFT -> GRPO accuracy -> merge -> GRPO refine
#   -> single / best-of-N / model-ToT / oracle-ToT evaluations -> REPORT.md

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/root/autodl-tmp/data}"
TRAINING_FILE="${TRAINING_FILE:-${RAW_DATA_DIR}/nlile_24_game.jsonl}"

if [[ -z "${RANKED_FILE:-}" ]]; then
  for candidate in \
    "${RAW_DATA_DIR}/game24.csv" \
    "${RAW_DATA_DIR}/24.csv" \
    "${RAW_DATA_DIR}/game-of-24.csv"
  do
    if [[ -f "${candidate}" ]]; then
      RANKED_FILE="${candidate}"
      break
    fi
  done
fi
RANKED_FILE="${RANKED_FILE:-${RAW_DATA_DIR}/game24.csv}"

RUN_NAME="${RUN_NAME:-a800_restart_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/${RUN_NAME}}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"

VALIDATION_SIZE="${VALIDATION_SIZE:-100}"
ID_TEST_SIZE="${ID_TEST_SIZE:-100}"
UNSOLVABLE_SIZE="${UNSOLVABLE_SIZE:-100}"
TRAIN_UNSOLVABLE_SIZE="${TRAIN_UNSOLVABLE_SIZE:-0}"

SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_SOLUTIONS_PER_EXAMPLE="${SFT_SOLUTIONS_PER_EXAMPLE:-2}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-1}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-768}"
SFT_LR="${SFT_LR:-2e-5}"

GRPO1_EPOCHS="${GRPO1_EPOCHS:-2}"
GRPO1_NUM_GENERATIONS="${GRPO1_NUM_GENERATIONS:-24}"
GRPO1_GRAD_ACCUM="${GRPO1_GRAD_ACCUM:-1}"
GRPO1_MAX_COMPLETION="${GRPO1_MAX_COMPLETION:-128}"
GRPO1_TEMPERATURE="${GRPO1_TEMPERATURE:-1.15}"
GRPO1_LR="${GRPO1_LR:-8e-6}"
GRPO1_BETA="${GRPO1_BETA:-0.02}"
GRPO1_REWARD_MODE="${GRPO1_REWARD_MODE:-accuracy}"

GRPO2_EPOCHS="${GRPO2_EPOCHS:-1}"
GRPO2_NUM_GENERATIONS="${GRPO2_NUM_GENERATIONS:-32}"
GRPO2_GRAD_ACCUM="${GRPO2_GRAD_ACCUM:-1}"
GRPO2_MAX_COMPLETION="${GRPO2_MAX_COMPLETION:-128}"
GRPO2_TEMPERATURE="${GRPO2_TEMPERATURE:-1.20}"
GRPO2_LR="${GRPO2_LR:-4e-6}"
GRPO2_BETA="${GRPO2_BETA:-0.015}"
GRPO2_REWARD_MODE="${GRPO2_REWARD_MODE:-accuracy}"

EVAL_LIMIT="${EVAL_LIMIT:-64}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
COMPLETION_LOG_STEPS="${COMPLETION_LOG_STEPS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
BEST_OF_N="${BEST_OF_N:-64}"

MODEL_TOT_BEAM_SIZE="${MODEL_TOT_BEAM_SIZE:-6}"
MODEL_TOT_CANDIDATES="${MODEL_TOT_CANDIDATES:-5}"
MODEL_TOT_BRANCH_SAMPLES="${MODEL_TOT_BRANCH_SAMPLES:-2}"
MODEL_TOT_FALLBACK_CANDIDATES="${MODEL_TOT_FALLBACK_CANDIDATES:-0}"
MODEL_TOT_EXTRA_FALLBACK_CANDIDATES="${MODEL_TOT_EXTRA_FALLBACK_CANDIDATES:-4}"
RUN_MODEL_TOT="${RUN_MODEL_TOT:-1}"
RUN_ORACLE_TOT="${RUN_ORACLE_TOT:-1}"

mkdir -p "${RUN_ROOT}"
LOG_FILE="${RUN_ROOT}/a800_restart.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Game24 A800 fresh accuracy run: ${RUN_NAME} ==="
echo "Started at: $(date -Is)"
echo "Repo: $(pwd)"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Model: ${MODEL_DIR}"
echo "Training data: ${TRAINING_FILE}"
echo "Ranked data: ${RANKED_FILE}"
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

evaluate_single() {
  local model_dir="$1"
  local data_path="$2"
  local output_prefix="$3"
  run_step_if_missing "${output_prefix}.summary.json" \
    "Single generation: ${output_prefix}" \
    python scripts/evaluate_baseline.py \
      --model "${model_dir}" \
      --data "${data_path}" \
      --output "${output_prefix}.jsonl" \
      --num-samples 1 \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}"
}

evaluate_best_of() {
  local model_dir="$1"
  local data_path="$2"
  local output_prefix="$3"
  run_step_if_missing "${output_prefix}.summary.json" \
    "Verified best-of-${BEST_OF_N}: ${output_prefix}" \
    python scripts/evaluate_verified_decoding.py \
      --model "${model_dir}" \
      --data "${data_path}" \
      --output "${output_prefix}.jsonl" \
      --num-samples "${BEST_OF_N}" \
      --temperature 0.9 \
      --top-p 0.95 \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}"
}

evaluate_unsolvable() {
  local model_dir="$1"
  local output_prefix="$2"
  run_step_if_missing "${output_prefix}.summary.json" \
    "Unsolvable false-positive check: ${output_prefix}" \
    python scripts/evaluate_baseline.py \
      --model "${model_dir}" \
      --data data/processed/test_unsolvable.jsonl \
      --output "${output_prefix}.jsonl" \
      --num-samples 1 \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}"
}

evaluate_model_pack() {
  local label="$1"
  local model_dir="$2"
  local eval_dir="$3"
  mkdir -p "${eval_dir}"
  evaluate_single "${model_dir}" data/processed/test_hard.jsonl "${eval_dir}/${label}_hard_single"
  evaluate_best_of "${model_dir}" data/processed/test_hard.jsonl "${eval_dir}/${label}_hard_best_of_${BEST_OF_N}"
  if [[ -f data/processed/test_id.jsonl ]]; then
    evaluate_single "${model_dir}" data/processed/test_id.jsonl "${eval_dir}/${label}_id_single"
  fi
  evaluate_unsolvable "${model_dir}" "${eval_dir}/${label}_unsolvable_single"
}

require_file "${MODEL_DIR}"
require_file "${TRAINING_FILE}"
require_file "${RANKED_FILE}"

run_step "Environment check" \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'bf16=', torch.cuda.is_bf16_supported()); print('mem_gb=', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))"

run_step "NVIDIA status" nvidia-smi

run_step "Validate GRPO config JSON" python -m json.tool configs/rtx4090_grpo.json

run_step "Prepare leakage-safe experiment data" \
  python scripts/prepare_experiment_data.py \
    --training-file "${TRAINING_FILE}" \
    --ranked-file "${RANKED_FILE}" \
    --output-dir data/processed \
    --test-start 900 \
    --test-end 1000 \
    --validation-size "${VALIDATION_SIZE}" \
    --id-test-size "${ID_TEST_SIZE}" \
    --unsolvable-size "${UNSOLVABLE_SIZE}" \
    --train-unsolvable-size "${TRAIN_UNSOLVABLE_SIZE}" \
    --seed "${SEED}"

evaluate_single "${MODEL_DIR}" data/processed/test_hard.jsonl "${RUN_ROOT}/baseline_hard_single"

SFT_DIR="${RUN_ROOT}/sft_trajectory"
SFT_MERGED="${RUN_ROOT}/sft_trajectory_merged"
run_step_if_missing "${SFT_MERGED}/config.json" \
  "Trajectory SFT from original model" \
  python scripts/train_sft_warmup.py \
    --model "${MODEL_DIR}" \
    --data data/processed/train_full.jsonl \
    --output "${SFT_DIR}" \
    --merged-output "${SFT_MERGED}" \
    --epochs "${SFT_EPOCHS}" \
    --precision "${PRECISION}" \
    --label-style trajectory \
    --solutions-per-example "${SFT_SOLUTIONS_PER_EXAMPLE}" \
    --batch-size "${SFT_BATCH_SIZE}" \
    --gradient-accumulation-steps "${SFT_GRAD_ACCUM}" \
    --max-length "${SFT_MAX_LENGTH}" \
    --learning-rate "${SFT_LR}" \
    --logging-steps 5 \
    --seed "${SEED}"

evaluate_model_pack "sft" "${SFT_MERGED}" "${RUN_ROOT}/eval_sft"

GRPO1="${RUN_ROOT}/grpo1_${GRPO1_REWARD_MODE}_g${GRPO1_NUM_GENERATIONS}"
run_step_if_missing "${GRPO1}/adapter_config.json" \
  "GRPO phase 1: ${GRPO1_REWARD_MODE}, g${GRPO1_NUM_GENERATIONS}" \
  python scripts/train_grpo.py \
    --config configs/rtx4090_grpo.json \
    --model "${SFT_MERGED}" \
    --data data/processed/train_full.jsonl \
    --eval-data data/processed/validation_id.jsonl \
    --output "${GRPO1}" \
    --reward-mode "${GRPO1_REWARD_MODE}" \
    --epochs "${GRPO1_EPOCHS}" \
    --num-generations "${GRPO1_NUM_GENERATIONS}" \
    --prompts-per-batch 1 \
    --gradient-accumulation-steps "${GRPO1_GRAD_ACCUM}" \
    --max-completion-length "${GRPO1_MAX_COMPLETION}" \
    --temperature "${GRPO1_TEMPERATURE}" \
    --beta "${GRPO1_BETA}" \
    --learning-rate "${GRPO1_LR}" \
    --precision "${PRECISION}" \
    --eval-limit "${EVAL_LIMIT}" \
    --eval-steps "${EVAL_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --completion-log-steps "${COMPLETION_LOG_STEPS}" \
    --seed "${SEED}"

evaluate_model_pack "grpo1" "${GRPO1}" "${GRPO1}/eval"

GRPO1_MERGED="${RUN_ROOT}/grpo1_merged"
run_step_if_missing "${GRPO1_MERGED}/config.json" \
  "Merge GRPO phase 1 adapter" \
  python scripts/merge_adapter.py \
    --adapter "${GRPO1}" \
    --output "${GRPO1_MERGED}" \
    --dtype "${PRECISION}"

GRPO2="${RUN_ROOT}/grpo2_${GRPO2_REWARD_MODE}_g${GRPO2_NUM_GENERATIONS}"
run_step_if_missing "${GRPO2}/adapter_config.json" \
  "GRPO phase 2: ${GRPO2_REWARD_MODE}, g${GRPO2_NUM_GENERATIONS}" \
  python scripts/train_grpo.py \
    --config configs/rtx4090_grpo.json \
    --model "${GRPO1_MERGED}" \
    --data data/processed/train_full.jsonl \
    --eval-data data/processed/validation_id.jsonl \
    --output "${GRPO2}" \
    --reward-mode "${GRPO2_REWARD_MODE}" \
    --epochs "${GRPO2_EPOCHS}" \
    --num-generations "${GRPO2_NUM_GENERATIONS}" \
    --prompts-per-batch 1 \
    --gradient-accumulation-steps "${GRPO2_GRAD_ACCUM}" \
    --max-completion-length "${GRPO2_MAX_COMPLETION}" \
    --temperature "${GRPO2_TEMPERATURE}" \
    --beta "${GRPO2_BETA}" \
    --learning-rate "${GRPO2_LR}" \
    --precision "${PRECISION}" \
    --eval-limit "${EVAL_LIMIT}" \
    --eval-steps "${EVAL_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --completion-log-steps "${COMPLETION_LOG_STEPS}" \
    --seed "${SEED}"

evaluate_model_pack "grpo2" "${GRPO2}" "${GRPO2}/eval"

if [[ "${RUN_MODEL_TOT}" == "1" ]]; then
  run_step_if_missing "${GRPO2}/eval/model_tot.summary.json" \
    "Model-guided ToT on final GRPO model" \
    python scripts/evaluate_model_tot.py \
      --model "${GRPO2}" \
      --data data/processed/test_hard.jsonl \
      --output "${GRPO2}/eval/model_tot.jsonl" \
      --beam-size "${MODEL_TOT_BEAM_SIZE}" \
      --candidates-per-state "${MODEL_TOT_CANDIDATES}" \
      --branch-samples "${MODEL_TOT_BRANCH_SAMPLES}" \
      --fallback-candidates "${MODEL_TOT_FALLBACK_CANDIDATES}" \
      --temperature 0.7 \
      --top-p 0.9 \
      --seed "${SEED}"

  run_step_if_missing "${GRPO2}/eval/model_tot_fallback.summary.json" \
    "Model-guided ToT plus heuristic fallback on final GRPO model" \
    python scripts/evaluate_model_tot.py \
      --model "${GRPO2}" \
      --data data/processed/test_hard.jsonl \
      --output "${GRPO2}/eval/model_tot_fallback.jsonl" \
      --beam-size "${MODEL_TOT_BEAM_SIZE}" \
      --candidates-per-state "${MODEL_TOT_CANDIDATES}" \
      --branch-samples "${MODEL_TOT_BRANCH_SAMPLES}" \
      --fallback-candidates "${MODEL_TOT_EXTRA_FALLBACK_CANDIDATES}" \
      --temperature 0.7 \
      --top-p 0.9 \
      --seed "${SEED}"
fi

if [[ "${RUN_ORACLE_TOT}" == "1" ]]; then
  run_step_if_missing "${RUN_ROOT}/oracle_tot_hard.summary.json" \
    "Programmatic exhaustive ToT oracle on hard test" \
    python scripts/evaluate_tot.py \
      --data data/processed/test_hard.jsonl \
      --output "${RUN_ROOT}/oracle_tot_hard.jsonl"
fi

echo "${GRPO2}" > "${RUN_ROOT}/latest_model.txt"

run_step "Write A800 report" \
  python - "${RUN_ROOT}" "${BEST_OF_N}" "${GRPO1}" "${GRPO2}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
best_n = sys.argv[2]
grpo1 = Path(sys.argv[3])
grpo2 = Path(sys.argv[4])

items = [
    ("baseline_single", root / "baseline_hard_single.summary.json", "accuracy_at_1"),
    ("sft_single", root / "eval_sft/sft_hard_single.summary.json", "accuracy_at_1"),
    (f"sft_best_of_{best_n}", root / f"eval_sft/sft_hard_best_of_{best_n}.summary.json", "selected_accuracy"),
    ("sft_unsolvable_false_claim", root / "eval_sft/sft_unsolvable_single.summary.json", "false_claim_rate"),
    ("grpo1_single", grpo1 / "eval/grpo1_hard_single.summary.json", "accuracy_at_1"),
    (f"grpo1_best_of_{best_n}", grpo1 / f"eval/grpo1_hard_best_of_{best_n}.summary.json", "selected_accuracy"),
    ("grpo1_unsolvable_false_claim", grpo1 / "eval/grpo1_unsolvable_single.summary.json", "false_claim_rate"),
    ("grpo2_single", grpo2 / "eval/grpo2_hard_single.summary.json", "accuracy_at_1"),
    (f"grpo2_best_of_{best_n}", grpo2 / f"eval/grpo2_hard_best_of_{best_n}.summary.json", "selected_accuracy"),
    ("grpo2_unsolvable_false_claim", grpo2 / "eval/grpo2_unsolvable_single.summary.json", "false_claim_rate"),
    ("grpo2_model_tot", grpo2 / "eval/model_tot.summary.json", "accuracy"),
    ("grpo2_model_tot_fallback", grpo2 / "eval/model_tot_fallback.summary.json", "accuracy"),
    ("oracle_tot_hard", root / "oracle_tot_hard.summary.json", "accuracy"),
]

rows = []
for label, path, metric in items:
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "label": label,
            "metric": metric,
            "value": data.get(metric),
            "total": data.get("total"),
            "model": data.get("model") or data.get("method"),
            "path": str(path),
        }
    )

csv_lines = ["label,metric,value,total,path"]
for row in rows:
    csv_lines.append(
        f"{row['label']},{row['metric']},{row['value']},{row['total']},{row['path']}"
    )
(root / "summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
(root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

report = ["# A800 Game24 Restart Summary", ""]
for row in rows:
    report.append(
        f"- {row['label']}: {row['metric']}={row['value']} "
        f"(total={row['total']})"
    )
report.extend(
    [
        "",
        "## Artifacts",
        f"- SFT merged model: {root / 'sft_trajectory_merged'}",
        f"- GRPO phase 1 adapter: {grpo1}",
        f"- Final GRPO adapter: {grpo2}",
        f"- Latest model pointer: {root / 'latest_model.txt'}",
        "",
        "Note: oracle_tot_hard is a programmatic search upper bound, not model-only accuracy.",
    ]
)
(root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))
PY

echo "Finished at: $(date -Is)"
echo "Report: ${RUN_ROOT}/REPORT.md"
echo "Latest model: $(cat "${RUN_ROOT}/latest_model.txt")"
