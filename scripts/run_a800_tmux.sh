#!/usr/bin/env bash
# Launch the A800 fresh training pipeline inside a detached tmux session.

set -Eeuo pipefail

SESSION="${SESSION:-game24_a800}"
ROOT="${ROOT:-/root/autodl-tmp/game24-grpo}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install it first, for example: apt-get update && apt-get install -y tmux" >&2
  exit 1
fi

if [[ ! -d "${ROOT}" ]]; then
  echo "Repo directory does not exist: ${ROOT}" >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "Attach with: tmux attach -t ${SESSION}"
  exit 1
fi

COMMAND=(bash scripts/a800_restart_train.sh)
if [[ -n "${CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV was set to ${CONDA_ENV}, but conda was not found" >&2
    exit 1
  fi
  COMMAND=(conda run --no-capture-output -n "${CONDA_ENV}" "${COMMAND[@]}")
fi

pass_env() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    COMMAND=(env "${name}=${!name}" "${COMMAND[@]}")
  fi
}

for name in \
  MODEL_DIR RAW_DATA_DIR TRAINING_FILE RANKED_FILE RUN_NAME RUN_ROOT PRECISION SEED \
  CONDA_ENV \
  VALIDATION_SIZE ID_TEST_SIZE UNSOLVABLE_SIZE TRAIN_UNSOLVABLE_SIZE \
  SFT_EPOCHS SFT_SOLUTIONS_PER_EXAMPLE SFT_BATCH_SIZE SFT_GRAD_ACCUM SFT_MAX_LENGTH SFT_LR \
  GRPO1_EPOCHS GRPO1_NUM_GENERATIONS GRPO1_GRAD_ACCUM GRPO1_MAX_COMPLETION \
  GRPO1_TEMPERATURE GRPO1_LR GRPO1_BETA GRPO1_REWARD_MODE \
  GRPO2_EPOCHS GRPO2_NUM_GENERATIONS GRPO2_GRAD_ACCUM GRPO2_MAX_COMPLETION \
  GRPO2_TEMPERATURE GRPO2_LR GRPO2_BETA GRPO2_REWARD_MODE \
  BEST_OF_N EVAL_LIMIT EVAL_STEPS SAVE_STEPS COMPLETION_LOG_STEPS MAX_NEW_TOKENS \
  RUN_MODEL_TOT RUN_ORACLE_TOT MODEL_TOT_BEAM_SIZE MODEL_TOT_CANDIDATES \
  MODEL_TOT_BRANCH_SAMPLES MODEL_TOT_FALLBACK_CANDIDATES MODEL_TOT_EXTRA_FALLBACK_CANDIDATES
do
  pass_env "${name}"
done

tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "$(printf '%q ' "${COMMAND[@]}")"

echo "Started tmux session: ${SESSION}"
echo "Attach:  tmux attach -t ${SESSION}"
echo "Detach:  Ctrl-b then d"
echo "Log:     tail -f ${ROOT}/outputs/<RUN_NAME>/a800_restart.log"
echo "Status:  tmux ls"
