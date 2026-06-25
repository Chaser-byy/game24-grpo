#!/usr/bin/env bash
# Launch the direct-answer SFT/GRPO run inside a detached tmux session.

set -Eeuo pipefail

SESSION="${SESSION:-game24_direct_answer}"
ROOT="${ROOT:-/root/autodl-tmp/game24-grpo}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Run: apt-get update && apt-get install -y tmux" >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "Attach with: tmux attach -t ${SESSION}"
  exit 1
fi

ENV_ARGS=()
for name in \
  MODEL_DIR RAW_DATA_DIR TRAINING_FILE RANKED_FILE RUN_NAME RUN_ROOT DATA_FILE \
  EVAL_DATA VALIDATION_DATA SFT_EPOCHS SFT_SOLUTIONS_PER_EXAMPLE SFT_LEARNING_RATE \
  BATCH_SIZE GRAD_ACCUM SFT_MAX_LENGTH EVAL_MAX_NEW_TOKENS BEST_OF_SAMPLES \
  BEST_OF_TEMPERATURE BEST_OF_TOP_P RUN_GRPO GRPO_EPOCHS GRPO_LEARNING_RATE \
  GRPO_BETA GRPO_NUM_GENERATIONS GRPO_COMPLETION_LENGTH GRPO_EVAL_LIMIT
do
  if [[ -n "${!name:-}" ]]; then
    ENV_ARGS+=("${name}=${!name}")
  fi
done

COMMAND=(bash scripts/direct_answer_4090.sh)
if [[ "${#ENV_ARGS[@]}" -gt 0 ]]; then
  COMMAND=(env "${ENV_ARGS[@]}" "${COMMAND[@]}")
fi

tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "$(printf '%q ' "${COMMAND[@]}")"

echo "Started tmux session: ${SESSION}"
echo "Attach:  tmux attach -t ${SESSION}"
echo "Detach:  Ctrl-b then d"
echo "Status:  tmux ls"
