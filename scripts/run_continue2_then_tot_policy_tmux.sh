#!/usr/bin/env bash
# Launch the continue2 evaluation -> ToT policy SFT queue in tmux.

set -Eeuo pipefail

SESSION="${SESSION:-game24_after_continue2_queue}"
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
  WAIT_PATTERN MODEL_DIR_FOR_EVAL DATA_FILE BEST64_OUTPUT BEST128_OUTPUT \
  TEMPERATURE TOP_P MAX_NEW_TOKENS WAIT_SECONDS LOG_FILE \
  MODEL_DIR RAW_DATA_DIR TRAINING_FILE RANKED_FILE RUN_NAME RUN_ROOT \
  TRAIN_LIMIT EVAL_LIMIT SFT_EPOCHS LEARNING_RATE BATCH_SIZE GRAD_ACCUM
do
  if [[ -n "${!name:-}" ]]; then
    ENV_ARGS+=("${name}=${!name}")
  fi
done

COMMAND=(bash scripts/continue2_then_tot_policy.sh)
if [[ "${#ENV_ARGS[@]}" -gt 0 ]]; then
  COMMAND=(env "${ENV_ARGS[@]}" "${COMMAND[@]}")
fi

tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "$(printf '%q ' "${COMMAND[@]}")"

echo "Started tmux session: ${SESSION}"
echo "Attach:  tmux attach -t ${SESSION}"
echo "Detach:  Ctrl-b then d"
echo "Log:     ${ROOT}/outputs/continue2_then_tot_policy.log"
