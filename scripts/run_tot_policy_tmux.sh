#!/usr/bin/env bash
# Launch the ToT policy SFT run inside a detached tmux session.

set -Eeuo pipefail

SESSION="${SESSION:-game24_tot_policy}"
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
  TRAIN_LIMIT EVAL_LIMIT SFT_EPOCHS LEARNING_RATE BATCH_SIZE GRAD_ACCUM \
  MAX_LENGTH CANDIDATES_PER_STATE MAX_STATES_PER_EXAMPLE MAX_ACTIONS_PER_STATE \
  BEAM_SIZE BRANCH_SAMPLES MAX_NEW_TOKENS TEMPERATURE TOP_P FALLBACK_CANDIDATES
do
  if [[ -n "${!name:-}" ]]; then
    ENV_ARGS+=("${name}=${!name}")
  fi
done

COMMAND=(bash scripts/tot_policy_4090.sh)
if [[ "${#ENV_ARGS[@]}" -gt 0 ]]; then
  COMMAND=(env "${ENV_ARGS[@]}" "${COMMAND[@]}")
fi

tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "$(printf '%q ' "${COMMAND[@]}")"

echo "Started tmux session: ${SESSION}"
echo "Attach:  tmux attach -t ${SESSION}"
echo "Detach:  Ctrl-b then d"
echo "Status:  tmux ls"
