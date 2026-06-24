#!/usr/bin/env bash
# Launch the AutoDL overnight experiment inside a detached tmux session.

set -Eeuo pipefail

SESSION="${SESSION:-game24_overnight}"
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

COMMAND=(bash scripts/autodl_overnight_4090.sh)
if [[ -n "${RUN_ROOT:-}" ]]; then
  COMMAND=(env "RUN_ROOT=${RUN_ROOT}" "${COMMAND[@]}")
fi
if [[ -n "${RUN_NAME:-}" ]]; then
  COMMAND=(env "RUN_NAME=${RUN_NAME}" "${COMMAND[@]}")
fi

tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "$(printf '%q ' "${COMMAND[@]}")"

echo "Started tmux session: ${SESSION}"
echo "Attach:  tmux attach -t ${SESSION}"
echo "Detach:  Ctrl-b then d"
echo "Status:  tmux ls"
