#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${COUNTDOWN_ACTOR_CKPT:-}" ]]; then
    echo "Set COUNTDOWN_ACTOR_CKPT to the Countdown GRPO actor checkpoint directory first."
    echo "Then this script will continue GRPO on Game of 24 with MODEL_PATH=COUNTDOWN_ACTOR_CKPT."
    exit 2
fi

MODEL_PATH=${COUNTDOWN_ACTOR_CKPT} \
EXPERIMENT_NAME=${EXPERIMENT_NAME:-countdown_then_game24_grpo} \
bash scripts/train_game24_grpo.sh "$@"
