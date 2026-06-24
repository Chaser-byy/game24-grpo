#!/usr/bin/env bash
# Evaluate one model with single generation, verified best-of-N, and model-guided ToT.

set -Eeuo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 MODEL DATA OUTPUT_DIR [NUM_SAMPLES]" >&2
  exit 1
fi

MODEL="$1"
DATA="$2"
OUTPUT_DIR="$3"
NUM_SAMPLES="${4:-32}"

mkdir -p "${OUTPUT_DIR}"

python scripts/evaluate_baseline.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --output "${OUTPUT_DIR}/single.jsonl" \
  --num-samples 1 \
  --max-new-tokens 192

python scripts/evaluate_verified_decoding.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --output "${OUTPUT_DIR}/best_of_${NUM_SAMPLES}.jsonl" \
  --num-samples "${NUM_SAMPLES}" \
  --temperature 0.9 \
  --top-p 0.95 \
  --max-new-tokens 192

python scripts/evaluate_model_tot.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --output "${OUTPUT_DIR}/model_tot.jsonl" \
  --beam-size 5 \
  --candidates-per-state 4 \
  --branch-samples 2 \
  --temperature 0.7 \
  --top-p 0.9

python - "${OUTPUT_DIR}" "${NUM_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
num_samples = sys.argv[2]
items = [
    ("single", root / "single.summary.json", "accuracy_at_1"),
    (f"best_of_{num_samples}", root / f"best_of_{num_samples}.summary.json", "selected_accuracy"),
    ("model_tot", root / "model_tot.summary.json", "accuracy"),
]
lines = ["method,metric,value,total"]
for label, path, metric in items:
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    lines.append(f"{label},{metric},{data.get(metric)},{data.get('total')}")
(root / "strategy_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
