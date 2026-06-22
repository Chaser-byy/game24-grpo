#!/usr/bin/env python3
"""Prepare local Countdown-Tasks-3to4 data for task-OOD experiments."""

import argparse
import random
from dataclasses import replace
from pathlib import Path

from game24.data import deduplicate, normalize_record, read_records, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Countdown OOD data")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", default="data/processed/countdown")
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = deduplicate(
        [
            normalize_record(record, index, "Jiayi-Pan/Countdown-Tasks-3to4")
            for index, record in enumerate(read_records(args.input_file))
        ]
    )
    random.Random(args.seed).shuffle(examples)
    if args.limit > 0:
        examples = examples[: args.limit]
    if len(examples) <= args.validation_size:
        raise SystemExit("Countdown data is smaller than --validation-size")
    validation = [replace(item, split="countdown_ood") for item in examples[: args.validation_size]]
    train = [replace(item, split="countdown_train") for item in examples[args.validation_size :]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(validation, output_dir / "validation_ood.jsonl")
    print(f"Countdown train: {len(train)}")
    print(f"Countdown OOD validation: {len(validation)}")


if __name__ == "__main__":
    main()
