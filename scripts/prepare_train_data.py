#!/usr/bin/env python3
"""Prepare reproducible train/validation splits from local Game of 24 data."""

import argparse
import random
from pathlib import Path

from game24.data import (
    deduplicate,
    load_jsonl,
    normalize_record,
    number_key,
    read_records,
    save_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local Game of 24 training data")
    parser.add_argument("--input-file", required=True, help="Local CSV, JSON, or JSONL file")
    parser.add_argument("--source", default="nlile/24-game", help="Dataset source label")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-file", help="Project JSONL test set used to prevent leakage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1")

    records = read_records(args.input_file)
    raw_examples = [
        normalize_record(record, index, args.source) for index, record in enumerate(records)
    ]
    raw_solvable = sum(example.solvable is True for example in raw_examples)
    raw_unsolvable = len(raw_examples) - raw_solvable

    unique_examples = deduplicate(raw_examples)
    duplicate_count = len(raw_examples) - len(unique_examples)

    overlap_count = 0
    if args.test_file:
        test_keys = {number_key(example) for example in load_jsonl(args.test_file)}
        filtered_examples = [
            example for example in unique_examples if number_key(example) not in test_keys
        ]
        overlap_count = len(unique_examples) - len(filtered_examples)
        unique_examples = filtered_examples

    solvable = [example for example in unique_examples if example.solvable is True]
    unsolvable = [example for example in unique_examples if example.solvable is not True]
    random.Random(args.seed).shuffle(solvable)

    validation_size = int(len(solvable) * args.val_ratio)
    validation = solvable[:validation_size]
    train = solvable[validation_size:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(validation, output_dir / "validation.jsonl")
    save_jsonl(unsolvable, output_dir / "unsolvable_test.jsonl")

    print(f"Raw samples: {len(raw_examples)}")
    print(f"Raw solvable: {raw_solvable}")
    print(f"Raw unsolvable: {raw_unsolvable}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Test overlaps removed: {overlap_count}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Unsolvable test: {len(unsolvable)}")


if __name__ == "__main__":
    main()
