#!/usr/bin/env python3
"""Convert a local CSV or Hugging Face dataset to project JSONL."""

import argparse
from pathlib import Path

from game24.data import deduplicate, normalize_record, read_records, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Game of 24 JSONL dataset")
    parser.add_argument(
        "--input-file",
        help="Local CSV file; when set, Hugging Face is not accessed",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset name, such as nlile/24-game",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_file:
        records = read_records(args.input_file)
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset, split=args.split)
        records = [dict(record) for record in dataset]

    if args.limit is not None:
        records = records[: args.limit]
    examples = [
        normalize_record(record, index, args.dataset)
        for index, record in enumerate(records)
    ]
    examples = deduplicate(examples)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(examples, output)
    print(f"Saved {len(examples)} unique puzzles to {output}")


if __name__ == "__main__":
    main()
