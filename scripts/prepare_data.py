#!/usr/bin/env python3
"""Convert a local CSV or Hugging Face dataset to project JSONL."""

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from game24.data import Game24Example, deduplicate, save_jsonl

NUMBER_FIELDS = ("numbers", "nums", "puzzle", "Puzzles", "input")
SOLUTION_FIELDS = ("reference_answer", "solution", "solutions", "answer", "Solutions")
SOLVABLE_FIELDS = ("solvable", "is_solvable", "Solvable")


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


def first_value(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        if field in record:
            return record[field]
    return None


def parse_numbers(record: dict[str, Any]) -> tuple[int, int, int, int]:
    value = first_value(record, NUMBER_FIELDS)
    if isinstance(value, str):
        values = [int(number) for number in re.findall(r"\d+", value)]
    elif isinstance(value, (list, tuple)):
        values = [int(number) for number in value]
    else:
        raise ValueError(f"cannot find puzzle numbers in fields {list(record)}")
    if len(values) != 4:
        raise ValueError(f"expected four numbers, got {values}")
    return values[0], values[1], values[2], values[3]


def parse_reference(record: dict[str, Any]) -> str | None:
    value = first_value(record, SOLUTION_FIELDS)
    if isinstance(value, list):
        value = value[0] if value else None
    if value in (None, "", "[]"):
        return None
    return str(value)


def parse_solvable(record: dict[str, Any], reference: str | None, source: str) -> bool:
    value = first_value(record, SOLVABLE_FIELDS)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if value is not None:
        return bool(value)
    if source == "test-time-compute/game-of-24":
        return True
    return reference is not None


def normalize_record(record: dict[str, Any], index: int, source: str) -> Game24Example:
    """Convert common 24-point dataset fields to one project example."""

    reference = parse_reference(record)
    rank = int(float(record["Rank"])) if record.get("Rank") else None
    solved_rate = parse_solved_rate(record.get("Solved rate"))
    return Game24Example(
        example_id=f"{source}:{rank if rank is not None else index}",
        numbers=parse_numbers(record),
        solvable=parse_solvable(record, reference, source),
        reference_answer=reference,
        source=source,
        rank=rank,
        solved_rate=solved_rate,
    )


def parse_solved_rate(value: Any) -> float | None:
    """Convert values such as '75%' or 0.75 to a 0-1 rate."""

    if value in (None, ""):
        return None
    text = str(value).strip()
    rate = float(text.removesuffix("%"))
    return rate / 100 if text.endswith("%") or rate > 1 else rate


def read_local_csv(path: str) -> list[dict[str, Any]]:
    """Read a local CSV without importing or contacting Hugging Face."""

    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main() -> None:
    args = parse_args()

    if args.input_file:
        records = read_local_csv(args.input_file)
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
