#!/usr/bin/env python3
"""Build the canonical leakage-safe Game24 experiment datasets."""

import argparse
import json
import random
from itertools import combinations_with_replacement
from pathlib import Path

from game24.data import Game24Example, normalize_record, read_records, save_jsonl
from game24.solver import find_solution
from game24.splits import build_game24_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical Game24 experiment splits")
    parser.add_argument("--training-file", required=True, help="Raw nlile/24-game file")
    parser.add_argument("--ranked-file", required=True, help="Raw ToT 24.csv or equivalent")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--test-start", type=int, default=900)
    parser.add_argument("--test-end", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--unsolvable-size", type=int, default=100)
    parser.add_argument("--train-unsolvable-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized(path: str, source: str) -> list[Game24Example]:
    """Read and normalize a raw dataset."""

    return [
        normalize_record(record, index, source) for index, record in enumerate(read_records(path))
    ]


def generate_unsolvable(
    test_size: int,
    train_size: int,
    seed: int,
) -> tuple[list[Game24Example], list[Game24Example], int]:
    """Generate lexicographically stable, exactly verified unsolvable puzzles."""

    candidates = []
    for index, numbers in enumerate(combinations_with_replacement(range(1, 14), 4)):
        if find_solution(numbers, 24) is None:
            candidates.append(
                Game24Example(
                    example_id=f"generated-unsolvable:{index}",
                    numbers=numbers,
                    solvable=False,
                    source="exact-enumeration",
                )
            )
    required = test_size + train_size
    if len(candidates) < required:
        raise RuntimeError(f"only found {len(candidates)} unsolvable puzzles")
    random.Random(seed).shuffle(candidates)
    test = [
        Game24Example(**{**candidate.__dict__, "split": "test_unsolvable"})
        for candidate in candidates[:test_size]
    ]
    train = [
        Game24Example(**{**candidate.__dict__, "split": "train_unsolvable"})
        for candidate in candidates[test_size:required]
    ]
    return test, train, len(candidates)


def main() -> None:
    args = parse_args()
    training = normalized(args.training_file, "nlile/24-game")
    ranked = normalized(args.ranked_file, "test-time-compute/game-of-24")
    splits, manifest = build_game24_splits(
        training,
        ranked,
        test_start=args.test_start,
        test_end=args.test_end,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    unsolvable, train_unsolvable, unsolvable_pool_size = generate_unsolvable(
        args.unsolvable_size,
        args.train_unsolvable_size,
        args.seed,
    )
    splits["test_unsolvable"] = unsolvable
    if train_unsolvable:
        splits["train_unsolvable"] = train_unsolvable

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        save_jsonl(examples, output_dir / f"{name}.jsonl")
    manifest["splits"]["test_unsolvable"] = {
        "count": len(splits["test_unsolvable"]),
        "pool_count": unsolvable_pool_size,
        "construction": (
            "exact exhaustive Fraction solver over combinations_with_replacement(1..13, 4)"
        ),
    }
    if train_unsolvable:
        manifest["splits"]["train_unsolvable"] = {
            "count": len(train_unsolvable),
            "pool_count": unsolvable_pool_size,
            "construction": (
                "exact exhaustive Fraction solver over combinations_with_replacement(1..13, 4); "
                "disjoint from test_unsolvable after seeded shuffle"
            ),
        }
    manifest["arguments"] = vars(args)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
