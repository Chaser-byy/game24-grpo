"""Leakage-safe experiment splits for the overlapping Game24 datasets."""

import random
from dataclasses import replace
from typing import Any

from game24.data import Game24Example, dataset_fingerprint, deduplicate, number_key


def build_game24_splits(
    training_examples: list[Game24Example],
    ranked_examples: list[Game24Example],
    *,
    test_start: int = 900,
    test_end: int = 1000,
    validation_size: int = 100,
    seed: int = 42,
) -> tuple[dict[str, list[Game24Example]], dict[str, Any]]:
    """Hold out the ToT slice, then create a reproducible ID validation split."""

    training_examples = deduplicate(training_examples)
    ranked_examples = deduplicate(ranked_examples)
    if not 0 <= test_start < test_end <= len(ranked_examples):
        raise ValueError("invalid zero-based test slice")

    hard_test = ranked_examples[test_start:test_end]
    hard_keys = {number_key(example) for example in hard_test}
    remaining = [
        example
        for example in training_examples
        if example.solvable is True and number_key(example) not in hard_keys
    ]
    if len(remaining) <= validation_size:
        raise ValueError("not enough non-test puzzles for the requested validation size")

    random.Random(seed).shuffle(remaining)
    validation = remaining[:validation_size]
    train = remaining[validation_size:]

    splits = {
        "train": [replace(example, split="train") for example in train],
        "validation_id": [replace(example, split="validation_id") for example in validation],
        "train_full": [replace(example, split="train_full") for example in remaining],
        "test_hard": [replace(example, split="test_hard") for example in hard_test],
    }

    assert not ({number_key(item) for item in splits["train_full"]} & hard_keys)
    manifest = {
        "seed": seed,
        "test_slice": {"start": test_start, "end": test_end, "semantics": "python [start:end)"},
        "source_training_total": len(training_examples),
        "source_ranked_total": len(ranked_examples),
        "source_overlap": len(
            {number_key(item) for item in training_examples}
            & {number_key(item) for item in ranked_examples}
        ),
        "splits": {
            name: {"count": len(items), "fingerprint": dataset_fingerprint(items)}
            for name, items in splits.items()
        },
    }
    return splits, manifest
