"""
Preprocess Game of 24 datasets for TinyZero/veRL GRPO training.
"""

import argparse
import csv
import itertools
import os
import random
import re
from collections import Counter
from fractions import Fraction
from urllib.request import urlopen

from datasets import Dataset, load_dataset

try:
    from verl.utils.hdfs_io import copy, makedirs
except ModuleNotFoundError:
    copy = None
    makedirs = None


DATA_SOURCE = "game24"
TARGET = 24
HARD_CSV_URL = "https://huggingface.co/datasets/test-time-compute/game-of-24/resolve/main/game24.csv"


def make_prompt(numbers, template_type="qwen-instruct"):
    numbers_text = " ".join(str(number) for number in numbers)
    task = f"""You are solving the Game of 24.

Use each of the four given numbers exactly once.
You may only use +, -, *, /, and parentheses.
The expression must evaluate to 24.

Return exactly:
<think>brief reasoning</think>
<answer>expression only</answer>

Numbers: {numbers_text}"""

    if template_type == "qwen-instruct":
        return (
            "<|im_start|>system\n"
            "You are a careful arithmetic reasoner. Follow the requested output format exactly."
            "<|im_end|>\n"
            f"<|im_start|>user\n{task}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if template_type == "base":
        return f"User: {task}\nAssistant:\n"
    raise ValueError(f"Unsupported template_type: {template_type}")


def canonical_key(numbers):
    return tuple(sorted(int(number) for number in numbers))


def _field(example, candidates, default=None):
    normalized = {key.lower().replace(" ", "_").replace("-", "_"): key for key in example.keys()}
    for candidate in candidates:
        key = normalized.get(candidate.lower().replace(" ", "_").replace("-", "_"))
        if key is not None:
            return example[key]
    return default


def parse_numbers(value):
    if isinstance(value, str):
        return [int(number) for number in re.findall(r"\d+", value)]
    if isinstance(value, (list, tuple)):
        return [int(number) for number in value]
    raise ValueError(f"Cannot parse Game of 24 numbers from {value!r}")


def parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def normalize_primary(example, index):
    numbers = parse_numbers(_field(example, ["numbers", "nums", "puzzles", "puzzle"]))
    solvable = parse_bool(_field(example, ["solvable"], True))
    return {
        "numbers": numbers,
        "solvable": solvable,
        "source": "nlile/24-game",
        "source_index": index,
        "solutions": _field(example, ["solutions", "solution"], None),
    }


def normalize_hard(example, index):
    numbers = parse_numbers(_field(example, ["puzzles", "puzzle", "numbers", "nums"]))
    rank = _field(example, ["rank"], index + 1)
    solved_rate = _field(example, ["solved_rate", "solved rate"], None)
    return {
        "numbers": numbers,
        "solvable": True,
        "source": "test-time-compute/game-of-24",
        "source_index": index,
        "rank": int(rank),
        "solved_rate": solved_rate,
    }


def load_hard_items():
    """Load the hard Game-of-24 set, with a CSV fallback for fragile pandas stacks."""
    try:
        hard_raw = load_dataset("test-time-compute/game-of-24", split="train")
        return [normalize_hard(example, index) for index, example in enumerate(hard_raw)]
    except Exception as exc:
        print(f"Warning: load_dataset failed for test-time-compute/game-of-24: {exc}")
        print(f"Falling back to direct CSV download: {HARD_CSV_URL}")

    with urlopen(HARD_CSV_URL) as response:
        text = response.read().decode("utf-8")
    rows = csv.DictReader(text.splitlines())
    return [normalize_hard(row, index) for index, row in enumerate(rows)]


def _solvable_values(values):
    if len(values) == 1:
        return {values[0]}
    results = set()
    for left_indices in _proper_subsets(len(values)):
        right_indices = tuple(index for index in range(len(values)) if index not in left_indices)
        if not right_indices:
            continue
        left_values = [values[index] for index in left_indices]
        right_values = [values[index] for index in right_indices]
        for left in _solvable_values(tuple(left_values)):
            for right in _solvable_values(tuple(right_values)):
                results.add(left + right)
                results.add(left - right)
                results.add(right - left)
                results.add(left * right)
                if right != 0:
                    results.add(left / right)
                if left != 0:
                    results.add(right / left)
    return results


def _proper_subsets(length):
    # Generate one side of each partition only; the operations cover both orders.
    indices = tuple(range(length))
    first = indices[0]
    for size in range(1, length):
        for combo in itertools.combinations(indices, size):
            if first in combo and len(combo) < length:
                yield combo


def is_solvable(numbers):
    values = tuple(Fraction(int(number), 1) for number in numbers)
    return Fraction(TARGET, 1) in _solvable_values(values)


def make_record(item, split, index, template_type):
    numbers = [int(number) for number in item["numbers"]]
    key = canonical_key(numbers)
    prompt = make_prompt(numbers, template_type=template_type)
    ground_truth = {
        "target": TARGET,
        "numbers": numbers,
        "solvable": bool(item.get("solvable", True)),
    }
    extra_info = {
        "split": split,
        "index": index,
        "key": " ".join(str(number) for number in key),
        "source": item.get("source"),
        "source_index": item.get("source_index"),
    }
    if "rank" in item:
        extra_info["rank"] = item["rank"]
    if "solved_rate" in item:
        extra_info["solved_rate"] = item["solved_rate"]

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "numbers": numbers,
        "target": TARGET,
        "solvable": bool(item.get("solvable", True)),
        "split": split,
        "index": index,
        "extra_info": extra_info,
    }


def unique_by_key(items):
    result = []
    seen = set()
    duplicates = 0
    for item in items:
        key = canonical_key(item["numbers"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        result.append(item)
    return result, duplicates


def synthesize_unsolvable(existing_keys, limit):
    items = []
    for numbers in itertools.combinations_with_replacement(range(1, 14), 4):
        key = canonical_key(numbers)
        if key in existing_keys:
            continue
        if not is_solvable(numbers):
            items.append({
                "numbers": list(numbers),
                "solvable": False,
                "source": "synthetic-1-13-unsolvable",
                "source_index": len(items),
            })
            if limit > 0 and len(items) >= limit:
                break
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/game24")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--template_type", default="qwen-instruct", choices=["qwen-instruct", "base"])
    parser.add_argument("--validation_size", type=int, default=128)
    parser.add_argument("--test_id_size", type=int, default=256)
    parser.add_argument("--test_hard_start", type=int, default=900)
    parser.add_argument("--test_hard_end", type=int, default=1000)
    parser.add_argument("--test_unsolvable_size", type=int, default=256)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthesize_unsolvable", action="store_true", default=True)
    parser.add_argument("--no_synthesize_unsolvable", action="store_false", dest="synthesize_unsolvable")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.local_dir, exist_ok=True)

    primary_raw = load_dataset("nlile/24-game", split="train")

    primary_items = [normalize_primary(example, index) for index, example in enumerate(primary_raw)]
    hard_items_all = load_hard_items()
    hard_slice = hard_items_all[args.test_hard_start:args.test_hard_end]

    solvable_items = [item for item in primary_items if item["solvable"]]
    primary_unsolvable_items = [item for item in primary_items if not item["solvable"]]
    solvable_unique, solvable_dupes = unique_by_key(solvable_items)
    unsolvable_unique, unsolvable_dupes = unique_by_key(primary_unsolvable_items)
    hard_unique, hard_dupes = unique_by_key(hard_slice)

    hard_keys = {canonical_key(item["numbers"]) for item in hard_unique}
    train_pool = [item for item in solvable_unique if canonical_key(item["numbers"]) not in hard_keys]
    excluded_for_hard = len(solvable_unique) - len(train_pool)
    random.shuffle(train_pool)

    val_size = min(args.validation_size, len(train_pool))
    validation_items = train_pool[:val_size]
    remaining = train_pool[val_size:]
    test_id_size = min(args.test_id_size, len(remaining))
    test_id_items = remaining[:test_id_size]
    train_items = remaining[test_id_size:]
    if args.train_size > 0:
        train_items = train_items[:args.train_size]

    train_keys = {canonical_key(item["numbers"]) for item in train_items}
    validation_keys = {canonical_key(item["numbers"]) for item in validation_items}
    test_id_keys = {canonical_key(item["numbers"]) for item in test_id_items}
    blocked_keys = train_keys | validation_keys | test_id_keys

    test_hard_items = [item for item in hard_unique if canonical_key(item["numbers"]) not in blocked_keys]
    hard_overlap_excluded = len(hard_unique) - len(test_hard_items)

    unsolvable_pool = [item for item in unsolvable_unique if canonical_key(item["numbers"]) not in blocked_keys]
    if args.synthesize_unsolvable and len(unsolvable_pool) < args.test_unsolvable_size:
        existing_keys = blocked_keys | {canonical_key(item["numbers"]) for item in unsolvable_pool}
        needed = args.test_unsolvable_size - len(unsolvable_pool)
        unsolvable_pool.extend(synthesize_unsolvable(existing_keys=existing_keys, limit=needed))
    test_unsolvable_items = unsolvable_pool[:args.test_unsolvable_size]

    splits = {
        "train": train_items,
        "validation": validation_items,
        "test_id": test_id_items,
        "test_hard": test_hard_items,
        "test_unsolvable": test_unsolvable_items,
    }

    for split, items in splits.items():
        records = [make_record(item, split=split, index=index, template_type=args.template_type)
                   for index, item in enumerate(items)]
        Dataset.from_list(records).to_parquet(os.path.join(args.local_dir, f"{split}.parquet"))

    stats = {
        "primary_raw": len(primary_items),
        "primary_solvable_raw": len(solvable_items),
        "primary_unsolvable_raw": len(primary_unsolvable_items),
        "solvable_unique": len(solvable_unique),
        "unsolvable_unique": len(unsolvable_unique),
        "solvable_duplicates_removed": solvable_dupes,
        "unsolvable_duplicates_removed": unsolvable_dupes,
        "hard_raw_slice": len(hard_slice),
        "hard_unique": len(hard_unique),
        "hard_duplicates_removed": hard_dupes,
        "excluded_train_pool_due_to_hard_overlap": excluded_for_hard,
        "excluded_hard_due_to_train_val_test_overlap": hard_overlap_excluded,
        **{f"{split}_count": len(items) for split, items in splits.items()},
    }

    print("Game24 preprocessing statistics")
    for key, value in stats.items():
        print(f"{key}: {value}")

    if args.hdfs_dir is not None:
        if copy is None or makedirs is None:
            raise RuntimeError("HDFS copy requires installing veRL dependencies, including torch.")
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
