"""Puzzle data and small JSONL dataset helpers."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Game24Example:
    """One Game of 24 puzzle."""

    example_id: str
    numbers: tuple[int, int, int, int]
    solvable: bool | None = None
    reference_answer: str | None = None
    source: str | None = None
    rank: int | None = None
    solved_rate: float | None = None

    def __post_init__(self) -> None:
        if len(self.numbers) != 4 or any(number < 1 or number > 13 for number in self.numbers):
            raise ValueError("numbers must contain four integers between 1 and 13")


def load_jsonl(path: str | Path) -> list[Game24Example]:
    """Read puzzle examples from a JSONL file."""

    examples = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record["numbers"] = tuple(record["numbers"])
                examples.append(Game24Example(**record))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid record on line {line_number}: {error}") from error
    return examples


def save_jsonl(examples: list[Game24Example], path: str | Path) -> None:
    """Write puzzle examples to a JSONL file."""

    with open(path, "w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def number_key(example: Game24Example) -> tuple[int, int, int, int]:
    """Return an order-independent key while preserving repeated numbers."""

    return tuple(sorted(example.numbers))


def deduplicate(examples: list[Game24Example]) -> list[Game24Example]:
    """Keep the first puzzle for each number combination."""

    result = []
    seen = set()
    for example in examples:
        key = number_key(example)
        if key not in seen:
            seen.add(key)
            result.append(example)
    return result


def find_overlaps(
    first: list[Game24Example], second: list[Game24Example]
) -> set[tuple[int, int, int, int]]:
    """Return number combinations that occur in both datasets."""

    return {number_key(item) for item in first} & {number_key(item) for item in second}
