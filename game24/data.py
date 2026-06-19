"""Puzzle data and small JSONL dataset helpers."""

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

NUMBER_FIELDS = ("numbers", "nums", "puzzle", "Puzzles", "input")
SOLUTION_FIELDS = ("reference_answer", "solution", "solutions", "answer", "Solutions")
SOLVABLE_FIELDS = ("solvable", "is_solvable", "Solvable")


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


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read raw records from a local CSV, JSON, or JSONL file."""

    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    if path.suffix.lower() == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(content, list):
            return content
        if isinstance(content, dict):
            for field in ("train", "data"):
                if isinstance(content.get(field), list):
                    return content[field]
            return [content]
    raise ValueError("input file must be CSV, JSON, or JSONL")


def normalize_record(record: dict[str, Any], index: int, source: str) -> Game24Example:
    """Convert common 24-point dataset fields to one project example."""

    numbers_value = _first_value(record, NUMBER_FIELDS)
    if isinstance(numbers_value, str):
        numbers = [int(number) for number in re.findall(r"\d+", numbers_value)]
    else:
        numbers = [int(number) for number in numbers_value]
    if len(numbers) != 4:
        raise ValueError(f"expected four numbers, got {numbers}")

    reference = _first_value(record, SOLUTION_FIELDS)
    if isinstance(reference, list):
        reference = reference[0] if reference else None
    if reference in ("", "[]"):
        reference = None

    solvable_value = _first_value(record, SOLVABLE_FIELDS)
    if isinstance(solvable_value, str):
        solvable = solvable_value.strip().lower() in {"true", "1", "yes"}
    elif solvable_value is not None:
        solvable = bool(solvable_value)
    else:
        solvable = source == "test-time-compute/game-of-24" or reference is not None

    rank = int(float(record["Rank"])) if record.get("Rank") else None
    solved_rate = _parse_solved_rate(record.get("Solved rate"))
    return Game24Example(
        example_id=f"{source}:{rank if rank is not None else index}",
        numbers=(numbers[0], numbers[1], numbers[2], numbers[3]),
        solvable=solvable,
        reference_answer=str(reference) if reference is not None else None,
        source=source,
        rank=rank,
        solved_rate=solved_rate,
    )


def _first_value(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        if field in record:
            return record[field]
    return None


def _parse_solved_rate(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    rate = float(text.removesuffix("%"))
    return rate / 100 if text.endswith("%") or rate > 1 else rate


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
