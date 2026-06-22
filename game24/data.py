"""Puzzle data and small JSONL dataset helpers."""

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

NUMBER_FIELDS = ("numbers", "nums", "puzzle", "Puzzles", "input")
SOLUTION_FIELDS = ("reference_answer", "solution", "solutions", "answer", "Solutions")
SOLVABLE_FIELDS = ("solvable", "is_solvable", "Solvable")
TARGET_FIELDS = ("target", "Target")


@dataclass
class Game24Example:
    """One fixed-24 or variable-target arithmetic puzzle."""

    example_id: str
    numbers: tuple[int, ...]
    solvable: bool | None = None
    reference_answer: str | None = None
    source: str | None = None
    rank: int | None = None
    solved_rate: float | None = None
    target: int = 24
    task_type: str = "game24"
    split: str | None = None

    def __post_init__(self) -> None:
        if not 3 <= len(self.numbers) <= 6:
            raise ValueError("numbers must contain between three and six integers")
        if any(type(number) is not int or number < 1 for number in self.numbers):
            raise ValueError("numbers must be positive integers")
        if type(self.target) is not int or self.target < 1:
            raise ValueError("target must be a positive integer")
        if self.task_type == "game24" and (
            len(self.numbers) != 4
            or self.target != 24
            or any(number > 13 for number in self.numbers)
        ):
            raise ValueError("Game24 requires four integers in [1, 13] and target 24")


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
    if not 3 <= len(numbers) <= 6:
        raise ValueError(f"expected three to six numbers, got {numbers}")

    target_value = _first_value(record, TARGET_FIELDS)
    target = int(target_value) if target_value is not None else 24
    is_countdown = "countdown" in source.lower() or target != 24 or len(numbers) != 4
    task_type = "countdown" if is_countdown else "game24"

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
        solvable = (
            source == "test-time-compute/game-of-24"
            or "countdown" in source.lower()
            or reference is not None
        )

    rank_value = record.get("Rank", record.get("rank"))
    rank = int(float(rank_value)) if rank_value not in (None, "") else None
    solved_rate = _parse_solved_rate(record.get("Solved rate", record.get("solved_rate")))
    return Game24Example(
        example_id=f"{source}:{rank if rank is not None else index}",
        numbers=tuple(numbers),
        solvable=solvable,
        reference_answer=str(reference) if reference is not None else None,
        source=source,
        rank=rank,
        solved_rate=solved_rate,
        target=target,
        task_type=task_type,
        split=record.get("split"),
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


def number_key(example: Game24Example) -> tuple[int, ...]:
    """Return an order-independent key while preserving repeated numbers."""

    return tuple(sorted(example.numbers))


def deduplicate(examples: list[Game24Example]) -> list[Game24Example]:
    """Keep the first puzzle for each target and number combination."""

    result = []
    seen = set()
    for example in examples:
        key = (example.target, number_key(example))
        if key not in seen:
            seen.add(key)
            result.append(example)
    return result


def find_overlaps(
    first: list[Game24Example], second: list[Game24Example]
) -> set[tuple[int, tuple[int, ...]]]:
    """Return target/number tasks that occur in both datasets."""

    return {(item.target, number_key(item)) for item in first} & {
        (item.target, number_key(item)) for item in second
    }


def dataset_fingerprint(examples: list[Game24Example]) -> str:
    """Return a stable identifier for the ordered evaluation population."""

    rows = [
        {"id": item.example_id, "numbers": sorted(item.numbers), "target": item.target}
        for item in examples
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
