"""Supervised warm-start examples for sparse-reward arithmetic RL."""

from dataclasses import dataclass

from game24.data import Game24Example
from game24.parser import parse_response
from game24.rewards import UNSOLVABLE_ANSWER
from game24.solver import find_solution
from game24.verifier import check_expression


@dataclass(frozen=True)
class SFTExample:
    """One solver-labeled warm-start row."""

    source_id: str
    numbers: tuple[int, ...]
    target: int
    expression: str
    response: str


def build_sft_response(expression: str, target: int = 24) -> str:
    """Create a compact R1-style response around a verified expression."""

    if expression == UNSOLVABLE_ANSWER:
        return (
            "<think>I checked the legal arithmetic combinations and found no expression "
            f"that reaches {target}.</think>\n"
            f"<answer>{UNSOLVABLE_ANSWER}</answer>"
        )
    return (
        f"<think>A legal expression is {expression}. "
        f"It uses each input number exactly once and evaluates to {target}.</think>\n"
        f"<answer>{expression}</answer>"
    )


def solver_label(example: Game24Example) -> SFTExample | None:
    """Return a supervised label for a solvable example, or ``None`` if unsolved."""

    expression = find_solution(example.numbers, example.target)
    if example.solvable is False:
        if expression is not None:
            raise ValueError(
                f"example is marked unsolvable but solver found {expression}: "
                f"{example.example_id}"
            )
        response = build_sft_response(UNSOLVABLE_ANSWER, example.target)
        parsed = parse_response(response)
        if not parsed.valid_format:
            raise ValueError(f"invalid SFT response for {example.example_id}: {parsed.reason}")
        return SFTExample(
            source_id=example.example_id,
            numbers=example.numbers,
            target=example.target,
            expression=UNSOLVABLE_ANSWER,
            response=response,
        )
    if expression is None:
        return None
    result = check_expression(expression, example.numbers, example.target)
    if not result.valid:
        raise ValueError(
            f"solver produced invalid expression for {example.example_id}: {expression}"
        )
    response = build_sft_response(expression, example.target)
    parsed = parse_response(response)
    if not parsed.valid_format:
        raise ValueError(f"invalid SFT response for {example.example_id}: {parsed.reason}")
    return SFTExample(
        source_id=example.example_id,
        numbers=example.numbers,
        target=example.target,
        expression=expression,
        response=response,
    )


def build_sft_examples(examples: list[Game24Example]) -> list[SFTExample]:
    """Label all exactly solvable or exactly unsolvable examples."""

    rows = []
    for example in examples:
        row = solver_label(example)
        if row is not None:
            rows.append(row)
    return rows
