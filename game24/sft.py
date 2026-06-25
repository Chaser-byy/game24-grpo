"""Supervised warm-start examples for sparse-reward arithmetic RL."""

from dataclasses import dataclass

from game24.data import Game24Example
from game24.parser import parse_response
from game24.rewards import UNSOLVABLE_ANSWER
from game24.solver import find_solution
from game24.trajectory import (
    enumerate_trajectories,
    trajectory_to_direct_tot_response,
    trajectory_to_response,
)
from game24.verifier import check_expression


@dataclass(frozen=True)
class SFTExample:
    """One solver-labeled warm-start row."""

    source_id: str
    numbers: tuple[int, ...]
    target: int
    expression: str
    response: str
    trace: tuple[dict, ...] = ()


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


def solver_label(
    example: Game24Example,
    *,
    label_style: str = "trajectory",
) -> SFTExample | None:
    """Return a supervised label for a solvable example, or ``None`` if unsolved."""

    _validate_label_style(label_style)
    if example.solvable is False:
        expression = find_solution(example.numbers, example.target)
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
    trajectories = enumerate_trajectories(example.numbers, example.target, max_solutions=1)
    if not trajectories:
        return None
    trajectory = trajectories[0]
    expression = trajectory.expression
    result = check_expression(expression, example.numbers, example.target)
    if not result.valid:
        raise ValueError(
            f"solver produced invalid expression for {example.example_id}: {expression}"
        )
    if label_style == "trajectory":
        response = trajectory_to_response(trajectory)
    elif label_style == "compact":
        response = build_sft_response(expression, example.target)
    elif label_style == "direct_tot":
        response = trajectory_to_direct_tot_response(trajectory)
    else:
        raise ValueError(f"unknown SFT label style: {label_style}")
    parsed = parse_response(response)
    if not parsed.valid_format:
        raise ValueError(f"invalid SFT response for {example.example_id}: {parsed.reason}")
    return SFTExample(
        source_id=example.example_id,
        numbers=example.numbers,
        target=example.target,
        expression=expression,
        response=response,
        trace=tuple(step.__dict__ for step in trajectory.steps),
    )


def build_sft_examples(
    examples: list[Game24Example],
    *,
    label_style: str = "trajectory",
    solutions_per_example: int = 1,
) -> list[SFTExample]:
    """Label all exactly solvable or exactly unsolvable examples."""

    if solutions_per_example < 1:
        raise ValueError("solutions_per_example must be positive")
    _validate_label_style(label_style)
    rows = []
    for example in examples:
        if example.solvable is False:
            row = solver_label(example, label_style=label_style)
            if row is not None:
                rows.append(row)
            continue
        trajectories = enumerate_trajectories(
            example.numbers,
            example.target,
            max_solutions=solutions_per_example,
        )
        for trajectory in trajectories:
            if label_style == "trajectory":
                response = trajectory_to_response(trajectory)
            elif label_style == "compact":
                response = build_sft_response(trajectory.expression, example.target)
            elif label_style == "direct_tot":
                response = trajectory_to_direct_tot_response(trajectory)
            else:
                raise ValueError(f"unknown SFT label style: {label_style}")
            parsed = parse_response(response)
            if not parsed.valid_format:
                raise ValueError(f"invalid SFT response for {example.example_id}: {parsed.reason}")
            rows.append(
                SFTExample(
                    source_id=example.example_id,
                    numbers=example.numbers,
                    target=example.target,
                    expression=trajectory.expression,
                    response=response,
                    trace=tuple(step.__dict__ for step in trajectory.steps),
                )
            )
    return rows


def _validate_label_style(label_style: str) -> None:
    if label_style not in {"trajectory", "compact", "direct_tot"}:
        raise ValueError(f"unknown SFT label style: {label_style}")
