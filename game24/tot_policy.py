"""Supervised data generation for model-guided ToT next-operation policies."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any

from game24.data import Game24Example
from game24.model_tot import (
    ModelToTState,
    OperationCandidate,
    apply_candidate,
    build_tot_prompt,
    initial_state,
)

OPERATIONS = ("+", "-", "*", "/")
OPERATION_ORDER = {operation: index for index, operation in enumerate(OPERATIONS)}


@dataclass(frozen=True)
class ToTPolicySample:
    """One prompt/label pair for a single verified next operation."""

    source_id: str
    numbers: tuple[int, ...]
    target: int
    state: ModelToTState
    prompt: str
    label: str
    action: OperationCandidate


def build_tot_policy_samples(
    examples: Sequence[Game24Example],
    *,
    candidates_per_state: int = 4,
    max_states_per_example: int = 16,
    max_actions_per_state: int = 4,
) -> list[ToTPolicySample]:
    """Build ToT next-operation SFT samples from exactly solvable examples."""

    samples = []
    for example in examples:
        samples.extend(
            generate_tot_policy_samples(
                example,
                candidates_per_state=candidates_per_state,
                max_states_per_example=max_states_per_example,
                max_actions_per_state=max_actions_per_state,
            )
        )
    return samples


def generate_tot_policy_samples(
    example: Game24Example,
    *,
    candidates_per_state: int = 4,
    max_states_per_example: int = 16,
    max_actions_per_state: int = 4,
) -> list[ToTPolicySample]:
    """Generate prompt/action labels for one solvable arithmetic puzzle."""

    _validate_limits(candidates_per_state, max_states_per_example, max_actions_per_state)
    if example.solvable is False:
        return []

    start = initial_state(example.numbers)
    if not state_can_reach_target(start, example.target):
        return []

    queue: deque[ModelToTState] = deque([start])
    seen = {_state_key(start)}
    states_used = 0
    samples: list[ToTPolicySample] = []

    while queue and states_used < max_states_per_example:
        state = queue.popleft()
        actions = continuing_actions(state, example.target)
        if not actions:
            continue

        states_used += 1
        selected_actions = actions[:max_actions_per_state]
        prompt = build_tot_prompt(state, example.target, candidates_per_state)
        for action in selected_actions:
            samples.append(
                ToTPolicySample(
                    source_id=example.example_id,
                    numbers=example.numbers,
                    target=example.target,
                    state=state,
                    prompt=prompt,
                    label=_action_label(action),
                    action=action,
                )
            )

        for action in selected_actions:
            child = apply_candidate(state, action, source="teacher")
            if child is None:
                continue
            key = _state_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append(child)

    return samples


def continuing_actions(state: ModelToTState, target: int) -> list[OperationCandidate]:
    """Return legal state actions whose child state can still reach the target."""

    if len(state.items) < 2:
        return []

    actions = []
    for left_index in range(len(state.items)):
        for right_index in range(len(state.items)):
            if left_index == right_index:
                continue
            for operation in OPERATIONS:
                candidate = OperationCandidate(
                    left_index,
                    operation,
                    right_index,
                    f"{left_index} {operation} {right_index}",
                )
                child = apply_candidate(state, candidate, source="teacher")
                if child is not None and state_can_reach_target(child, target):
                    actions.append(candidate)

    actions.sort(key=lambda action: _action_sort_key(state, action, Fraction(target)))
    return actions


def state_can_reach_target(state: ModelToTState, target: int) -> bool:
    """Return whether the remaining exact values can be combined into target."""

    return values_can_reach_target(tuple(item.value for item in state.items), target)


def values_can_reach_target(values: Sequence[Fraction | int], target: int) -> bool:
    """Return whether a multiset of exact values can reach target by binary operations."""

    canonical_values = _canonical_values(Fraction(value) for value in values)
    return _can_reach_values(canonical_values, Fraction(target))


def sample_to_record(sample: ToTPolicySample) -> dict[str, Any]:
    """Serialize a policy sample without exposing non-JSON Fraction objects."""

    return {
        "source_id": sample.source_id,
        "numbers": list(sample.numbers),
        "target": sample.target,
        "prompt": sample.prompt,
        "label": sample.label,
        "state": [
            {
                "index": index,
                "expression": item.expression,
                "value": _format_fraction(item.value),
            }
            for index, item in enumerate(sample.state.items)
        ],
        "action": {
            "left_index": sample.action.left_index,
            "operation": sample.action.operation,
            "right_index": sample.action.right_index,
        },
    }


@lru_cache(maxsize=500_000)
def _can_reach_values(values: tuple[Fraction, ...], target: Fraction) -> bool:
    if len(values) == 1:
        return values[0] == target

    count = len(values)
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            left = values[left_index]
            right = values[right_index]
            rest = tuple(
                value
                for index, value in enumerate(values)
                if index not in {left_index, right_index}
            )
            for next_value in _combined_values(left, right):
                if _can_reach_values(_canonical_values((*rest, next_value)), target):
                    return True
    return False


def _combined_values(left: Fraction, right: Fraction) -> tuple[Fraction, ...]:
    values = [left + right, left - right, right - left, left * right]
    if right:
        values.append(left / right)
    if left:
        values.append(right / left)
    return tuple(dict.fromkeys(values))


def _action_sort_key(
    state: ModelToTState,
    action: OperationCandidate,
    target: Fraction,
) -> tuple[float, int, int, int, int]:
    child = apply_candidate(state, action, source="teacher")
    if child is None:
        return (float("inf"), len(state.items), action.left_index, 99, action.right_index)
    closest = min(abs(float(item.value - target)) for item in child.items)
    return (
        closest,
        len(child.items),
        action.left_index,
        OPERATION_ORDER[action.operation],
        action.right_index,
    )


def _state_key(state: ModelToTState) -> tuple[tuple[int, int], ...]:
    return tuple(
        (value.numerator, value.denominator)
        for value in _canonical_values(item.value for item in state.items)
    )


def _canonical_values(values: Iterable[Fraction]) -> tuple[Fraction, ...]:
    return tuple(sorted(values))


def _action_label(action: OperationCandidate) -> str:
    return f"{action.left_index} {action.operation} {action.right_index}"


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _validate_limits(
    candidates_per_state: int,
    max_states_per_example: int,
    max_actions_per_state: int,
) -> None:
    if candidates_per_state < 1:
        raise ValueError("candidates_per_state must be positive")
    if max_states_per_example < 1:
        raise ValueError("max_states_per_example must be positive")
    if max_actions_per_state < 1:
        raise ValueError("max_actions_per_state must be positive")
