"""TRL-compatible reward functions backed by the shared strict scorer."""

from typing import Any

from game24.rewards import score_response


def completion_text(completion: Any) -> str:
    """Read text from either standard or conversational TRL completions."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return ""


def _breakdowns(
    completions: list[Any],
    numbers: list[list[int]],
    target: list[int],
    solvable: list[bool | None],
):
    return [
        score_response(completion_text(completion), expected, goal, can_solve)
        for completion, expected, goal, can_solve in zip(
            completions, numbers, target, solvable, strict=True
        )
    ]


def strict_format_reward(
    completions: list[Any],
    numbers: list[list[int]],
    target: list[int],
    solvable: list[bool | None],
    **_: Any,
) -> list[float]:
    """Reward only a complete, strict R1 XML response."""

    return [item.format for item in _breakdowns(completions, numbers, target, solvable)]


def syntax_reward(
    completions: list[Any],
    numbers: list[list[int]],
    target: list[int],
    solvable: list[bool | None],
    **_: Any,
) -> list[float]:
    """Reward an expression that passes the strict lexer and parser."""

    return [item.syntax for item in _breakdowns(completions, numbers, target, solvable)]


def number_usage_reward(
    completions: list[Any],
    numbers: list[list[int]],
    target: list[int],
    solvable: list[bool | None],
    **_: Any,
) -> list[float]:
    """Reward exact multiset use of the provided numbers."""

    return [item.number_usage for item in _breakdowns(completions, numbers, target, solvable)]


def correctness_reward(
    completions: list[Any],
    numbers: list[list[int]],
    target: list[int],
    solvable: list[bool | None],
    **_: Any,
) -> list[float]:
    """Return a strictly binary verifiable correctness reward."""

    return [item.correctness for item in _breakdowns(completions, numbers, target, solvable)]


REWARD_FUNCTIONS = [
    strict_format_reward,
    syntax_reward,
    number_usage_reward,
    correctness_reward,
]

ACCURACY_REWARD_FUNCTIONS = [
    syntax_reward,
    number_usage_reward,
    correctness_reward,
]

CORRECTNESS_ONLY_REWARD_FUNCTIONS = [
    correctness_reward,
]

REWARD_MODES = {
    "default": REWARD_FUNCTIONS,
    "accuracy": ACCURACY_REWARD_FUNCTIONS,
    "correctness": CORRECTNESS_ONLY_REWARD_FUNCTIONS,
}


def get_reward_functions(mode: str):
    """Return the reward functions for one named training objective."""

    try:
        return REWARD_MODES[mode]
    except KeyError as error:
        raise ValueError(f"unknown reward mode: {mode}") from error
