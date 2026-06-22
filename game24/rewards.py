"""Shared rule rewards for strict arithmetic RLVR."""

from collections.abc import Sequence
from dataclasses import dataclass

from game24.parser import parse_response
from game24.verifier import check_expression

FORMAT_REWARD = 0.1
SYNTAX_REWARD = 0.1
NUMBER_REWARD = 0.2
CORRECT_REWARD = 1.0
UNSOLVABLE_ANSWER = "UNSOLVABLE"


@dataclass(frozen=True)
class RewardBreakdown:
    """Individual signals and their sum for one completion."""

    format: float
    syntax: float
    number_usage: float
    correctness: float

    @property
    def total(self) -> float:
        return self.format + self.syntax + self.number_usage + self.correctness


def score_response(
    response: str,
    numbers: Sequence[int],
    target: int = 24,
    solvable: bool | None = True,
) -> RewardBreakdown:
    """Score a full response without granting correctness for malformed output."""

    parsed = parse_response(response)
    if not parsed.valid_format or parsed.answer is None:
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0)

    if parsed.answer == UNSOLVABLE_ANSWER:
        correct = CORRECT_REWARD if solvable is False else 0.0
        return RewardBreakdown(FORMAT_REWARD, 0.0, 0.0, correct)

    result = check_expression(parsed.answer, numbers, target)
    syntax = SYNTAX_REWARD if result.syntax_valid else 0.0
    number_usage = NUMBER_REWARD if result.numbers_valid else 0.0
    correctness = CORRECT_REWARD if result.valid and solvable is not False else 0.0
    return RewardBreakdown(FORMAT_REWARD, syntax, number_usage, correctness)


def compute_reward(
    expression: str | None,
    numbers: Sequence[int],
    target: int = 24,
) -> float:
    """Backward-compatible binary reward for an already extracted expression."""

    if expression is None or expression == UNSOLVABLE_ANSWER:
        return 0.0
    return float(check_expression(expression, numbers, target).valid)
