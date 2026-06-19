"""Simple verifiable reward used by the future GRPO trainer."""

from game24.verifier import verify_expression


def compute_reward(expression: str | None, numbers: tuple[int, int, int, int]) -> float:
    """Return 1 for a correct answer and 0 otherwise."""

    return float(expression is not None and verify_expression(expression, numbers))
