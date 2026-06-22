"""Exact dynamic-programming solver used to label and audit arithmetic tasks."""

from collections.abc import Sequence
from fractions import Fraction


def find_solution(numbers: Sequence[int], target: int = 24) -> str | None:
    """Return one expression reaching target, or ``None`` when none exists."""

    count = len(numbers)
    values: dict[int, dict[Fraction, str]] = {
        1 << index: {Fraction(number): str(number)} for index, number in enumerate(numbers)
    }

    for size in range(2, count + 1):
        for mask in range(1, 1 << count):
            if mask.bit_count() != size:
                continue
            results: dict[Fraction, str] = {}
            left_mask = (mask - 1) & mask
            while left_mask:
                right_mask = mask ^ left_mask
                if right_mask and left_mask < right_mask:
                    for left_value, left_expr in values[left_mask].items():
                        for right_value, right_expr in values[right_mask].items():
                            candidates = [
                                (left_value + right_value, f"({left_expr}+{right_expr})"),
                                (left_value - right_value, f"({left_expr}-{right_expr})"),
                                (right_value - left_value, f"({right_expr}-{left_expr})"),
                                (left_value * right_value, f"({left_expr}*{right_expr})"),
                            ]
                            if right_value:
                                candidates.append(
                                    (left_value / right_value, f"({left_expr}/{right_expr})")
                                )
                            if left_value:
                                candidates.append(
                                    (right_value / left_value, f"({right_expr}/{left_expr})")
                                )
                            for value, expression in candidates:
                                results.setdefault(value, expression)
                left_mask = (left_mask - 1) & mask
            values[mask] = results

    return values[(1 << count) - 1].get(Fraction(target))


def is_solvable(numbers: Sequence[int], target: int = 24) -> bool:
    """Return whether the exact solver can reach the target."""

    return find_solution(numbers, target) is not None
