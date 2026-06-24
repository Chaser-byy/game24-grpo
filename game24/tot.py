"""Tree-search style test-time compute for arithmetic tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from game24.verifier import check_expression


@dataclass(frozen=True)
class ToTStep:
    """One state transition in the arithmetic search tree."""

    left: str
    right: str
    operation: str
    expression: str
    value: str


@dataclass(frozen=True)
class ToTResult:
    """Result of one Tree-of-Thought style arithmetic search."""

    found: bool
    expression: str | None
    nodes_expanded: int
    depth: int
    trace: tuple[ToTStep, ...]
    reason: str


@dataclass(frozen=True)
class _Node:
    values: tuple[Fraction, ...]
    expressions: tuple[str, ...]
    trace: tuple[ToTStep, ...]


def tot_search(
    numbers: Sequence[int],
    target: int = 24,
    *,
    beam_size: int | None = None,
    max_nodes: int = 100_000,
) -> ToTResult:
    """Search over arithmetic-combination thoughts until one expression reaches target.

    A node is a partial arithmetic state: remaining numeric values plus the expression that
    produced each value. Expanding a node combines two remaining values with a legal operator.
    When ``beam_size`` is ``None`` the search is exhaustive for the four-number 24 game.
    """

    if beam_size is not None and beam_size < 1:
        raise ValueError("beam_size must be positive or None")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    start = _Node(
        tuple(Fraction(number) for number in numbers),
        tuple(str(number) for number in numbers),
        (),
    )
    goal = Fraction(target)
    frontier = [start]
    seen = {_state_key(start.values)}
    nodes_expanded = 0

    for depth in range(len(numbers)):
        next_frontier: list[_Node] = []
        for node in frontier:
            if len(node.values) == 1 and node.values[0] == goal:
                expression = node.expressions[0]
                if check_expression(expression, numbers, target).valid:
                    return ToTResult(True, expression, nodes_expanded, depth, node.trace, "found")
            if len(node.values) == 1:
                continue
            nodes_expanded += 1
            if nodes_expanded > max_nodes:
                return ToTResult(
                    False,
                    None,
                    nodes_expanded,
                    depth,
                    (),
                    f"exceeded max_nodes={max_nodes}",
                )
            for child in _expand(node):
                key = _state_key(child.values)
                if key in seen:
                    continue
                seen.add(key)
                if len(child.values) == 1 and child.values[0] == goal:
                    expression = child.expressions[0]
                    if check_expression(expression, numbers, target).valid:
                        return ToTResult(
                            True,
                            expression,
                            nodes_expanded,
                            depth + 1,
                            child.trace,
                            "found",
                        )
                next_frontier.append(child)
        if not next_frontier:
            break
        next_frontier.sort(key=lambda item: _heuristic(item.values, goal))
        frontier = next_frontier[:beam_size] if beam_size is not None else next_frontier

    return ToTResult(False, None, nodes_expanded, len(numbers) - 1, (), "search exhausted")


def _expand(node: _Node) -> list[_Node]:
    children = []
    count = len(node.values)
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            left_value = node.values[left_index]
            right_value = node.values[right_index]
            left_expr = node.expressions[left_index]
            right_expr = node.expressions[right_index]
            rest_values = [
                value
                for index, value in enumerate(node.values)
                if index not in {left_index, right_index}
            ]
            rest_exprs = [
                expression
                for index, expression in enumerate(node.expressions)
                if index not in {left_index, right_index}
            ]
            operations = [
                (left_value + right_value, "+", left_expr, right_expr),
                (left_value - right_value, "-", left_expr, right_expr),
                (right_value - left_value, "-", right_expr, left_expr),
                (left_value * right_value, "*", left_expr, right_expr),
            ]
            if right_value:
                operations.append((left_value / right_value, "/", left_expr, right_expr))
            if left_value:
                operations.append((right_value / left_value, "/", right_expr, left_expr))

            local_seen: set[Fraction] = set()
            for value, operator, first_expr, second_expr in operations:
                if value in local_seen:
                    continue
                local_seen.add(value)
                expression = f"({first_expr}{operator}{second_expr})"
                step = ToTStep(
                    left=first_expr,
                    right=second_expr,
                    operation=operator,
                    expression=expression,
                    value=_format_fraction(value),
                )
                children.append(
                    _Node(
                        tuple(rest_values + [value]),
                        tuple(rest_exprs + [expression]),
                        node.trace + (step,),
                    )
                )
    return children


def _state_key(values: tuple[Fraction, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((value.numerator, value.denominator) for value in values))


def _heuristic(values: tuple[Fraction, ...], target: Fraction) -> tuple[float, int]:
    closest = min(abs(float(value - target)) for value in values)
    return (closest, len(values))


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
