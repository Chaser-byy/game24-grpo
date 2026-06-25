"""Verified multi-step arithmetic trajectories for SFT warm starts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction

from game24.verifier import check_expression


@dataclass(frozen=True)
class TrajectoryStep:
    """One verified arithmetic transition."""

    before: tuple[str, ...]
    left: str
    right: str
    operation: str
    expression: str
    value: str
    after: tuple[str, ...]


@dataclass(frozen=True)
class SolutionTrajectory:
    """A complete trace from input numbers to a verified target expression."""

    numbers: tuple[int, ...]
    target: int
    expression: str
    steps: tuple[TrajectoryStep, ...]

    def as_dict(self) -> dict:
        return {
            "numbers": list(self.numbers),
            "target": self.target,
            "expression": self.expression,
            "steps": [asdict(step) for step in self.steps],
        }


@dataclass(frozen=True)
class _StateItem:
    value: Fraction
    expression: str


def find_trajectory(numbers: Sequence[int], target: int = 24) -> SolutionTrajectory | None:
    """Return one verified step trajectory, or ``None`` if no solution is found."""

    trajectories = enumerate_trajectories(numbers, target, max_solutions=1)
    return trajectories[0] if trajectories else None


def enumerate_trajectories(
    numbers: Sequence[int],
    target: int = 24,
    *,
    max_solutions: int = 4,
    max_nodes: int = 100_000,
) -> list[SolutionTrajectory]:
    """Enumerate verified solution paths using exact rational arithmetic."""

    if max_solutions < 1:
        raise ValueError("max_solutions must be positive")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")

    start = tuple(_StateItem(Fraction(number), str(number)) for number in numbers)
    results: list[SolutionTrajectory] = []
    seen_expressions: set[str] = set()
    nodes = 0

    def search(state: tuple[_StateItem, ...], trace: tuple[TrajectoryStep, ...]) -> None:
        nonlocal nodes
        if len(results) >= max_solutions or nodes >= max_nodes:
            return
        nodes += 1
        if len(state) == 1:
            expression = state[0].expression
            if (
                expression not in seen_expressions
                and check_expression(expression, numbers, target).valid
            ):
                seen_expressions.add(expression)
                results.append(
                    SolutionTrajectory(
                        numbers=tuple(int(number) for number in numbers),
                        target=target,
                        expression=expression,
                        steps=trace,
                    )
                )
            return

        children = _expand_state(state)
        children.sort(key=lambda item: _trajectory_heuristic(item[0], Fraction(target)))
        for child_state, step in children:
            search(child_state, trace + (step,))
            if len(results) >= max_solutions:
                return

    search(start, ())
    return results


def trajectory_to_response(trajectory: SolutionTrajectory) -> str:
    """Render a compact, stateful R1-style SFT response."""

    lines = []
    for index, step in enumerate(trajectory.steps, 1):
        before = ", ".join(step.before)
        after = ", ".join(step.after)
        lines.append(
            f"Step {index}: state [{before}]. Combine {step.left} {step.operation} "
            f"{step.right} = {step.value}, giving {step.expression}. Remaining [{after}]."
        )
    lines.append(
        f"Check: final expression {trajectory.expression} evaluates to {trajectory.target} "
        "and uses each original number exactly once."
    )
    think = "\n".join(lines)
    return f"<think>{think}</think>\n<answer>{trajectory.expression}</answer>"


def trajectory_to_direct_tot_response(trajectory: SolutionTrajectory) -> str:
    """Render a direct-answer response that teaches internal stateful ToT reasoning."""

    lines = [
        (
            "I will keep the remaining expressions as a state and combine two "
            "entries at a time until one expression reaches the target."
        )
    ]
    for index, step in enumerate(trajectory.steps, 1):
        before = ", ".join(step.before)
        after = ", ".join(step.after)
        lines.append(f"State {index - 1}: [{before}].")
        lines.append(
            f"Step {index}: combine {step.left} {step.operation} {step.right} "
            f"to get {step.expression} = {step.value}."
        )
        lines.append(f"New state: [{after}].")
    lines.append(
        f"The final expression is {trajectory.expression}, which evaluates to "
        f"{trajectory.target} and uses each original number exactly once."
    )
    think = "\n".join(lines)
    return f"<think>{think}</think>\n<answer>{trajectory.expression}</answer>"


def _expand_state(
    state: tuple[_StateItem, ...],
) -> list[tuple[tuple[_StateItem, ...], TrajectoryStep]]:
    children = []
    count = len(state)
    before = tuple(_format_item(item) for item in state)
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            left = state[left_index]
            right = state[right_index]
            rest = tuple(
                item
                for index, item in enumerate(state)
                if index not in {left_index, right_index}
            )
            operations = [
                (left.value + right.value, "+", left.expression, right.expression),
                (left.value - right.value, "-", left.expression, right.expression),
                (right.value - left.value, "-", right.expression, left.expression),
                (left.value * right.value, "*", left.expression, right.expression),
            ]
            if right.value:
                operations.append(
                    (left.value / right.value, "/", left.expression, right.expression)
                )
            if left.value:
                operations.append(
                    (right.value / left.value, "/", right.expression, left.expression)
                )

            local_seen: set[Fraction] = set()
            for value, operator, first_expr, second_expr in operations:
                if value in local_seen:
                    continue
                local_seen.add(value)
                expression = f"({first_expr}{operator}{second_expr})"
                result = _StateItem(value, expression)
                child_state = rest + (result,)
                step = TrajectoryStep(
                    before=before,
                    left=first_expr,
                    right=second_expr,
                    operation=operator,
                    expression=expression,
                    value=_format_fraction(value),
                    after=tuple(_format_item(item) for item in child_state),
                )
                children.append((child_state, step))
    return children


def _trajectory_heuristic(state: tuple[_StateItem, ...], target: Fraction) -> tuple[float, int]:
    closest = min(abs(float(item.value - target)) for item in state)
    return (closest, len(state))


def _format_item(item: _StateItem) -> str:
    return f"{item.expression}={_format_fraction(item.value)}"


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
