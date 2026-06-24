"""Model-guided Tree-of-Thought search with program-verified state updates."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

from game24.verifier import check_expression

OPERATION_PATTERN = re.compile(r"(?P<left>\d+)\s*(?P<op>[+\-*/])\s*(?P<right>\d+)")


@dataclass(frozen=True)
class ThoughtItem:
    """One remaining value/expression in a ToT state."""

    value: Fraction
    expression: str


@dataclass(frozen=True)
class OperationCandidate:
    """A model-proposed operation over two state indices."""

    left_index: int
    operation: str
    right_index: int
    raw_text: str


@dataclass(frozen=True)
class ModelToTStep:
    """One validated state transition."""

    before: tuple[str, ...]
    operation: str
    expression: str
    value: str
    after: tuple[str, ...]
    proposal: str
    source: str


@dataclass(frozen=True)
class ModelToTState:
    """Structured arithmetic state maintained by the program."""

    items: tuple[ThoughtItem, ...]
    trace: tuple[ModelToTStep, ...] = ()


@dataclass(frozen=True)
class ModelToTResult:
    """Result and diagnostics from model-guided ToT."""

    found: bool
    expression: str | None
    nodes_expanded: int
    model_calls: int
    valid_proposals: int
    invalid_proposals: int
    fallback_expansions: int
    trace: tuple[ModelToTStep, ...]
    reason: str


def initial_state(numbers: Sequence[int]) -> ModelToTState:
    """Create the initial arithmetic state."""

    return ModelToTState(tuple(ThoughtItem(Fraction(number), str(number)) for number in numbers))


def parse_operation_candidates(text: str, item_count: int) -> list[OperationCandidate]:
    """Parse candidate operations like ``0 * 1`` from model text."""

    candidates = []
    seen = set()
    for match in OPERATION_PATTERN.finditer(text):
        left = int(match.group("left"))
        right = int(match.group("right"))
        op = match.group("op")
        key = (left, op, right)
        if key in seen:
            continue
        seen.add(key)
        if 0 <= left < item_count and 0 <= right < item_count and left != right:
            candidates.append(OperationCandidate(left, op, right, match.group(0)))
    return candidates


def apply_candidate(
    state: ModelToTState,
    candidate: OperationCandidate,
    *,
    source: str = "model",
) -> ModelToTState | None:
    """Execute one candidate exactly, rejecting invalid operations."""

    left = state.items[candidate.left_index]
    right = state.items[candidate.right_index]
    if candidate.operation == "+":
        value = left.value + right.value
        expression = f"({left.expression}+{right.expression})"
    elif candidate.operation == "-":
        value = left.value - right.value
        expression = f"({left.expression}-{right.expression})"
    elif candidate.operation == "*":
        value = left.value * right.value
        expression = f"({left.expression}*{right.expression})"
    elif candidate.operation == "/":
        if right.value == 0:
            return None
        value = left.value / right.value
        expression = f"({left.expression}/{right.expression})"
    else:
        return None

    rest = tuple(
        item
        for index, item in enumerate(state.items)
        if index not in {candidate.left_index, candidate.right_index}
    )
    child_items = rest + (ThoughtItem(value, expression),)
    step = ModelToTStep(
        before=tuple(_format_item(item) for item in state.items),
        operation=f"{candidate.left_index} {candidate.operation} {candidate.right_index}",
        expression=expression,
        value=_format_fraction(value),
        after=tuple(_format_item(item) for item in child_items),
        proposal=candidate.raw_text,
        source=source,
    )
    return ModelToTState(child_items, state.trace + (step,))


def model_tot_search(
    tokenizer: Any,
    model: Any,
    numbers: Sequence[int],
    target: int = 24,
    *,
    beam_size: int = 5,
    candidates_per_state: int = 4,
    branch_samples: int = 2,
    max_depth: int | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    fallback_candidates: int = 0,
) -> ModelToTResult:
    """Run model-guided ToT while the program validates all arithmetic."""

    if beam_size < 1 or candidates_per_state < 1 or branch_samples < 1:
        raise ValueError("beam_size, candidates_per_state, and branch_samples must be positive")
    if fallback_candidates < 0:
        raise ValueError("fallback_candidates cannot be negative")
    max_depth = max_depth if max_depth is not None else len(numbers) - 1
    goal = Fraction(target)
    frontier = [initial_state(numbers)]
    seen = {_state_key(frontier[0])}
    nodes_expanded = 0
    model_calls = 0
    valid_proposals = 0
    invalid_proposals = 0
    fallback_expansions = 0

    for _depth in range(max_depth):
        next_frontier: list[ModelToTState] = []
        for state in frontier:
            if _is_goal(state, numbers, target):
                return _success(
                    state,
                    nodes_expanded,
                    model_calls,
                    valid_proposals,
                    invalid_proposals,
                    fallback_expansions,
                )
            if len(state.items) == 1:
                continue
            nodes_expanded += 1
            proposals = _generate_proposals(
                tokenizer,
                model,
                state,
                target,
                candidates_per_state=candidates_per_state,
                branch_samples=branch_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            model_calls += 1
            parsed: list[OperationCandidate] = []
            for text in proposals:
                parsed.extend(parse_operation_candidates(text, len(state.items)))

            children = []
            for candidate in parsed:
                child = apply_candidate(state, candidate, source="model")
                if child is None:
                    invalid_proposals += 1
                    continue
                valid_proposals += 1
                key = _state_key(child)
                if key in seen:
                    continue
                seen.add(key)
                if _is_goal(child, numbers, target):
                    return _success(
                        child,
                        nodes_expanded,
                        model_calls,
                        valid_proposals,
                        invalid_proposals,
                        fallback_expansions,
                    )
                children.append(child)

            if fallback_candidates and len(children) < candidates_per_state:
                for child in _heuristic_children(state, target, fallback_candidates):
                    fallback_expansions += 1
                    key = _state_key(child)
                    if key in seen:
                        continue
                    seen.add(key)
                    if _is_goal(child, numbers, target):
                        return _success(
                            child,
                            nodes_expanded,
                            model_calls,
                            valid_proposals,
                            invalid_proposals,
                            fallback_expansions,
                        )
                    children.append(child)

            next_frontier.extend(children)

        if not next_frontier:
            break
        next_frontier.sort(key=lambda item: _heuristic(item, goal))
        frontier = next_frontier[:beam_size]

    return ModelToTResult(
        False,
        None,
        nodes_expanded,
        model_calls,
        valid_proposals,
        invalid_proposals,
        fallback_expansions,
        (),
        "search exhausted",
    )


def result_to_record(result: ModelToTResult) -> dict[str, Any]:
    """Serialize a result for JSONL outputs."""

    return {
        "found": result.found,
        "expression": result.expression,
        "nodes_expanded": result.nodes_expanded,
        "model_calls": result.model_calls,
        "valid_proposals": result.valid_proposals,
        "invalid_proposals": result.invalid_proposals,
        "fallback_expansions": result.fallback_expansions,
        "reason": result.reason,
        "trace": [asdict(step) for step in result.trace],
    }


def _generate_proposals(
    tokenizer: Any,
    model: Any,
    state: ModelToTState,
    target: int,
    *,
    candidates_per_state: int,
    branch_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    prompt = build_tot_prompt(state, target, candidates_per_state)
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    generated = model.generate(
        **model_inputs,
        do_sample=True,
        num_return_sequences=branch_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_length = model_inputs.input_ids.shape[1]
    return tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)


def build_tot_prompt(state: ModelToTState, target: int, candidates_per_state: int = 4) -> str:
    """Build the exact prompt used by model-guided ToT proposal generation."""

    rows = "\n".join(
        f"{index}: {item.expression} = {_format_fraction(item.value)}"
        for index, item in enumerate(state.items)
    )
    return f"""We are solving the 24-point game with exact arithmetic.
Target: {target}
Current state:
{rows}

Propose up to {candidates_per_state} promising next operations.
Use only existing state indices. Choose two different indices and one operator from + - * /.
Return one operation per line in exactly this form:
0 * 1
2 - 0
Do not explain or calculate; the program will verify arithmetic."""


_build_tot_prompt = build_tot_prompt


def _heuristic_children(state: ModelToTState, target: int, limit: int) -> list[ModelToTState]:
    candidates = []
    for left_index in range(len(state.items)):
        for right_index in range(len(state.items)):
            if left_index == right_index:
                continue
            for op in ("+", "-", "*", "/"):
                candidate = OperationCandidate(
                    left_index,
                    op,
                    right_index,
                    f"{left_index} {op} {right_index}",
                )
                child = apply_candidate(state, candidate, source="heuristic")
                if child is not None:
                    candidates.append(child)
    candidates.sort(key=lambda item: _heuristic(item, Fraction(target)))
    return candidates[:limit]


def _success(
    state: ModelToTState,
    nodes_expanded: int,
    model_calls: int,
    valid_proposals: int,
    invalid_proposals: int,
    fallback_expansions: int,
) -> ModelToTResult:
    return ModelToTResult(
        True,
        state.items[0].expression,
        nodes_expanded,
        model_calls,
        valid_proposals,
        invalid_proposals,
        fallback_expansions,
        state.trace,
        "found",
    )


def _is_goal(state: ModelToTState, numbers: Sequence[int], target: int) -> bool:
    return (
        len(state.items) == 1
        and check_expression(state.items[0].expression, numbers, target).valid
    )


def _state_key(state: ModelToTState) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((item.value.numerator, item.value.denominator) for item in state.items))


def _heuristic(state: ModelToTState, target: Fraction) -> tuple[float, int]:
    closest = min(abs(float(item.value - target)) for item in state.items)
    return (closest, len(state.items))


def _format_item(item: ThoughtItem) -> str:
    return f"{item.expression}={_format_fraction(item.value)}"


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
