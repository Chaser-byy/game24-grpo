"""Prompt construction for Game24 and Countdown-style arithmetic tasks."""

from collections.abc import Sequence


def build_prompt(
    numbers: Sequence[int],
    target: int = 24,
    *,
    allow_unsolvable: bool = True,
) -> str:
    """Build the common strict RLVR prompt."""

    values = ", ".join(map(str, numbers))
    unsolvable_rule = (
        "If and only if no legal expression exists, write UNSOLVABLE in the answer tag."
        if allow_unsolvable
        else "A legal solution exists; do not answer UNSOLVABLE."
    )
    return f"""Reach the target {target} using exactly the numbers [{values}].
Use every input number exactly once. Use only +, -, *, / and parentheses.
Do not concatenate digits, introduce constants, or use an equals sign.
{unsolvable_rule}

Return exactly this XML structure and nothing else:
<think>Reason through candidate operations and verify the final result.</think>
<answer>one arithmetic expression, or UNSOLVABLE</answer>

Keep the reasoning focused, but include enough detail to check every number and the target."""
