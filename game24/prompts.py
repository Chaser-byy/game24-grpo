"""Prompt construction for the Game of 24 task."""


def build_prompt(numbers: tuple[int, int, int, int]) -> str:
    """Build the instruction passed to the chat model."""

    values = ", ".join(map(str, numbers))
    return f"""Use the numbers [{values}] exactly once to make 24.
You may only use +, -, *, / and parentheses.
Think step by step. Before answering, check that all four numbers are each used exactly once
and that the expression equals 24.

Output exactly these two blocks and nothing else:
<think>one short sentence with your check</think>
<answer>one arithmetic expression</answer>"""
