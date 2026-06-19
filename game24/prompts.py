"""Prompt construction for the Game of 24 task."""


def build_prompt(numbers: tuple[int, int, int, int]) -> str:
    """Build the instruction passed to the chat model."""

    values = ", ".join(map(str, numbers))
    return f"""Use the numbers [{values}] exactly once to make 24.
You may only use +, -, *, / and parentheses.
Think step by step, then reply in this format:
<think>your reasoning</think>
<answer>your expression</answer>"""
