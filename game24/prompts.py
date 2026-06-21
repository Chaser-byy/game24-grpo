"""Prompt construction for the Game of 24 task."""


def build_prompt(numbers: tuple[int, int, int, int]) -> str:
    """Build the instruction passed to the chat model."""

    values = ", ".join(map(str, numbers))
    return f"""Solve the Game24 puzzle with the numbers [{values}].
Use every given number exactly once. Use only +, -, *, / and parentheses.
Copy the four input integers exactly; never replace, omit, or invent a number.
Before answering, verify the number usage and that the expression equals 24.

Your response must follow this XML protocol:
- Start with the literal tag <think>, write one brief verification sentence, then close </think>.
- Immediately open <answer>, write only the arithmetic expression, then close </answer>.

The response must begin with <think> and end with </answer>. Write nothing outside the tags.
Do not copy these instructions or add an equals sign.
Keep the entire response under 40 words."""
