"""Extract the final expression from model output."""

import re


def extract_answer(text: str) -> str | None:
    """Return the expression inside the first non-empty answer tag."""

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    answer = match.group(1).strip()
    answer = re.sub(r"\s*=\s*24\s*$", "", answer).strip()
    return answer or None
