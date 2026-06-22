"""Parse the strict R1-style response protocol used by training and evaluation."""

import re
from dataclasses import dataclass

RESPONSE_PATTERN = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedResponse:
    """Structured result of parsing one model completion."""

    valid_format: bool
    think: str | None
    answer: str | None
    reason: str


def parse_response(text: str) -> ParsedResponse:
    """Require exactly one non-empty think block followed by one answer block."""

    match = RESPONSE_PATTERN.fullmatch(text)
    if match is None:
        return ParsedResponse(False, None, None, "response does not match the R1 XML protocol")

    think = match.group("think").strip()
    answer = match.group("answer").strip()
    if not think:
        return ParsedResponse(False, None, None, "think block is empty")
    if not answer:
        return ParsedResponse(False, think, None, "answer block is empty")
    if "<" in think or ">" in think or "<" in answer or ">" in answer:
        return ParsedResponse(False, None, None, "nested or additional tags are not allowed")
    return ParsedResponse(True, think, answer, "ok")


def extract_answer(text: str, *, strict: bool = True) -> str | None:
    """Extract an answer, using the strict full-response protocol by default."""

    if strict:
        return parse_response(text).answer

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match is None:
        return None
    answer = match.group(1).strip()
    return answer or None
