#!/usr/bin/env python3
"""Run the current data-to-reward pipeline with a mock model response."""

from pathlib import Path

from game24.data import load_jsonl
from game24.parser import extract_answer
from game24.prompts import build_prompt
from game24.rewards import compute_reward
from game24.verifier import verify_expression


def main() -> None:
    example = load_jsonl(Path(__file__).parents[1] / "data" / "sample.jsonl")[0]
    prompt = build_prompt(example.numbers)

    # Replace this string with Qwen generation later.
    model_output = "<think>Divide 6 by one quarter.</think><answer>6/(1-3/4)</answer>"
    answer = extract_answer(model_output)

    print("Numbers:", example.numbers)
    print("Prompt:\n", prompt)
    print("Model output:", model_output)
    print("Answer:", answer)
    print("Valid:", answer is not None and verify_expression(answer, example.numbers))
    print("Reward:", compute_reward(answer, example.numbers))


if __name__ == "__main__":
    main()
