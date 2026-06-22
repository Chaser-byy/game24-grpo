#!/usr/bin/env python3
"""Run Qwen2.5-Instruct on one Game of 24 puzzle."""

import argparse

from game24.inference import generate_response, load_qwen
from game24.parser import parse_response
from game24.rewards import score_response
from game24.verifier import VerificationResult, check_expression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve one Game of 24 puzzle with Qwen")
    parser.add_argument(
        "--model",
        required=True,
        help="Local model directory or Hugging Face model name",
    )
    parser.add_argument(
        "--numbers",
        nargs=4,
        type=int,
        required=True,
        metavar=("A", "B", "C", "D"),
        help="Four integers between 1 and 13",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--target", type=int, default=24)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling with temperature=0.7 and top_p=0.9",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numbers = tuple(args.numbers)
    if any(number < 1 or number > 13 for number in numbers):
        raise SystemExit("--numbers must contain four integers between 1 and 13")

    tokenizer, model = load_qwen(args.model)
    response = generate_response(
        tokenizer,
        model,
        numbers,
        args.target,
        max_new_tokens=args.max_new_tokens,
        sample=args.sample,
    )

    parsed = parse_response(response)
    expression = parsed.answer
    if expression is None:
        result = VerificationResult(False, [], None, "missing or empty <answer> tag")
    else:
        result = check_expression(expression, numbers, args.target)
    reward = score_response(response, numbers, args.target, True)

    print("Numbers:", numbers)
    print("Model response:\n", response)
    print("Expression:", expression)
    print("Used numbers:", result.used_numbers)
    print("Value:", result.value)
    print("Valid:", result.valid)
    print("Reason:", result.reason)
    print("Strict format:", parsed.valid_format)
    print("Reward:", reward.total, reward)


if __name__ == "__main__":
    main()
