#!/usr/bin/env python3
"""Run Qwen2.5-Instruct on one Game of 24 puzzle."""

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from game24.parser import extract_answer
from game24.prompts import build_prompt
from game24.rewards import compute_reward
from game24.verifier import verify_expression


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numbers = tuple(args.numbers)
    if any(number < 1 or number > 13 for number in numbers):
        raise SystemExit("--numbers must contain four integers between 1 and 13")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
    )
    model.to("cuda")
    model.eval()

    messages = [{"role": "user", "content": build_prompt(numbers)}]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    # model.generate returns prompt tokens followed by newly generated tokens.
    new_token_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids, strict=True)
    ]
    response = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]

    expression = extract_answer(response)
    valid = expression is not None and verify_expression(expression, numbers)
    reward = compute_reward(expression, numbers)

    print("Numbers:", numbers)
    print("Model response:\n", response)
    print("Expression:", expression)
    print("Valid:", valid)
    print("Reward:", reward)


if __name__ == "__main__":
    main()
