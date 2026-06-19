#!/usr/bin/env python3
"""Evaluate a Qwen baseline on a small Game of 24 JSONL dataset."""

import argparse
import json
from pathlib import Path

from game24.data import load_jsonl
from game24.inference import generate_response, load_qwen, set_seed
from game24.parser import extract_answer
from game24.rewards import compute_reward
from game24.verifier import VerificationResult, check_expression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen Game of 24 baseline")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=20, help="Number of puzzles; 0 uses all")
    parser.add_argument("--output", default="outputs/baseline_20.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.data)
    if args.limit > 0:
        examples = examples[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    tokenizer, model = load_qwen(args.model)
    correct = extracted = legal = reward_sum = 0

    with output.open("w", encoding="utf-8") as file:
        for index, example in enumerate(examples, 1):
            response = generate_response(
                tokenizer,
                model,
                example.numbers,
                max_new_tokens=args.max_new_tokens,
                sample=args.sample,
            )
            expression = extract_answer(response)
            if expression is None:
                result = VerificationResult(False, [], None, "missing or empty <answer> tag")
            else:
                extracted += 1
                result = check_expression(expression, example.numbers)

            is_legal = (
                result.value is not None
                and sorted(result.used_numbers) == sorted(example.numbers)
            )
            reward = compute_reward(expression, example.numbers)
            correct += int(result.valid)
            legal += int(is_legal)
            reward_sum += reward

            record = {
                "example_id": example.example_id,
                "numbers": list(example.numbers),
                "solvable": example.solvable,
                "source": example.source,
                "rank": example.rank,
                "solved_rate": example.solved_rate,
                "model_output": response,
                "expression": expression,
                "extracted": expression is not None,
                "legal": is_legal,
                "correct": result.valid,
                "used_numbers": result.used_numbers,
                "value": str(result.value) if result.value is not None else None,
                "reason": result.reason,
                "reward": reward,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(examples)}] {example.numbers} correct={result.valid}")

    total = len(examples)
    summary = {
        "model": args.model,
        "data": args.data,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "extracted": extracted,
        "extraction_rate": extracted / total if total else 0.0,
        "legal": legal,
        "legal_rate": legal / total if total else 0.0,
        "average_reward": reward_sum / total if total else 0.0,
        "sample": args.sample,
        "seed": args.seed,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")

    print("\nBaseline summary")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Extracted: {extracted}")
    print(f"Legal rate: {summary['legal_rate']:.2%}")
    print(f"Average reward: {summary['average_reward']:.4f}")
    print(f"Details: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
