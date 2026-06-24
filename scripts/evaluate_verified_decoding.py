#!/usr/bin/env python3
"""Evaluate best-of-N verified decoding, optionally with exact-solver fallback."""

import argparse
import json
from pathlib import Path
from typing import Any

from game24.data import dataset_fingerprint, load_jsonl
from game24.evaluation import _score_attempt, _wilson_interval
from game24.inference import generate_responses, load_qwen, set_seed
from game24.sft import build_sft_response
from game24.solver import find_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate verifier-selected best-of-N decoding")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--solver-fallback",
        action="store_true",
        help=(
            "If no model candidate verifies, fill in an exact DP solver answer "
            "as an oracle fallback"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")

    examples = load_jsonl(args.data)
    if args.limit > 0:
        examples = examples[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    tokenizer, model = load_qwen(args.model)

    totals = {
        "total": 0,
        "first_correct": 0,
        "model_any_correct": 0,
        "selected_correct": 0,
        "solver_fallback_used": 0,
        "selected_format": 0,
        "selected_syntax": 0,
        "selected_legal": 0,
    }

    with output.open("w", encoding="utf-8") as file:
        for index, example in enumerate(examples, 1):
            responses = generate_responses(
                tokenizer,
                model,
                example.numbers,
                example.target,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.num_samples,
                sample=args.num_samples > 1,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            attempts = [_score_attempt(response, example) for response in responses]
            first = attempts[0]
            correct_indices = [i for i, attempt in enumerate(attempts) if attempt["correct"]]
            selected_index = correct_indices[0] if correct_indices else 0
            selected = attempts[selected_index]
            selected_source = "model_verified" if correct_indices else "model_first"
            solver_expression = None

            if args.solver_fallback and not selected["correct"]:
                solver_expression = find_solution(example.numbers, example.target)
                if solver_expression is not None:
                    fallback_response = build_sft_response(solver_expression, example.target)
                    selected = _score_attempt(fallback_response, example)
                    selected_index = None
                    selected_source = "solver_fallback"
                    totals["solver_fallback_used"] += 1

            totals["total"] += 1
            totals["first_correct"] += int(first["correct"])
            totals["model_any_correct"] += int(bool(correct_indices))
            totals["selected_correct"] += int(selected["correct"])
            totals["selected_format"] += int(selected["format_valid"])
            totals["selected_syntax"] += int(selected["syntax_valid"])
            totals["selected_legal"] += int(selected["numbers_valid"])

            record = {
                "example_id": example.example_id,
                "numbers": list(example.numbers),
                "target": example.target,
                "solvable": example.solvable,
                "first_correct": first["correct"],
                f"model_pass_at_{args.num_samples}": bool(correct_indices),
                "selected_correct": selected["correct"],
                "selected_index": selected_index,
                "selected_source": selected_source,
                "selected_attempt": selected,
                "solver_expression": solver_expression,
                "attempts": attempts,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(examples)}] {example.numbers} "
                f"first={first['correct']} pass={bool(correct_indices)} "
                f"selected={selected['correct']} source={selected_source}"
            )

    summary = _summary(args, examples, totals)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _summary(
    args: argparse.Namespace,
    examples: list[Any],
    totals: dict[str, int],
) -> dict[str, Any]:
    total = totals["total"]
    return {
        "model": args.model,
        "data": args.data,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "total": total,
        "num_samples": args.num_samples,
        "solver_fallback": args.solver_fallback,
        "first_correct": totals["first_correct"],
        "first_accuracy": _rate(totals["first_correct"], total),
        f"model_pass_at_{args.num_samples}": _rate(totals["model_any_correct"], total),
        f"model_pass_at_{args.num_samples}_ci95": _wilson_interval(
            totals["model_any_correct"], total
        ),
        "selected_correct": totals["selected_correct"],
        "selected_accuracy": _rate(totals["selected_correct"], total),
        "selected_accuracy_ci95": _wilson_interval(totals["selected_correct"], total),
        "solver_fallback_used": totals["solver_fallback_used"],
        "selected_strict_format_rate": _rate(totals["selected_format"], total),
        "selected_syntax_rate": _rate(totals["selected_syntax"], total),
        "selected_legal_number_rate": _rate(totals["selected_legal"], total),
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


if __name__ == "__main__":
    main()
