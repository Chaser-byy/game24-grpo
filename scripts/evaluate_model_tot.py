#!/usr/bin/env python3
"""Evaluate model-guided Tree-of-Thought search."""

import argparse
import json
from pathlib import Path

from game24.data import dataset_fingerprint, load_jsonl
from game24.inference import load_qwen, set_seed
from game24.model_tot import model_tot_search, result_to_record
from game24.rewards import UNSOLVABLE_ANSWER, score_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model-guided ToT")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--candidates-per-state", type=int, default=4)
    parser.add_argument("--branch-samples", type=int, default=2)
    parser.add_argument("--fallback-candidates", type=int, default=0)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0)
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
    totals = {
        "total": 0,
        "found": 0,
        "correct": 0,
        "format": 0,
        "syntax": 0,
        "legal": 0,
        "model_calls": 0,
        "nodes": 0,
        "valid_proposals": 0,
        "invalid_proposals": 0,
        "fallback_expansions": 0,
        "unsolvable": 0,
        "correct_abstention": 0,
        "false_claim": 0,
    }

    with output.open("w", encoding="utf-8") as file:
        for index, example in enumerate(examples, 1):
            result = model_tot_search(
                tokenizer,
                model,
                example.numbers,
                example.target,
                beam_size=args.beam_size,
                candidates_per_state=args.candidates_per_state,
                branch_samples=args.branch_samples,
                max_depth=args.max_depth,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                fallback_candidates=args.fallback_candidates,
            )
            answer = result.expression if result.expression is not None else UNSOLVABLE_ANSWER
            response = (
                "<think>Model-guided tree search proposed operations; the program verified "
                f"state updates.</think><answer>{answer}</answer>"
            )
            reward = score_response(response, example.numbers, example.target, example.solvable)
            correct = reward.correctness == 1.0
            totals["total"] += 1
            totals["found"] += int(result.found)
            totals["correct"] += int(correct)
            totals["format"] += int(reward.format > 0)
            totals["syntax"] += int(reward.syntax > 0)
            totals["legal"] += int(reward.number_usage > 0)
            totals["model_calls"] += result.model_calls
            totals["nodes"] += result.nodes_expanded
            totals["valid_proposals"] += result.valid_proposals
            totals["invalid_proposals"] += result.invalid_proposals
            totals["fallback_expansions"] += result.fallback_expansions
            if example.solvable is False:
                totals["unsolvable"] += 1
                totals["correct_abstention"] += int(answer == UNSOLVABLE_ANSWER)
                totals["false_claim"] += int(result.expression is not None)
            record = {
                "example_id": example.example_id,
                "numbers": list(example.numbers),
                "target": example.target,
                "solvable": example.solvable,
                "correct": correct,
                "reward": {
                    "format": reward.format,
                    "syntax": reward.syntax,
                    "number_usage": reward.number_usage,
                    "correctness": reward.correctness,
                    "total": reward.total,
                },
                "tot": result_to_record(result),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(examples)}] {example.numbers} "
                f"found={result.found} correct={correct} calls={result.model_calls}"
            )

    summary = _summary(args, examples, totals)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _summary(args: argparse.Namespace, examples, totals: dict[str, int]) -> dict:
    total = totals["total"]
    return {
        "method": "model_guided_tot",
        "model": args.model,
        "data": args.data,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "total": total,
        "beam_size": args.beam_size,
        "candidates_per_state": args.candidates_per_state,
        "branch_samples": args.branch_samples,
        "fallback_candidates": args.fallback_candidates,
        "found": totals["found"],
        "found_rate": _rate(totals["found"], total),
        "correct": totals["correct"],
        "accuracy": _rate(totals["correct"], total),
        "strict_format_rate": _rate(totals["format"], total),
        "syntax_rate": _rate(totals["syntax"], total),
        "legal_number_rate": _rate(totals["legal"], total),
        "unsolvable_total": totals["unsolvable"],
        "correct_abstention_rate": _rate(
            totals["correct_abstention"], totals["unsolvable"]
        ),
        "false_claim_rate": _rate(totals["false_claim"], totals["unsolvable"]),
        "avg_model_calls": totals["model_calls"] / total if total else None,
        "avg_nodes_expanded": totals["nodes"] / total if total else None,
        "avg_valid_proposals": totals["valid_proposals"] / total if total else None,
        "avg_invalid_proposals": totals["invalid_proposals"] / total if total else None,
        "avg_fallback_expansions": totals["fallback_expansions"] / total if total else None,
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
