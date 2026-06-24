#!/usr/bin/env python3
"""Evaluate programmatic Tree-of-Thought arithmetic search."""

import argparse
import json
from pathlib import Path

from game24.data import dataset_fingerprint, load_jsonl
from game24.rewards import UNSOLVABLE_ANSWER, score_response
from game24.tot import tot_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tree-of-Thought arithmetic search")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beam-size", type=int, default=0, help="0 means exhaustive search")
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beam_size = args.beam_size or None
    examples = load_jsonl(args.data)
    if args.limit > 0:
        examples = examples[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "total": 0,
        "found": 0,
        "correct": 0,
        "format": 0,
        "syntax": 0,
        "legal": 0,
        "nodes": 0,
    }
    with output.open("w", encoding="utf-8") as file:
        for index, example in enumerate(examples, 1):
            result = tot_search(
                example.numbers,
                example.target,
                beam_size=beam_size,
                max_nodes=args.max_nodes,
            )
            answer = result.expression if result.expression is not None else UNSOLVABLE_ANSWER
            response = (
                "<think>Tree search combined pairs of remaining expressions and verified the "
                f"final result.</think><answer>{answer}</answer>"
            )
            reward = score_response(response, example.numbers, example.target, example.solvable)
            totals["total"] += 1
            totals["found"] += int(result.found)
            totals["correct"] += int(reward.correctness == 1.0)
            totals["format"] += int(reward.format > 0)
            totals["syntax"] += int(reward.syntax > 0)
            totals["legal"] += int(reward.number_usage > 0)
            totals["nodes"] += result.nodes_expanded
            record = {
                "example_id": example.example_id,
                "numbers": list(example.numbers),
                "target": example.target,
                "solvable": example.solvable,
                "found": result.found,
                "expression": result.expression,
                "correct": reward.correctness == 1.0,
                "nodes_expanded": result.nodes_expanded,
                "depth": result.depth,
                "reason": result.reason,
                "reward": {
                    "format": reward.format,
                    "syntax": reward.syntax,
                    "number_usage": reward.number_usage,
                    "correctness": reward.correctness,
                    "total": reward.total,
                },
            }
            if args.include_trace:
                record["trace"] = [step.__dict__ for step in result.trace]
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(examples)}] {example.numbers} "
                f"found={result.found} correct={reward.correctness == 1.0} "
                f"nodes={result.nodes_expanded}"
            )

    total = totals["total"]
    summary = {
        "method": "programmatic_tot_exhaustive" if beam_size is None else "programmatic_tot_beam",
        "data": args.data,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "total": total,
        "beam_size": beam_size,
        "max_nodes": args.max_nodes,
        "found": totals["found"],
        "found_rate": _rate(totals["found"], total),
        "correct": totals["correct"],
        "accuracy": _rate(totals["correct"], total),
        "strict_format_rate": _rate(totals["format"], total),
        "syntax_rate": _rate(totals["syntax"], total),
        "legal_number_rate": _rate(totals["legal"], total),
        "avg_nodes_expanded": totals["nodes"] / total if total else None,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


if __name__ == "__main__":
    main()
