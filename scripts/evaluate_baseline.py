#!/usr/bin/env python3
"""Evaluate a Qwen baseline on a Game of 24 JSONL dataset."""

import argparse

from game24.evaluation import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen Game of 24 baseline")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=20, help="Number of puzzles; 0 uses all")
    parser.add_argument("--output", default="outputs/baseline_20.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank-min", type=int)
    parser.add_argument("--rank-max", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        args.model,
        args.data,
        args.output,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        sample=args.sample,
        seed=args.seed,
        rank_min=args.rank_min,
        rank_max=args.rank_max,
    )


if __name__ == "__main__":
    main()
