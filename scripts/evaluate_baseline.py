#!/usr/bin/env python3
"""Evaluate a Qwen baseline on a Game of 24 JSONL dataset."""

import argparse

from game24.evaluation import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen Game of 24 baseline")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Number of puzzles; 0 uses all")
    parser.add_argument("--output", default="outputs/evaluation.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-start", type=int)
    parser.add_argument("--index-end", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        args.model,
        args.data,
        args.output,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        index_start=args.index_start,
        index_end=args.index_end,
    )


if __name__ == "__main__":
    main()
