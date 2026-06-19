#!/usr/bin/env python3
"""Run a small single-GPU GRPO training job with LoRA."""

import argparse
from typing import Any

from game24.data import load_jsonl
from game24.parser import extract_answer
from game24.prompts import build_prompt
from game24.rewards import compute_reward
from game24.verifier import check_expression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen on Game of 24 with GRPO")
    parser.add_argument("--model", required=True, help="Local Qwen model directory")
    parser.add_argument("--data", required=True, help="Local training JSONL file")
    parser.add_argument("--output", required=True, help="Directory for LoRA weights")
    parser.add_argument("--limit", type=int, default=20, help="Training examples; 0 uses all")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--max-completion-length",
        "--max-new-tokens",
        dest="max_completion_length",
        type=int,
        default=256,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def completion_text(completion: Any) -> str:
    """Read text from either standard or conversational TRL completions."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        message = completion[0]
        if isinstance(message, dict):
            return str(message.get("content", ""))
    return ""


def extraction_reward(completions: list[Any], **_: Any) -> list[float]:
    """Give a small reward when an answer tag can be extracted."""

    return [0.1 if extract_answer(completion_text(item)) else 0.0 for item in completions]


def number_usage_reward(
    completions: list[Any], numbers: list[list[int]], **_: Any
) -> list[float]:
    """Reward valid syntax that uses every input number exactly once."""

    rewards = []
    for completion, expected in zip(completions, numbers, strict=True):
        expression = extract_answer(completion_text(completion))
        if expression is None:
            rewards.append(0.0)
            continue
        result = check_expression(expression, tuple(expected))
        numbers_match = result.value is not None and sorted(result.used_numbers) == sorted(expected)
        rewards.append(0.3 if numbers_match else 0.0)
    return rewards


def correctness_reward(
    completions: list[Any], numbers: list[list[int]], **_: Any
) -> list[float]:
    """Give the largest reward to expressions that equal 24."""

    return [
        compute_reward(extract_answer(completion_text(completion)), tuple(expected))
        for completion, expected in zip(completions, numbers, strict=True)
    ]


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    examples = [example for example in load_jsonl(args.data) if example.solvable is not False]
    if args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("no solvable training examples found")

    rows = [
        {
            "prompt": [{"role": "user", "content": build_prompt(example.numbers)}],
            "numbers": list(example.numbers),
        }
        for example in examples
    ]
    train_dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    training_args = GRPOConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=args.max_completion_length,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_steps=args.max_steps,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[extraction_reward, number_usage_reward, correctness_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"Training on {len(examples)} puzzles; saving to {args.output}")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
