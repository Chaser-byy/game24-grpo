#!/usr/bin/env python3
"""Run a small single-GPU GRPO training job with LoRA."""

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from game24.data import load_jsonl
from game24.evaluation import evaluate_model
from game24.parser import extract_answer
from game24.prompts import build_prompt
from game24.rewards import compute_reward
from game24.verifier import check_expression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen on Game of 24 with GRPO")
    parser.add_argument("--model", required=True, help="Local Qwen model directory")
    parser.add_argument("--data", required=True, help="Local training JSONL file")
    parser.add_argument("--output", required=True, help="Directory for LoRA weights")
    parser.add_argument("--eval-data", help="Optional validation or test JSONL file")
    parser.add_argument("--run-eval", action="store_true", help="Evaluate after training")
    parser.add_argument("--eval-limit", type=int, default=20, help="Eval examples; 0 uses all")
    parser.add_argument("--limit", type=int, default=20, help="Training examples; 0 uses all")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument(
        "--max-completion-length",
        "--max-new-tokens",
        dest="max_completion_length",
        type=int,
        default=128,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", help="Path to a TRL checkpoint")
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
    if args.run_eval and not args.eval_data:
        raise SystemExit("--run-eval requires --eval-data")
    if not Path(args.model).is_dir():
        raise SystemExit(f"local model directory not found: {args.model}")
    if not Path(args.data).is_file():
        raise SystemExit(f"training data not found: {args.data}")
    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).is_dir():
        raise SystemExit(f"checkpoint not found: {args.resume_from_checkpoint}")

    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; GRPO training requires one NVIDIA GPU")
    properties = torch.cuda.get_device_properties(0)
    memory_gb = properties.total_memory / 1024**3
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"Transformers: {transformers.__version__}")
    print(f"TRL: {trl.__version__}")
    print(f"PEFT: {peft.__version__}")
    print(f"GPU: {properties.name} ({memory_gb:.1f} GB)")

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
        temperature=1.0,
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
    # T4 FP16 generation can occasionally produce invalid logits before sampling.
    trainer.generation_config.remove_invalid_values = True
    trainer.generation_config.renormalize_logits = True
    trainer.generation_config.use_cache = False

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    run_info = {
        "command_args": vars(args),
        "grpo_config": training_args.to_dict(),
        "train_examples": len(examples),
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "targets": ["q_proj", "v_proj"]},
        "started_at": started_at,
        "completed_at": None,
        "final_step": 0,
    }
    _write_json(output_dir / "training_args.json", run_info)

    print(f"Training on {len(examples)} puzzles; saving to {args.output}")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    completed_at = datetime.now(timezone.utc).isoformat()
    final_record = {
        "event": "train_complete",
        "step": trainer.state.global_step,
        "completed_at": completed_at,
        **train_result.metrics,
    }
    with (output_dir / "train_metrics.jsonl").open("w", encoding="utf-8") as file:
        for record in [*trainer.state.log_history, final_record]:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    run_info["completed_at"] = completed_at
    run_info["final_step"] = trainer.state.global_step
    run_info["train_result"] = train_result.metrics
    _write_json(output_dir / "training_args.json", run_info)

    if args.run_eval:
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        evaluate_model(
            args.output,
            args.eval_data,
            output_dir / "eval_results.jsonl",
            summary_path=output_dir / "eval_summary.json",
            limit=args.eval_limit,
            max_new_tokens=args.max_completion_length,
            seed=args.seed,
        )


def _write_json(path: Path, content: dict[str, Any]) -> None:
    text = json.dumps(content, ensure_ascii=False, indent=2, default=str) + "\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
