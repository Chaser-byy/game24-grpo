#!/usr/bin/env python3
"""Train a LoRA SFT adapter for ToT next-operation policies."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from game24.data import load_jsonl
from game24.tot_policy import build_tot_policy_samples, sample_to_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ToT next-operation policy SFT")
    parser.add_argument("--model", required=True, help="Local base model directory")
    parser.add_argument("--data", required=True, nargs="+", help="Processed train JSONL files")
    parser.add_argument("--output", required=True, help="LoRA adapter output directory")
    parser.add_argument(
        "--merged-output",
        help="Full merged model directory for ToT evaluation; defaults to OUTPUT_merged",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit input examples before labeling")
    parser.add_argument("--candidates-per-state", type=int, default=4)
    parser.add_argument("--max-states-per-example", type=int, default=16)
    parser.add_argument("--max-actions-per-state", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; ToT policy SFT requires one NVIDIA GPU")

    set_seed(args.seed)
    examples = []
    for path in args.data:
        examples.extend(load_jsonl(path))
    candidate_examples = [example for example in examples if example.solvable is not False]
    if args.limit > 0:
        candidate_examples = candidate_examples[: args.limit]

    samples = build_tot_policy_samples(
        candidate_examples,
        candidates_per_state=args.candidates_per_state,
        max_states_per_example=args.max_states_per_example,
        max_actions_per_state=args.max_actions_per_state,
    )
    if not samples:
        raise SystemExit("no ToT policy samples available from the provided data")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_output = Path(args.merged_output or f"{args.output}_merged")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)

    tokenized = [_tokenize_sample(tokenizer, sample, args.max_length) for sample in samples]
    tokenized = [item for item in tokenized if item is not None]
    if not tokenized:
        raise SystemExit("all ToT policy rows were truncated before the response")

    class ToTPolicyDataset(Dataset):
        def __len__(self) -> int:
            return len(tokenized)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return tokenized[index]

    run_info = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "arguments": vars(args),
        "input_examples": len(examples),
        "candidate_solvable_examples": len(candidate_examples),
        "policy_labeled_samples": len(samples),
        "tokenized_examples": len(tokenized),
        "merged_output": str(merged_output),
        "sample_labels": [sample_to_record(sample) for sample in samples[:5]],
    }
    _write_json(output_dir / "tot_policy_sft_run_config.json", run_info)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ToTPolicyDataset(),
        data_collator=_collate(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    merged_output.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_output, safe_serialization=True)
    tokenizer.save_pretrained(merged_output)

    run_info["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "tot_policy_sft_run_config.json", run_info)
    print(f"ToT policy SFT adapter saved to {output_dir}")
    print(f"Merged model for ToT evaluation saved to {merged_output}")


def _tokenize_sample(tokenizer: Any, sample, max_length: int) -> dict[str, list[int]] | None:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    eos = tokenizer.eos_token or ""
    full_text = prompt + sample.label + eos
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    labels = list(input_ids)
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length
    if all(label == -100 for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def _collate(pad_token_id: int):
    def collate(features: list[dict[str, list[int]]]):
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_token_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    return collate


def _validate_args(args: argparse.Namespace) -> None:
    if not Path(args.model).is_dir():
        raise SystemExit(f"local model directory not found: {args.model}")
    for path in args.data:
        if not Path(path).is_file():
            raise SystemExit(f"training data not found: {path}")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise SystemExit("batch size and gradient accumulation must be positive")
    if args.max_length < 128:
        raise SystemExit("--max-length is too small for chat-formatted ToT policy rows")
    if args.candidates_per_state < 1:
        raise SystemExit("--candidates-per-state must be positive")
    if args.max_states_per_example < 1:
        raise SystemExit("--max-states-per-example must be positive")
    if args.max_actions_per_state < 1:
        raise SystemExit("--max-actions-per-state must be positive")


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
