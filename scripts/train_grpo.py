#!/usr/bin/env python3
"""Train Qwen with leakage-safe, verifiable GRPO rewards."""

import argparse
import gc
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from game24.data import load_jsonl
from game24.evaluation import evaluate_model
from game24.grpo_rewards import REWARD_FUNCTIONS, completion_text
from game24.prompts import build_prompt
from game24.rewards import score_response


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    known, _ = config_parser.parse_known_args()
    defaults: dict[str, Any] = {}
    if known.config:
        defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description="Train Qwen on arithmetic RLVR with GRPO")
    parser.add_argument("--config", help="JSON file containing default hyperparameters")
    parser.add_argument("--model", required=True, help="Local Qwen model directory")
    parser.add_argument("--data", required=True, nargs="+", help="One or more training JSONL files")
    parser.add_argument("--output", required=True, help="Training and adapter output directory")
    parser.add_argument("--eval-data", help="Optional periodic validation JSONL")
    parser.add_argument("--run-final-eval", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04, help="Reference-policy KL coefficient")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--prompts-per-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--completion-log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    valid_config_keys = {action.dest for action in parser._actions}
    unknown_defaults = set(defaults) - valid_config_keys
    if unknown_defaults:
        parser.error(f"unknown config keys: {', '.join(sorted(unknown_defaults))}")
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    _validate_args(args)

    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_callback import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; GRPO training requires one NVIDIA GPU")

    examples = []
    for path in args.data:
        examples.extend(load_jsonl(path))
    examples = [example for example in examples if example.solvable is not False]
    if args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("no solvable training examples found")

    rows = [_training_row(example) for example in examples]
    train_dataset = Dataset.from_list(rows)
    eval_examples = load_jsonl(args.eval_data) if args.eval_data else []
    if args.eval_limit > 0:
        eval_examples = eval_examples[: args.eval_limit]
    eval_dataset = (
        Dataset.from_list([_training_row(item) for item in eval_examples])
        if eval_examples
        else None
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "train_metrics.jsonl"
    completions_path = output_dir / "completion_samples.jsonl"
    log_file = log_path.open("a", encoding="utf-8")
    metrics_file = metrics_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        print(message, file=log_file, flush=True)

    properties = torch.cuda.get_device_properties(0)
    memory_gb = properties.total_memory / 1024**3
    log(f"\n=== Training run {datetime.now(timezone.utc).isoformat()} ===")
    log(f"Python: {sys.version.split()[0]}")
    log(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    log(
        f"Transformers: {transformers.__version__}; "
        f"TRL: {trl.__version__}; PEFT: {peft.__version__}"
    )
    log(f"GPU: {properties.name} ({memory_gb:.1f} GB)")
    log(f"Arguments: {json.dumps(vars(args), ensure_ascii=False)}")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    with redirect_stdout(log_file), redirect_stderr(log_file):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model.config.use_cache = False

    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    batch_size = args.prompts_per_batch * args.num_generations
    training_args = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        beta=args.beta,
        reward_weights=[1.0] * len(REWARD_FUNCTIONS),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=384,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    class AuditedGRPOTrainer(GRPOTrainer):
        """Persist representative on-policy completions for qualitative analysis."""

        _logged_markers: set[tuple[int, str]] = set()

        def _prepare_inputs(self, inputs):
            # Gradient checkpointing disables KV cache while the model is in train mode.
            # Rollouts and reference log-probs need no gradients, so use eval mode here;
            # this restores cached autoregressive generation and removes dropout noise.
            was_training = self.model.training
            self.model.eval()
            try:
                prepared = super()._prepare_inputs(inputs)
            finally:
                if was_training:
                    self.model.train()
            phase = "train" if was_training else "eval"
            marker = (self.state.global_step, phase)
            should_log = (
                self.accelerator.is_main_process
                and self.state.global_step % args.completion_log_steps == 0
                and marker not in self._logged_markers
            )
            if should_log:
                texts = self.processing_class.batch_decode(
                    prepared["completion_ids"], skip_special_tokens=True
                )
                with completions_path.open("a", encoding="utf-8") as sample_file:
                    for item, text in zip(inputs, texts, strict=True):
                        reward = score_response(
                            completion_text(text),
                            item["numbers"],
                            item["target"],
                            item["solvable"],
                        )
                        record = {
                            "step": self.state.global_step,
                            "phase": phase,
                            "numbers": item["numbers"],
                            "target": item["target"],
                            "completion": text,
                            "reward": reward.total,
                            "correct": reward.correctness == 1.0,
                        }
                        sample_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._logged_markers.add(marker)
            return prepared

    with redirect_stdout(log_file), redirect_stderr(log_file):
        trainer = AuditedGRPOTrainer(
            model=model,
            reward_funcs=REWARD_FUNCTIONS,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    trainer.generation_config.remove_invalid_values = True
    trainer.generation_config.renormalize_logits = True
    trainer.generation_config.use_cache = True

    class JsonlMetricsCallback(TrainerCallback):
        """Persist every train and validation metric while the run is active."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs:
                record = {"step": state.global_step, **logs}
                line = json.dumps(record, ensure_ascii=False, default=str)
                print(line, file=metrics_file, flush=True)
                log(line)

    trainer.add_callback(JsonlMetricsCallback())
    run_info = {
        "command_args": vars(args),
        "grpo_config": training_args.to_dict(),
        "train_examples": len(examples),
        "eval_examples": len(eval_examples),
        "reward_functions": [function.__name__ for function in REWARD_FUNCTIONS],
        "completion_samples": str(completions_path),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": lora_targets,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    _write_json(output_dir / "run_config.json", run_info)

    try:
        with redirect_stdout(log_file):
            train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except Exception:
        traceback.print_exc(file=log_file)
        metrics_file.close()
        log_file.close()
        raise SystemExit(f"Training failed; see {log_path}")

    with redirect_stdout(log_file), redirect_stderr(log_file):
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)
    run_info["completed_at"] = datetime.now(timezone.utc).isoformat()
    run_info["final_step"] = trainer.state.global_step
    run_info["train_result"] = train_result.metrics
    _write_json(output_dir / "run_config.json", run_info)
    metrics_file.close()

    if args.run_final_eval and args.eval_data:
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        with redirect_stdout(log_file), redirect_stderr(log_file):
            evaluate_model(
                args.output,
                args.eval_data,
                output_dir / "final_eval.jsonl",
                limit=args.eval_limit,
                max_new_tokens=args.max_completion_length,
                num_samples=1,
                seed=args.seed,
            )
    log(f"Training complete; live metrics saved to {metrics_path}")
    log_file.close()


def _training_row(example) -> dict[str, Any]:
    return {
        "prompt": [
            {
                "role": "user",
                "content": build_prompt(example.numbers, example.target, allow_unsolvable=True),
            }
        ],
        "numbers": list(example.numbers),
        "target": example.target,
        "solvable": example.solvable,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not Path(args.model).is_dir():
        raise SystemExit(f"local model directory not found: {args.model}")
    for path in args.data:
        if not Path(path).is_file():
            raise SystemExit(f"training data not found: {path}")
    if args.eval_data and not Path(args.eval_data).is_file():
        raise SystemExit(f"evaluation data not found: {args.eval_data}")
    if args.run_final_eval and not args.eval_data:
        raise SystemExit("--run-final-eval requires --eval-data")
    if args.num_generations < 2:
        raise SystemExit("--num-generations must be at least 2 for GRPO")
    if args.completion_log_steps < 1 or args.logging_steps < 1:
        raise SystemExit("logging intervals must be positive")
    if args.eval_steps < 1 or args.save_steps < 1:
        raise SystemExit("evaluation and save intervals must be positive")
    if args.prompts_per_batch < 1 or args.gradient_accumulation_steps < 1:
        raise SystemExit("batch and gradient accumulation sizes must be positive")
    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).is_dir():
        raise SystemExit(f"checkpoint not found: {args.resume_from_checkpoint}")


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
