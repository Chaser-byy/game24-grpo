#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model for another training stage."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into a full model")
    parser.add_argument("--adapter", required=True, help="Directory containing adapter_config.json")
    parser.add_argument("--output", required=True, help="Directory for the merged full model")
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter)
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise SystemExit(f"adapter_config.json not found: {adapter_config_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model = adapter_config["base_model_name_or_path"]
    dtype = {
        "auto": "auto",
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    print(f"Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("Merging adapter")
    merged = model.merge_and_unload()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(output)
    print(f"Merged model saved to {output}")


if __name__ == "__main__":
    main()
