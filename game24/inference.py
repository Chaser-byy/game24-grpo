"""Shared Qwen/LoRA loading and generation code."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from game24.prompts import build_prompt


def load_qwen(model_name: str) -> tuple[Any, Any]:
    """Load a base Qwen model or a saved PEFT adapter on one GPU."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_name)
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        from peft import PeftModel

        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model = adapter_config["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
        model = PeftModel.from_pretrained(model, model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to("cuda")
    model.eval()
    return tokenizer, model


def generate_responses(
    tokenizer: Any,
    model: Any,
    numbers: Sequence[int],
    target: int = 24,
    *,
    max_new_tokens: int = 256,
    num_samples: int = 1,
    sample: bool | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """Generate one greedy completion or multiple stochastic candidates."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if sample is None:
        sample = num_samples > 1
    if num_samples > 1 and not sample:
        raise ValueError("multiple return sequences require sampling")

    messages = [
        {
            "role": "user",
            "content": build_prompt(numbers, target, allow_unsolvable=True),
        }
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    generation_args: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "num_return_sequences": num_samples,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if sample:
        generation_args.update(temperature=temperature, top_p=top_p)

    generated_ids = model.generate(**model_inputs, **generation_args)
    prompt_length = model_inputs.input_ids.shape[1]
    return tokenizer.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
    )


def generate_response(
    tokenizer: Any,
    model: Any,
    numbers: Sequence[int],
    target: int = 24,
    max_new_tokens: int = 256,
    sample: bool = False,
) -> str:
    """Backward-compatible single-response helper."""

    return generate_responses(
        tokenizer,
        model,
        numbers,
        target,
        max_new_tokens=max_new_tokens,
        num_samples=1,
        sample=sample,
    )[0]


def set_seed(seed: int) -> None:
    """Set the Transformers random seed before evaluation."""

    from transformers import set_seed as transformers_set_seed

    transformers_set_seed(seed)
