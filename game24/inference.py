"""Shared Qwen loading and generation code."""

import json
from pathlib import Path
from typing import Any

from game24.prompts import build_prompt


def load_qwen(model_name: str) -> tuple[Any, Any]:
    """Load a Qwen tokenizer and model on one NVIDIA GPU."""

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
    model.to("cuda")
    model.eval()
    return tokenizer, model


def generate_response(
    tokenizer: Any,
    model: Any,
    numbers: tuple[int, int, int, int],
    max_new_tokens: int = 512,
    sample: bool = False,
) -> str:
    """Generate one response and decode only newly generated tokens."""

    messages = [{"role": "user", "content": build_prompt(numbers)}]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    generation_args = {"max_new_tokens": max_new_tokens, "do_sample": sample}
    if sample:
        generation_args.update(temperature=0.7, top_p=0.9)
    else:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    generated_ids = model.generate(**model_inputs, **generation_args)
    new_token_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids, strict=True)
    ]
    return tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]


def set_seed(seed: int) -> None:
    """Set the Transformers random seed before evaluation."""

    from transformers import set_seed as transformers_set_seed

    transformers_set_seed(seed)
