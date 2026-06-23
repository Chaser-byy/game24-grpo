import argparse
import json
import os
from statistics import mean

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score.game24 import compute_score, extract_solution


def load_rows(data_dir, split, limit):
    path = os.path.join(data_dir, f"{split}.parquet")
    frame = pd.read_parquet(path)
    rows = frame.to_dict("records")
    if limit is not None:
        rows = rows[:limit]
    return rows


def generation_prompt(row):
    prompt = row["prompt"]
    if isinstance(prompt, list):
        return prompt[0]["content"]
    return prompt


def generate_outputs(model, tokenizer, prompt, n, args):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=n > 1 or args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(output[input_length:], skip_special_tokens=False) for output in outputs]


def summarize(split, rows, results, pass_at):
    total = len(rows)
    flat = [item for item in results if item["split"] == split]
    by_index = {}
    for item in flat:
        by_index.setdefault(item["index"], []).append(item)

    metrics = {"split": split, "num_examples": total, "num_generations": len(flat)}
    if not flat:
        return metrics

    metrics.update({
        "format_rate": mean(1.0 if item["format_ok"] else 0.0 for item in flat),
        "parse_rate": mean(1.0 if item["parse_ok"] else 0.0 for item in flat),
        "number_use_rate": mean(1.0 if item["numbers_ok"] else 0.0 for item in flat),
        "valid_expression_rate": mean(1.0 if item["parse_ok"] and item["numbers_ok"] else 0.0 for item in flat),
        "average_reward": mean(float(item["reward"]) for item in flat),
        "average_response_length": mean(len(item["model_output"]) for item in flat),
    })
    for k in pass_at:
        metrics[f"pass@{k}"] = mean(
            1.0 if any(item["correct"] for item in by_index[index][:k]) else 0.0
            for index in by_index
        )
    metrics["solved_rate"] = metrics.get("pass@1", 0.0)

    if split == "test_unsolvable":
        first_outputs = [items[0] for items in by_index.values() if items]
        metrics["false_solution_rate"] = mean(1.0 if item["correct"] else 0.0 for item in first_outputs)
        metrics["abstention_rate"] = mean(
            1.0 if "NO_SOLUTION" in (item["extracted_answer"] or item["model_output"]).upper() else 0.0
            for item in first_outputs
        )
        metrics["invalid_answer_rate"] = mean(1.0 if item["reward"] == 0.0 else 0.0 for item in first_outputs)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_dir", default="data/game24")
    parser.add_argument("--output_dir", default="outputs/game24_eval")
    parser.add_argument("--splits", nargs="+", default=["validation", "test_id", "test_hard", "test_unsolvable"])
    parser.add_argument("--pass_at", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    max_n = max(args.pass_at)
    all_results = []
    all_metrics = []
    for split in args.splits:
        rows = load_rows(args.data_dir, split, args.limit)
        split_results = []
        for row_index, row in enumerate(rows):
            prompt = generation_prompt(row)
            ground_truth = row["reward_model"]["ground_truth"]
            outputs = generate_outputs(model, tokenizer, prompt, max_n, args)
            for generation_index, output in enumerate(outputs):
                details = compute_score(output, ground_truth, return_details=True)
                result = {
                    "split": split,
                    "index": row_index,
                    "generation_index": generation_index,
                    "numbers": ground_truth["numbers"],
                    "target": ground_truth.get("target", 24),
                    "model_output": output,
                    "extracted_answer": extract_solution(output),
                    **details,
                }
                split_results.append(result)
        all_results.extend(split_results)
        all_metrics.append(summarize(split, rows, split_results, args.pass_at))

    results_path = os.path.join(args.output_dir, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as handle:
        for item in all_results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"Saved per-example results to {results_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
