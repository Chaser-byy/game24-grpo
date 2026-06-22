"""Reproducible evaluation for Game24 and Countdown arithmetic tasks."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from game24.data import dataset_fingerprint, load_jsonl
from game24.inference import generate_responses, load_qwen, set_seed
from game24.parser import extract_answer, parse_response
from game24.rewards import UNSOLVABLE_ANSWER, score_response
from game24.verifier import VerificationResult, check_expression


def evaluate_model(
    model_name: str,
    data_path: str,
    output_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    limit: int = 0,
    max_new_tokens: int = 256,
    num_samples: int = 1,
    sample: bool | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    index_start: int | None = None,
    index_end: int | None = None,
) -> dict[str, Any]:
    """Evaluate exact accuracy@1 and pass@k while retaining attempt diagnostics."""

    examples = load_jsonl(data_path)
    start = index_start or 0
    end = index_end if index_end is not None else len(examples)
    examples = examples[start:end]
    if limit > 0:
        examples = examples[:limit]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    tokenizer, model = load_qwen(model_name)
    totals: defaultdict[str, int] = defaultdict(int)
    difficulty: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    with output.open("w", encoding="utf-8") as file:
        for index, example in enumerate(examples, 1):
            responses = generate_responses(
                tokenizer,
                model,
                example.numbers,
                example.target,
                max_new_tokens=max_new_tokens,
                num_samples=num_samples,
                sample=sample,
                temperature=temperature,
                top_p=top_p,
            )
            attempts = [_score_attempt(response, example) for response in responses]
            first = attempts[0]
            any_correct = any(item["correct"] for item in attempts)
            bucket = _difficulty_bucket(example.solved_rate)

            totals["total"] += 1
            totals["format"] += int(first["format_valid"])
            totals["syntax"] += int(first["syntax_valid"])
            totals["legal"] += int(first["numbers_valid"])
            totals["correct"] += int(first["correct"])
            totals["pass"] += int(any_correct)
            difficulty[bucket]["total"] += 1
            difficulty[bucket]["correct"] += int(first["correct"])
            difficulty[bucket]["pass"] += int(any_correct)

            if example.solvable is False:
                totals["unsolvable"] += 1
                totals["correct_abstention"] += int(first["answer"] == UNSOLVABLE_ANSWER)
                totals["false_claim"] += int(first["claimed_solution"])

            record = {
                "example_id": example.example_id,
                "numbers": list(example.numbers),
                "target": example.target,
                "solvable": example.solvable,
                "source": example.source,
                "split": example.split,
                "rank": example.rank,
                "solved_rate": example.solved_rate,
                "difficulty_bucket": bucket,
                "correct_at_1": first["correct"],
                f"pass_at_{num_samples}": any_correct,
                "attempts": attempts,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(examples)}] {example.numbers} correct={first['correct']}")

    total = totals["total"]
    unsolvable = totals["unsolvable"]
    summary = {
        "model": model_name,
        "data": data_path,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "total": total,
        "num_samples": num_samples,
        "correct": totals["correct"],
        "accuracy_at_1": _rate(totals["correct"], total),
        "accuracy_at_1_ci95": _wilson_interval(totals["correct"], total),
        f"pass_at_{num_samples}": _rate(totals["pass"], total),
        f"pass_at_{num_samples}_ci95": _wilson_interval(totals["pass"], total),
        "strict_format_rate": _rate(totals["format"], total),
        "syntax_rate": _rate(totals["syntax"], total),
        "legal_number_rate": _rate(totals["legal"], total),
        "unsolvable_total": unsolvable,
        "correct_abstention_rate": _rate(totals["correct_abstention"], unsolvable),
        "false_claim_rate": _rate(totals["false_claim"], unsolvable),
        "difficulty": {
            name: {
                "total": values["total"],
                "accuracy_at_1": _rate(values["correct"], values["total"]),
                f"pass_at_{num_samples}": _rate(values["pass"], values["total"]),
            }
            for name, values in sorted(difficulty.items())
        },
        "generation": {
            "sample": sample if sample is not None else num_samples > 1,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        },
        "slice": {"start": start, "end": end},
    }
    summary_output = Path(summary_path) if summary_path else output.with_suffix(".summary.json")
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _score_attempt(response: str, example) -> dict[str, Any]:
    parsed = parse_response(response)
    lenient_answer = extract_answer(response, strict=False)
    reward = score_response(response, example.numbers, example.target, example.solvable)
    answer = parsed.answer
    if answer is None or answer == UNSOLVABLE_ANSWER:
        result = VerificationResult(
            False,
            [],
            None,
            "declared unsolvable" if answer == UNSOLVABLE_ANSWER else parsed.reason,
        )
    else:
        result = check_expression(answer, example.numbers, example.target)
    return {
        "model_output": response,
        "think": parsed.think,
        "answer": answer,
        "claimed_solution": (lenient_answer is not None and lenient_answer != UNSOLVABLE_ANSWER),
        "format_valid": parsed.valid_format,
        "syntax_valid": result.syntax_valid,
        "numbers_valid": result.numbers_valid,
        "expression_reaches_target": result.valid,
        "correct": reward.correctness == 1.0,
        "used_numbers": result.used_numbers,
        "value": str(result.value) if result.value is not None else None,
        "reason": result.reason,
        "reward": {
            "format": reward.format,
            "syntax": reward.syntax,
            "number_usage": reward.number_usage,
            "correctness": reward.correctness,
            "total": reward.total,
        },
    }


def _difficulty_bucket(solved_rate: float | None) -> str:
    if solved_rate is None:
        return "unknown"
    if solved_rate >= 0.8:
        return "easy_80_100"
    if solved_rate >= 0.5:
        return "medium_50_80"
    return "hard_0_50"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    """Return a 95% Wilson confidence interval for a Bernoulli rate."""

    if total == 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * ((rate * (1 - rate) / total + z**2 / (4 * total**2)) ** 0.5)
    margin /= denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]
