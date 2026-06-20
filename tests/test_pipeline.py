"""Small tests for the course project's core pipeline."""

from pathlib import Path

import pytest

from game24.data import Game24Example, load_jsonl, save_jsonl
from game24.parser import extract_answer
from game24.prompts import build_prompt
from game24.rewards import compute_reward
from game24.verifier import check_expression, verify_expression
from scripts.train_grpo import correctness_reward, extraction_reward, number_usage_reward


def test_jsonl_round_trip(tmp_path: Path) -> None:
    examples = [Game24Example("demo", (1, 3, 4, 6), True)]
    path = tmp_path / "data.jsonl"
    save_jsonl(examples, path)
    assert load_jsonl(path) == examples


def test_answer_extraction_and_verification() -> None:
    response = "<think>some reasoning</think><answer>6 / (1 - 3 / 4)</answer>"
    answer = extract_answer(response)
    assert answer == "6 / (1 - 3 / 4)"
    assert verify_expression(answer, (1, 3, 4, 6))


def test_wrong_expressions_fail() -> None:
    assert not verify_expression("6 * 4", (1, 3, 4, 6))
    assert not verify_expression("__import__('os').getcwd()", (1, 3, 4, 6))
    assert not verify_expression("1 / (3 - 3)", (1, 3, 3, 4))

    result = check_expression("6 * 4", (1, 3, 4, 6))
    assert result.used_numbers == [6, 4]
    assert result.value == 24
    assert "expected numbers" in result.reason


def test_minimal_pipeline_reward() -> None:
    example = Game24Example("demo", (1, 3, 4, 6))
    assert "1, 3, 4, 6" in build_prompt(example.numbers)
    answer = extract_answer("<answer>6/(1-3/4)</answer>")
    assert compute_reward(answer, example.numbers) == 1.0
    assert compute_reward(None, example.numbers) == 0.0


def test_grpo_shaped_rewards() -> None:
    completions = [
        "<answer>1 + 3</answer>",
        "<answer>(1 + 3) * 4 + 6</answer>",
        "no answer tag",
    ]
    numbers = [[1, 3, 4, 6]] * 3

    assert extraction_reward(completions) == [0.1, 0.1, 0.0]
    assert number_usage_reward(completions, numbers) == [0.1, 0.3, 0.0]

    close_answers = [
        "<answer>(6 - 1) * 4 + 3</answer>",
        "<answer>6 / (1 - 3 / 4)</answer>",
        "<answer>6 * 4</answer>",
    ]
    rewards = correctness_reward(close_answers, numbers)
    assert rewards[0] == pytest.approx(0.2 * 23 / 24)
    assert rewards[1] == 1.0
    assert rewards[2] == 0.0
