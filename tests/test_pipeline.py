"""Tests for strict arithmetic RLVR data, rewards, and experiment splits."""

import json
from pathlib import Path

import pytest

from game24.data import (
    Game24Example,
    dataset_fingerprint,
    deduplicate,
    load_jsonl,
    normalize_record,
    save_jsonl,
)
from game24.evaluation import _score_attempt, _wilson_interval
from game24.grpo_rewards import (
    correctness_reward,
    get_reward_functions,
    number_usage_reward,
    strict_format_reward,
    syntax_reward,
)
from game24.model_tot import (
    apply_candidate,
    build_tot_prompt,
    initial_state,
    parse_operation_candidates,
)
from game24.parser import extract_answer, parse_response
from game24.prompts import build_prompt
from game24.rewards import compute_reward, score_response
from game24.sft import build_sft_examples, build_sft_response, solver_label
from game24.solver import find_solution, is_solvable
from game24.splits import build_game24_splits
from game24.tot import tot_search
from game24.tot_policy import build_tot_policy_samples, state_can_reach_target
from game24.trajectory import find_trajectory, trajectory_to_response
from game24.verifier import check_expression, verify_expression


def response(answer: str, think: str = "I checked all numbers and the target.") -> str:
    return f"<think>{think}</think><answer>{answer}</answer>"


def test_jsonl_round_trip_and_fingerprint(tmp_path: Path) -> None:
    examples = [Game24Example("demo", (1, 3, 4, 6), True)]
    path = tmp_path / "data.jsonl"
    save_jsonl(examples, path)
    assert load_jsonl(path) == examples
    assert dataset_fingerprint(examples) == dataset_fingerprint(load_jsonl(path))


def test_strict_response_protocol() -> None:
    valid = response("6/(1-3/4)")
    parsed = parse_response(valid)
    assert parsed.valid_format
    assert parsed.answer == "6/(1-3/4)"
    assert extract_answer(valid) == "6/(1-3/4)"

    assert not parse_response("<answer>6/(1-3/4)</answer>").valid_format
    assert not parse_response(f"junk {valid}").valid_format
    assert not parse_response(valid + " trailing").valid_format
    assert not parse_response("<think></think><answer>1+2</answer>").valid_format
    assert extract_answer(f"junk {valid}", strict=False) == "6/(1-3/4)"


@pytest.mark.parametrize(
    "expression",
    [
        "0xD+6+4+1",
        "1_3+6+4+1",
        "13+6+4+1 # comment",
        "13+6+4+1.0",
        "__import__('os').getcwd()",
        "13+6+4+1=24",
    ],
)
def test_verifier_rejects_forbidden_lexical_forms(expression: str) -> None:
    result = check_expression(expression, (1, 4, 6, 13))
    assert not result.valid
    assert not result.syntax_valid


def test_exact_expression_verification_and_diagnostics() -> None:
    assert verify_expression("6/(1-3/4)", (1, 3, 4, 6))
    assert verify_expression("(7-1)*4", (1, 4, 7), target=24)
    assert not verify_expression("6*4", (1, 3, 4, 6))
    assert not verify_expression("1/(3-3)", (1, 3, 3, 4))

    wrong_numbers = check_expression("6*4", (1, 3, 4, 6))
    assert wrong_numbers.syntax_valid
    assert not wrong_numbers.numbers_valid
    assert wrong_numbers.value == 24

    wrong_target = check_expression("(1+3)*4+6", (1, 3, 4, 6))
    assert wrong_target.syntax_valid and wrong_target.numbers_valid
    assert not wrong_target.target_valid


def test_reward_breakdown_is_strict_and_binary() -> None:
    correct = score_response(response("6/(1-3/4)"), (1, 3, 4, 6))
    assert correct.total == pytest.approx(1.4)
    assert correct.correctness == 1.0

    wrong = score_response(response("(1+3)*4+6"), (1, 3, 4, 6))
    assert wrong.total == pytest.approx(0.4)
    assert wrong.correctness == 0.0
    assert score_response("<answer>6/(1-3/4)</answer>", (1, 3, 4, 6)).total == 0.0
    assert compute_reward("6/(1-3/4)", (1, 3, 4, 6)) == 1.0


def test_unsolvable_abstention_reward() -> None:
    abstention = score_response(response("UNSOLVABLE"), (1, 1, 1, 1), solvable=False)
    assert abstention.correctness == 1.0
    assert abstention.total == pytest.approx(1.1)
    assert score_response(response("UNSOLVABLE"), (1, 3, 4, 6), solvable=True).correctness == 0.0


def test_unsolvable_evaluation_distinguishes_abstention_and_false_claim() -> None:
    example = Game24Example("none", (1, 1, 1, 1), False)
    abstention = _score_attempt(response("UNSOLVABLE"), example)
    claim = _score_attempt(response("(1+1+1+1)"), example)
    malformed_claim = _score_attempt("junk <answer>1+1+1+1</answer>", example)
    assert abstention["correct"] and not abstention["claimed_solution"]
    assert not claim["correct"] and claim["claimed_solution"]
    assert not malformed_claim["format_valid"] and malformed_claim["claimed_solution"]
    assert _wilson_interval(50, 100) == pytest.approx([0.4038315, 0.5961685])


def test_trl_reward_functions_share_the_strict_scorer() -> None:
    completions = [response("6/(1-3/4)"), response("(1+3)*4+6"), "no tags"]
    numbers = [[1, 3, 4, 6]] * 3
    target = [24] * 3
    solvable = [True] * 3
    kwargs = {"numbers": numbers, "target": target, "solvable": solvable}
    assert strict_format_reward(completions, **kwargs) == [0.1, 0.1, 0.0]
    assert syntax_reward(completions, **kwargs) == [0.1, 0.1, 0.0]
    assert number_usage_reward(completions, **kwargs) == [0.2, 0.2, 0.0]
    assert correctness_reward(completions, **kwargs) == [1.0, 0.0, 0.0]
    assert [fn.__name__ for fn in get_reward_functions("default")] == [
        "strict_format_reward",
        "syntax_reward",
        "number_usage_reward",
        "correctness_reward",
    ]
    assert [fn.__name__ for fn in get_reward_functions("accuracy")] == [
        "syntax_reward",
        "number_usage_reward",
        "correctness_reward",
    ]
    assert [fn.__name__ for fn in get_reward_functions("correctness")] == [
        "correctness_reward"
    ]
    with pytest.raises(ValueError):
        get_reward_functions("unknown")


def test_exact_solver() -> None:
    solution = find_solution((1, 3, 4, 6))
    assert solution is not None
    assert verify_expression(solution, (1, 3, 4, 6))
    assert not is_solvable((1, 1, 1, 1))


def test_verified_trajectory_response_contains_stateful_steps() -> None:
    trajectory = find_trajectory((1, 3, 4, 6))
    assert trajectory is not None
    assert verify_expression(trajectory.expression, (1, 3, 4, 6))
    assert len(trajectory.steps) == 3
    response_text = trajectory_to_response(trajectory)
    parsed = parse_response(response_text)
    assert parsed.valid_format
    assert "Remaining" in parsed.think
    assert score_response(response_text, (1, 3, 4, 6)).correctness == 1.0


def test_tot_search_finds_verified_expression() -> None:
    result = tot_search((4, 5, 6, 10))
    assert result.found
    assert result.expression is not None
    assert verify_expression(result.expression, (4, 5, 6, 10))
    assert result.trace

    unsolvable = tot_search((1, 1, 1, 1))
    assert not unsolvable.found


def test_model_tot_candidate_parser_and_executor() -> None:
    candidates = parse_operation_candidates("Try 0 * 1 = 20\n2 / 3", 4)
    assert [(item.left_index, item.operation, item.right_index) for item in candidates] == [
        (0, "*", 1),
        (2, "/", 3),
    ]
    state = initial_state((4, 5, 6, 10))
    child = apply_candidate(state, candidates[0])
    assert child is not None
    assert child.items[-1].expression == "(4*5)"
    assert child.items[-1].value == 20


def test_tot_policy_samples_are_verified_next_operations() -> None:
    example = Game24Example("demo", (1, 3, 4, 6), True)
    samples = build_tot_policy_samples(
        [example],
        candidates_per_state=3,
        max_states_per_example=4,
        max_actions_per_state=2,
    )
    assert samples

    sample = samples[0]
    candidates = parse_operation_candidates(sample.label, len(sample.state.items))
    assert len(candidates) == 1
    assert candidates[0].raw_text == sample.label

    child = apply_candidate(sample.state, candidates[0])
    assert child is not None
    assert state_can_reach_target(child, sample.target)
    assert sample.prompt == build_tot_prompt(sample.state, sample.target, 3)


def test_dynamic_countdown_normalization_and_deduplication() -> None:
    first = normalize_record(
        {"nums": [2, 3, 7], "target": 23},
        0,
        "Jiayi-Pan/Countdown-Tasks-3to4",
    )
    second = normalize_record(
        {"nums": [2, 3, 7], "target": 24},
        1,
        "Jiayi-Pan/Countdown-Tasks-3to4",
    )
    assert first.task_type == "countdown" and first.target == 23
    assert len(deduplicate([first, first, second])) == 2


def test_tot_slice_is_zero_based_half_open_and_leakage_free() -> None:
    ranked = [
        Game24Example(f"rank:{index + 1}", numbers, True, rank=index + 1)
        for index, numbers in enumerate([(1, 1, 1, 8), (1, 1, 2, 6), (1, 1, 3, 8), (1, 1, 4, 6)])
    ]
    training = ranked + [Game24Example("extra", (1, 2, 3, 4), True)]
    splits, manifest = build_game24_splits(
        training,
        ranked,
        test_start=1,
        test_end=3,
        validation_size=1,
        id_test_size=1,
        seed=7,
    )
    assert [item.rank for item in splits["test_hard"]] == [2, 3]
    assert len(splits["train_full"]) == 3
    assert len(splits["train"]) == 1
    assert len(splits["test_id"]) == 1
    assert manifest["test_slice"]["semantics"] == "python [start:end)"


def test_prompt_supports_fixed_and_dynamic_targets() -> None:
    prompt = build_prompt((1, 3, 4, 6), 24)
    assert "target 24" in prompt
    assert "[1, 3, 4, 6]" in prompt
    assert "<think>" in prompt and "<answer>" in prompt
    assert "UNSOLVABLE" in prompt
    assert "equals sign" in prompt
    assert "Reason through candidate operations" not in prompt


def test_sft_labels_are_verified_r1_responses() -> None:
    example = Game24Example("demo", (1, 3, 4, 6), True)
    label = solver_label(example)
    assert label is not None
    assert verify_expression(label.expression, example.numbers)
    assert parse_response(label.response).valid_format
    assert "Step 1" in label.response
    assert score_response(label.response, example.numbers).correctness == 1.0

    manual = build_sft_response("6/(1-3/4)")
    assert parse_response(manual).valid_format

    unsolvable = Game24Example("none", (1, 1, 1, 1), False)
    unsolvable_label = solver_label(unsolvable)
    assert unsolvable_label is not None
    assert unsolvable_label.expression == "UNSOLVABLE"
    assert score_response(unsolvable_label.response, unsolvable.numbers, solvable=False).correctness
    assert build_sft_examples([example, unsolvable]) == [label, unsolvable_label]


def test_direct_tot_sft_labels_train_final_answers() -> None:
    example = Game24Example("demo", (1, 3, 4, 6), True)
    rows = build_sft_examples([example], label_style="direct_tot", solutions_per_example=2)
    assert rows
    assert len(rows) <= 2

    parsed = parse_response(rows[0].response)
    assert parsed.valid_format
    assert parsed.answer == rows[0].expression
    assert "State 0" in rows[0].response
    assert "The final expression" in rows[0].response
    assert score_response(rows[0].response, example.numbers).correctness == 1.0
    assert verify_expression(rows[0].expression, example.numbers)


def test_manifest_json_shape_is_serializable() -> None:
    example = Game24Example("demo", (1, 3, 4, 6), True)
    assert json.dumps({"fingerprint": dataset_fingerprint([example])})
