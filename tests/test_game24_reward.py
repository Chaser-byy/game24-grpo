import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "verl" / "utils" / "reward_score" / "game24.py"
SPEC = importlib.util.spec_from_file_location("game24_reward", MODULE_PATH)
game24_reward = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(game24_reward)

compute_score = game24_reward.compute_score
validate_solution = game24_reward.validate_solution


def wrap(answer):
    return f"<think>try arithmetic</think><answer>{answer}</answer>"


def gt(numbers):
    return {"target": 24, "numbers": numbers, "solvable": True}


def test_correct_answer():
    assert compute_score(wrap("(7 - (8 / 8)) * 4"), gt([4, 7, 8, 8])) == 1.0


def test_wrong_value_gets_format_score():
    assert compute_score(wrap("4 + 7 + 8 + 8"), gt([4, 7, 8, 8])) == 0.1


def test_missing_number_rejected():
    details = validate_solution(wrap("(7 - 1) * 4"), gt([4, 7, 8, 8]))
    assert not details["correct"]
    assert details["error_type"] == "wrong_numbers"


def test_repeated_number_rejected():
    details = validate_solution(wrap("4 * 5 + 4"), gt([2, 3, 4, 5]))
    assert not details["correct"]
    assert details["error_type"] == "wrong_numbers"


def test_number_outside_prompt_rejected():
    details = validate_solution(wrap("6 * 4"), gt([1, 2, 3, 4]))
    assert not details["correct"]
    assert details["error_type"] == "wrong_numbers"


def test_duplicate_input_numbers_are_counted_exactly():
    assert compute_score(wrap("(1 + 1) * 8 + 8"), gt([1, 1, 8, 8])) == 1.0


def test_multi_digit_numbers():
    assert compute_score(wrap("12 + 10 + 1 + 1"), gt([1, 1, 10, 12])) == 1.0


def test_division_by_zero_rejected():
    details = validate_solution(wrap("1 / (2 - 2) + 3"), gt([1, 2, 2, 3]))
    assert not details["correct"]
    assert details["error_type"] == "division_by_zero"


def test_illegal_characters_rejected():
    details = validate_solution(wrap("4 + 7 + x"), gt([4, 5, 7, 8]))
    assert not details["correct"]
    assert details["error_type"] == "invalid_characters"


def test_function_call_rejected():
    details = validate_solution(wrap("abs(24)"), gt([1, 2, 3, 4]))
    assert not details["correct"]
    assert details["error_type"] == "invalid_characters"


def test_power_operator_rejected():
    details = validate_solution(wrap("2 ** 3 + 4 + 15"), gt([2, 3, 4, 15]))
    assert not details["correct"]
    assert details["error_type"] == "unsupported_operator"


def test_missing_answer_tag_gets_zero():
    assert compute_score("<think>done</think>24", gt([1, 2, 3, 4])) == 0.0


def test_explanation_inside_answer_rejected():
    details = validate_solution(wrap("(7 - 8 / 8) * 4 = 24"), gt([4, 7, 8, 8]))
    assert not details["correct"]
    assert details["error_type"] == "invalid_characters"


def test_fraction_intermediate_solution():
    assert compute_score(wrap("8 / (3 - 8 / 3)"), gt([3, 3, 8, 8])) == 1.0


def test_qwen_prompt_region_ignores_user_format_example():
    text = (
        "<|im_start|>user\nReturn <answer>expression only</answer><|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>use the duplicate eights</think><answer>(7 - 8 / 8) * 4</answer>"
    )
    assert compute_score(text, gt([4, 7, 8, 8])) == 1.0
