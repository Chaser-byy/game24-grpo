import ast
import random
import re
from collections import Counter
from fractions import Fraction


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ALLOWED_EXPRESSION_RE = re.compile(r"^[0-9+\-*/()\s]+$")


class ValidationError(Exception):
    def __init__(self, error_type):
        super().__init__(error_type)
        self.error_type = error_type


def _assistant_region(solution_str):
    text = solution_str or ""
    markers = ("<|im_start|>assistant", "Assistant:")
    for marker in markers:
        if marker in text:
            return text.rsplit(marker, 1)[1]
    return text


def extract_solution(solution_str):
    """Extract the expression inside the assistant's final answer tag."""
    response = _assistant_region(solution_str)
    matches = list(ANSWER_RE.finditer(response))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _new_details(expression=None, error_type=None):
    return {
        "answer_found": expression is not None,
        "format_ok": False,
        "parse_ok": False,
        "numbers_ok": False,
        "value_ok": False,
        "correct": False,
        "expression": expression,
        "value": None,
        "error_type": error_type,
    }


def _extract_with_format(solution_str, require_think=True):
    response = _assistant_region(solution_str)
    matches = list(ANSWER_RE.finditer(response))
    if not matches:
        return _new_details(error_type="missing_answer")

    answer_match = matches[-1]
    expression = answer_match.group(1).strip()
    details = _new_details(expression=expression)

    if len(matches) != 1:
        details["error_type"] = "multiple_answers"
        return details
    if not expression:
        details["error_type"] = "empty_answer"
        return details

    if require_think:
        think_matches = list(THINK_RE.finditer(response))
        if not think_matches:
            details["error_type"] = "missing_think"
            return details
        if not any(match.end() <= answer_match.start() for match in think_matches):
            details["error_type"] = "think_after_answer"
            return details

    details["format_ok"] = True
    return details


def _eval_node(node, used_numbers):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, used_numbers)

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, used_numbers)
        right = _eval_node(node.right, used_numbers)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValidationError("division_by_zero")
            return left / right
        raise ValidationError("unsupported_operator")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValidationError("non_integer_number")
        used_numbers.append(int(node.value))
        return Fraction(int(node.value), 1)

    # Python 3.8 compatibility when Num is still emitted by some tooling.
    if isinstance(node, ast.Num):
        if isinstance(node.n, bool) or not isinstance(node.n, int):
            raise ValidationError("non_integer_number")
        used_numbers.append(int(node.n))
        return Fraction(int(node.n), 1)

    raise ValidationError("unsupported_syntax")


def validate_expression(expression, numbers, target=24):
    """Validate and exactly evaluate a Game of 24 expression."""
    details = _new_details(expression=expression)
    details["answer_found"] = expression is not None
    details["format_ok"] = expression is not None

    if expression is None or not str(expression).strip():
        details["error_type"] = "empty_answer"
        return details

    expression = str(expression).strip()
    details["expression"] = expression

    if not ALLOWED_EXPRESSION_RE.fullmatch(expression):
        details["error_type"] = "invalid_characters"
        return details

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        details["error_type"] = "syntax_error"
        return details

    used_numbers = []
    try:
        value = _eval_node(tree, used_numbers)
    except ValidationError as exc:
        details["error_type"] = exc.error_type
        return details

    details["parse_ok"] = True
    details["value"] = str(value)

    expected = Counter(int(number) for number in numbers)
    observed = Counter(used_numbers)
    if observed != expected:
        details["numbers_ok"] = False
        details["error_type"] = "wrong_numbers"
        return details

    details["numbers_ok"] = True
    details["value_ok"] = value == Fraction(int(target), 1)
    if not details["value_ok"]:
        details["error_type"] = "wrong_value"
        return details

    details["correct"] = True
    details["error_type"] = None
    return details


def validate_solution(solution_str, ground_truth, require_think=True):
    target = ground_truth.get("target", 24)
    numbers = ground_truth["numbers"]
    format_details = _extract_with_format(solution_str, require_think=require_think)
    expression = format_details["expression"]

    if expression is None:
        return format_details

    expression_details = validate_expression(expression=expression, numbers=numbers, target=target)
    details = {**format_details, **expression_details}
    details["format_ok"] = format_details["format_ok"]
    details["answer_found"] = True
    details["correct"] = bool(
        details["format_ok"]
        and details["parse_ok"]
        and details["numbers_ok"]
        and details["value_ok"]
    )
    if not details["correct"] and format_details["error_type"] is not None:
        details["error_type"] = format_details["error_type"]
    return details


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0, return_details=False):
    """Rule reward for Game of 24.

    Scores:
    - 0.0: no extractable answer or severe format error
    - 0.1: answer format is present but the expression is wrong/invalid
    - 1.0: exact format, exact number multiset, legal AST, and exact value 24
    """
    require_think = method == "strict"
    details = validate_solution(solution_str, ground_truth, require_think=require_think)

    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print(f"Target: {ground_truth.get('target', 24)} | Numbers: {ground_truth['numbers']}")
        print(f"Extracted expression: {details['expression']}")
        print(f"Details: {details}")

    if details["correct"]:
        result = score
    elif details["format_ok"]:
        result = format_score
    else:
        result = 0.0

    if return_details:
        details = dict(details)
        details["reward"] = float(result)
        return details
    return float(result)
