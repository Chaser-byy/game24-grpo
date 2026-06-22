"""Strict, deterministic verification for arithmetic RLVR tasks."""

import ast
import operator
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
ALLOWED_CHARACTERS = re.compile(r"[0-9+\-*/()\s]+")
MAX_EXPRESSION_LENGTH = 256
MAX_AST_NODES = 64


@dataclass(frozen=True)
class VerificationResult:
    """Diagnostics from checking one arithmetic expression."""

    valid: bool
    used_numbers: list[int]
    value: Fraction | None
    reason: str
    syntax_valid: bool = False
    numbers_valid: bool = False
    target_valid: bool = False


def _evaluate(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        if node.value < 0:
            raise ValueError("integer literals must be non-negative")
        return Fraction(node.value), [node.value]
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, left_numbers = _evaluate(node.left)
        right, right_numbers = _evaluate(node.right)
        return OPERATORS[type(node.op)](left, right), left_numbers + right_numbers
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value, numbers = _evaluate(node.operand)
        return (-value if isinstance(node.op, ast.USub) else value), numbers
    raise ValueError("only integers and +, -, *, /, parentheses are allowed")


def check_expression(
    expression: str,
    numbers: Sequence[int],
    target: int = 24,
    *,
    tolerance: float = 1e-6,
) -> VerificationResult:
    """Check characters, syntax, number multiset, and target value in that order."""

    if not expression or len(expression) > MAX_EXPRESSION_LENGTH:
        return VerificationResult(False, [], None, "expression is empty or too long")
    if ALLOWED_CHARACTERS.fullmatch(expression) is None:
        return VerificationResult(False, [], None, "expression contains forbidden characters")

    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise ValueError("expression is too complex")
        value, used_numbers = _evaluate(tree.body)
    except SyntaxError:
        return VerificationResult(False, [], None, "invalid expression syntax")
    except ZeroDivisionError:
        return VerificationResult(False, [], None, "division by zero")
    except (RecursionError, ValueError) as error:
        return VerificationResult(False, [], None, str(error))

    expected = sorted(int(number) for number in numbers)
    numbers_valid = sorted(used_numbers) == expected
    if not numbers_valid:
        reason = f"expected numbers {expected}, but used {sorted(used_numbers)}"
        return VerificationResult(False, used_numbers, value, reason, True, False, False)

    target_valid = abs(float(value) - float(target)) <= tolerance
    if not target_valid:
        reason = f"expression equals {value}, not {target}"
        return VerificationResult(False, used_numbers, value, reason, True, True, False)
    return VerificationResult(True, used_numbers, value, "ok", True, True, True)


def verify_expression(expression: str, numbers: Sequence[int], target: int = 24) -> bool:
    """Return whether an expression uses every number once and reaches the target."""

    return check_expression(expression, numbers, target).valid
