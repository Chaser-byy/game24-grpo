"""Verify Game of 24 arithmetic expressions."""

import ast
import operator
from dataclasses import dataclass
from fractions import Fraction

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


@dataclass
class VerificationResult:
    """Useful details from checking one expression."""

    valid: bool
    used_numbers: list[int]
    value: Fraction | None
    reason: str


def _evaluate(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value), [node.value]
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, left_numbers = _evaluate(node.left)
        right, right_numbers = _evaluate(node.right)
        return OPERATORS[type(node.op)](left, right), left_numbers + right_numbers
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value, numbers = _evaluate(node.operand)
        return (-value if isinstance(node.op, ast.USub) else value), numbers
    raise ValueError("only numbers and +, -, *, / are allowed")


def check_expression(
    expression: str, numbers: tuple[int, int, int, int]
) -> VerificationResult:
    """Check an expression and return details useful for debugging."""

    try:
        tree = ast.parse(expression, mode="eval")
        value, used_numbers = _evaluate(tree.body)
    except SyntaxError:
        return VerificationResult(False, [], None, "invalid expression syntax")
    except ZeroDivisionError:
        return VerificationResult(False, [], None, "division by zero")
    except ValueError as error:
        return VerificationResult(False, [], None, str(error))

    if sorted(used_numbers) != sorted(numbers):
        reason = f"expected numbers {sorted(numbers)}, but used {sorted(used_numbers)}"
        return VerificationResult(False, used_numbers, value, reason)
    if value != 24:
        return VerificationResult(False, used_numbers, value, f"expression equals {value}, not 24")
    return VerificationResult(True, used_numbers, value, "ok")


def verify_expression(expression: str, numbers: tuple[int, int, int, int]) -> bool:
    """Return whether an expression uses the given numbers once and equals 24."""

    return check_expression(expression, numbers).valid
