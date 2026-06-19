"""Verify Game of 24 arithmetic expressions."""

import ast
import operator
from fractions import Fraction

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


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


def verify_expression(expression: str, numbers: tuple[int, int, int, int]) -> bool:
    """Check that an expression uses the given numbers once and equals 24."""

    try:
        tree = ast.parse(expression, mode="eval")
        value, used_numbers = _evaluate(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return False
    return sorted(used_numbers) == sorted(numbers) and value == 24
