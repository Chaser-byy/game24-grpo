"""Reward functions for GRPO training on the Game of 24 task.

Future content:
    - Correctness reward: 1.0 if the parsed solution is arithmetically valid
      and evaluates to 24, 0.0 otherwise.
    - Format reward: small bonus for adhering to the expected output structure.
    - Composite reward combining correctness, format, and optional shaping terms.
"""
