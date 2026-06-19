#!/usr/bin/env python3
"""Prepare Game of 24 puzzle datasets for GRPO training.

This script will:
    1. Generate puzzle instances (four numbers, target 24).
    2. Filter to solvable puzzles.
    3. Create train / val / test splits.
    4. Save to ``data/processed/`` in a format suitable for Hugging Face
       ``datasets`` or direct PyTorch ``DataLoader`` consumption.

Usage (once implemented):
    python scripts/prepare_data.py --num-samples 100000

Status: NOT YET IMPLEMENTED — project skeleton phase.
"""
