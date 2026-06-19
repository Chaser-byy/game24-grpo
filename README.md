# Game 24 — GRPO Training

Train [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
to solve the classic **Game of 24** puzzle using **Group Relative Policy
Optimization (GRPO)**.

In the Game of 24, the model is given four integers (e.g. `[3, 8, 3, 8]`) and
must produce an arithmetic expression using each number exactly once that
evaluates to 24 (e.g. `8 / (3 - 8 / 3) = 24`).

## Current Status

> **Project skeleton phase.**  
> Directory structure, tooling configuration, and module placeholders have been
> created. No data processing, solver, verifier, reward function, or GRPO
> training logic has been implemented yet.

## Directory Layout

```
game24-grpo/
├── configs/            # YAML configuration files
│   └── base.yaml       #   Base config: seed, accelerator, paths
├── data/
│   ├── raw/            #   Raw puzzle datasets (git-ignored except .gitkeep)
│   └── processed/      #   Train/val/test splits (git-ignored except .gitkeep)
├── game24/             #   Core Python package
│   ├── __init__.py     #     Package init, version
│   ├── schemas.py      #     Pydantic data models
│   ├── prompts.py      #     Prompt templates
│   ├── parser.py       #     Model output parser
│   ├── verifier.py     #     Solution verifier
│   ├── solver.py       #     Brute-force solver
│   ├── rewards.py      #     Reward functions
│   └── data.py         #     Data loading & generation
├── scripts/            #   Standalone scripts
│   ├── prepare_data.py #     Generate & preprocess puzzle datasets
│   └── smoke_test.py   #     End-to-end smoke test
├── tests/              #   Unit tests
│   ├── test_parser.py
│   ├── test_verifier.py
│   ├── test_solver.py
│   └── test_rewards.py
├── outputs/            #   Training outputs (git-ignored except .gitkeep)
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Requirements

- **Python**: 3.10 or higher
- **GPU**: NVIDIA CUDA (Huawei Cloud GPU instances or lab GPUs)
- Training dependencies (torch, transformers, datasets, trl, peft, accelerate,
  bitsandbytes, flash-attn) will be specified separately once the target CUDA
  environment is confirmed.

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests & Lint

```bash
pytest
ruff check .
```

> At this stage ``pytest`` will report "no tests ran" because the test files
> contain only module docstrings — no actual test functions have been written
> yet.

## Roadmap

1. **Data pipeline** — implement puzzle generation, solver, and dataset splits.
2. **Verification & rewards** — implement the correctness verifier, format
   parser, and composite reward functions.
3. **GRPO training loop** — integrate with `trl.GRPOTrainer` (or a custom loop)
   using Qwen2.5-1.5B-Instruct on NVIDIA GPUs.
4. **Evaluation & analysis** — benchmark against a held-out test set; log
   metrics and figures.
