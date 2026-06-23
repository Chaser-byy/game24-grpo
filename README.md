# Game24 GRPO on TinyZero

This project is a TinyZero-derived RLVR project for solving the Game of 24 with Qwen2.5-1.5B-Instruct, veRL, and GRPO. It keeps TinyZero as the engineering base and replaces the Countdown task with a strict 24-point arithmetic verifier, 24-point data preprocessing, GRPO launch scripts, evaluation, plotting, and tests.

Source base:

- TinyZero upstream: https://github.com/Jiayi-Pan/TinyZero
- TinyZero commit used for this branch: `95df88f2dcb05f33bd18da546531b52d0954c18b`
- vendored veRL package version file: `verl/version/version` = `0.1`
- dependency pins in this repo include `transformers<4.48` and `vllm<=0.6.3`

The training entry point is still `verl.trainer.main_ppo`, because that is how this TinyZero/veRL version exposes PPO-family trainers. The Game24 training script explicitly sets `algorithm.adv_estimator=grpo`, `actor_rollout_ref.rollout.n=8` by default, and `actor_rollout_ref.actor.use_kl_loss=True`. With `adv_estimator=grpo`, `RayPPOTrainer` sets `use_critic=False`, so the main Game24 GRPO run does not train a separate critic/value model.

## Why Game24 Fits TinyZero

Countdown asks the model to combine given numbers with arithmetic operators to hit a target. Game of 24 is the four-number, target-24 specialization of that setup. The important difference is stricter validation: a response is correct only when it uses each of the four input numbers exactly once, uses only `+ - * /` and parentheses, and evaluates exactly to 24.

The required output format is:

```text
<think>brief reasoning</think>
<answer>expression only</answer>
```

The `<answer>` content must be only the final expression. Do not include `=24` or explanatory text inside `<answer>`.

## Layout

```text
examples/data_preprocess/game24.py       # builds train/validation/test parquet files
verl/utils/reward_score/game24.py        # AST + Fraction rule verifier and reward
scripts/train_game24_grpo.sh             # main 24-point GRPO launch script
scripts/evaluate_game24.py               # unified baseline/checkpoint evaluator
scripts/plot_game24_metrics.py           # plots exported logs or console metrics
scripts/train_countdown_then_game24_grpo.sh # optional continuation from Countdown checkpoint
tests/test_game24_reward.py              # verifier unit tests
```

## Install

```bash
conda create -n game24-grpo python=3.9 -y
conda activate game24-grpo
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
pip install pytest matplotlib pyarrow
```

`flash-attn` can be installed separately when your CUDA/PyTorch stack supports it:

```bash
pip install flash-attn --no-build-isolation
```

## Data Preprocessing

The primary train source is `nlile/24-game` with `solvable=True`. The hard test source is `test-time-compute/game-of-24`, using the Tree-of-Thoughts-style rank/index 900-1000 region. Splits are keyed by `tuple(sorted(numbers))` so test combinations do not leak into train.

```bash
python examples/data_preprocess/game24.py \
  --local_dir data/game24 \
  --template_type qwen-instruct \
  --validation_size 128 \
  --test_id_size 256 \
  --test_hard_start 900 \
  --test_hard_end 1000
```

Outputs:

```text
data/game24/train.parquet
data/game24/validation.parquet
data/game24/test_id.parquet
data/game24/test_hard.parquet
data/game24/test_unsolvable.parquet
```

The script prints raw counts, solvable/unsolvable counts, duplicate counts, split sizes, and overlap exclusions. If the primary dataset has too few `solvable=False` rows, the script can synthesize unsolvable 1-13 combinations with the same exact arithmetic solver; this fallback is reported through the source field.

## Unit Tests

```bash
pytest tests/test_game24_reward.py
```

The tests cover correct answers, wrong values, missing numbers, repeated numbers, numbers outside the prompt, duplicate input numbers, multi-digit numbers, division by zero, illegal characters, function calls, power operators, missing answer tags, explanatory text inside answers, and fraction intermediates.

## Baseline Evaluation

Evaluate the untrained base model:

```bash
python scripts/evaluate_game24.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data_dir data/game24 \
  --output_dir outputs/baseline_qwen25_15b \
  --pass_at 1 4 8
```

The evaluator reports `format_rate`, `parse_rate`, `number_use_rate`, `valid_expression_rate`, `solved_rate/pass@1`, `pass@4`, `pass@8`, `average_reward`, and `average_response_length` for `validation`, `test_id`, `test_hard`, and `test_unsolvable`. It also saves per-sample JSONL with numbers, target, model output, extracted answer, reward, correctness, error type, and split.

## GRPO Training

The main experiment is 24-point-only GRPO from Qwen2.5-1.5B-Instruct:

```bash
export MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export DATA_DIR=$PWD/data/game24
export OUTPUT_DIR=$PWD/checkpoints
export N_GPUS=1
export ROLLOUT_TP_SIZE=1
export ROLLOUT_N=8
export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE=8
export TOTAL_EPOCHS=3
export SAVE_FREQ=50
export TEST_FREQ=25
export EXPERIMENT_NAME=qwen2_5_1_5b_game24_grpo

bash scripts/train_game24_grpo.sh 2>&1 | tee outputs/train_game24_grpo.log
```

Important settings in the script:

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
```

The local development environment is not assumed to have enough GPU memory for a full run. The codebase supports data preprocessing, verifier tests, script/config checks, and small dry runs locally; full GRPO training should be run on a GPU server.

## Checkpoint Evaluation

After training, evaluate the saved actor checkpoint:

```bash
python scripts/evaluate_game24.py \
  --model checkpoints/qwen2_5_1_5b_game24_grpo/actor/global_step_50 \
  --data_dir data/game24 \
  --output_dir outputs/grpo_step50_eval \
  --pass_at 1 4 8
```

Use the same evaluator for the base model and all checkpoints so the metrics are comparable.

## Plot Curves

For a console log produced with `tee`, or a W&B CSV/JSONL export:

```bash
python scripts/plot_game24_metrics.py \
  --input outputs/train_game24_grpo.log \
  --output_dir outputs/game24_plots
```

The trainer logs `game24/format_rate`, `game24/solved_rate`, `game24/group_reward_std_mean`, and `game24/group_reward_zero_std_rate` in addition to the existing veRL reward, KL, response length, learning rate, gradient norm, policy loss, and validation metrics. The plotting script draws these keys when they are present in the log/export.

## Optional Countdown Warm Start

Experiment C should remain separate from the main 24-point-only GRPO experiment. First train a Countdown GRPO checkpoint, then continue with Game24:

```bash
export COUNTDOWN_ACTOR_CKPT=/path/to/countdown/actor/global_step_x
export EXPERIMENT_NAME=countdown_then_game24_grpo
bash scripts/train_countdown_then_game24_grpo.sh
```

Report this separately as `Countdown GRPO -> Game24 GRPO`; do not mix Countdown data into the main Game24-only run.

## Unsolvable Evaluation

`test_unsolvable.parquet` uses `solvable=False` examples when available, with an exact-solver fallback for missing rows. The current prompt asks for an expression only, so abstention is not trained as the primary behavior. The evaluator still records `false_solution_rate`, `abstention_rate` when a model emits `NO_SOLUTION`, and `invalid_answer_rate` for analysis.

## Known Limits

- Full GRPO training is not run by this repository setup step; it requires a suitable GPU server.
- TinyZero's bundled veRL is older than current upstream veRL, so this project keeps a minimal TinyZero-compatible integration instead of rewriting the trainer stack.
- The main reward is sparse: `1.0` for a fully valid solution, `0.1` for an answer-tagged but wrong/invalid expression, and `0.0` for missing or severe format errors.
- The Game24-specific trainer metrics are intentionally lightweight and are derived from the scalar rule reward; deeper rollout diagnostics can still be added later if needed.

## Citation

```text
TinyZero: https://github.com/Jiayi-Pan/TinyZero
veRL: https://github.com/volcengine/verl
Qwen2.5: https://github.com/QwenLM/Qwen2.5
```
