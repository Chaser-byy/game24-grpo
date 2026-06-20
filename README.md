# 基于 GRPO 的 24 点游戏求解

课程项目目标：使用 Qwen2.5-1.5B-Instruct 和规则奖励训练模型解决 24 点。项目采用
LoRA + TRL GRPO，面向单张 NVIDIA T4 16GB，不包含 SFT、DeepSpeed、vLLM 或多卡训练。

```text
本地数据 -> Prompt -> Qwen 生成 -> 答案提取 -> 规则验证/奖励
        -> GRPO + LoRA -> 评测 -> 图表
```

## 主要文件

```text
game24/
  data.py             # JSONL、原始数据规范化、去重
  prompts.py          # 24 点 Prompt
  parser.py           # 提取 <answer>
  verifier.py         # 数字和算术验证
  rewards.py          # 0/1 正确性奖励
  inference.py        # Qwen/LoRA 加载与生成
  evaluation.py       # 可复用批量评测
scripts/
  download_model.py   # 从 ModelScope 下载 Qwen
  download_data.py    # 从 ModelScope/Hugging Face 下载数据
  prepare_data.py     # 准备外部测试集
  prepare_train_data.py # 准备 train/validation/unsolvable
  evaluate_baseline.py  # 基线或 LoRA 评测
  train_grpo.py         # 单卡 LoRA GRPO
  plot_results.py       # 报告图表
```

## 1. 持久化环境

模型、数据、虚拟环境和输出都放在 `/home/ma-user/work`，避免 Notebook 重启后丢失：

```text
/home/ma-user/work/
  game24-grpo/
  models/Qwen2.5-1.5B-Instruct/
  data/game24.csv
  data/nlile_24_game.csv
  venvs/game24/
```

目标环境为 Python 3.10、PyTorch 2.1.0 + CUDA 11.8、Transformers 4.46.3。
项目不会主动安装或替换 PyTorch：

```bash
source /home/ma-user/work/venvs/game24/bin/activate
cd /home/ma-user/work/game24-grpo
pip install -e ".[grpo,analysis]"

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 2. 准备本地模型

华为云能访问 ModelScope 时：

```bash
pip install -e ".[download]"
python scripts/download_model.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /home/ma-user/work/models/Qwen2.5-1.5B-Instruct
```

若 ModelScope 也不可访问，在其他联网机器下载模型，通过 OBS 或浏览器上传到上述目录。
训练、评测命令都使用本地模型路径，不依赖 Hugging Face Hub。

## 3. 准备数据

华为云优先从 ModelScope 下载可用于 GRPO 的 1000 条 24 点训练题：

```bash
python scripts/download_data.py \
  --provider modelscope \
  --dataset cqupthzr/game24 \
  --output /home/ma-user/work/data/game24_train.jsonl
```

该数据集字段为 `nums` 和 `target`，脚本会补充 `solvable=true` 和数据来源。它适合完成
最小训练闭环，但不是课程指定 `nlile/24-game` 的同名镜像。ModelScope 下载使用公开 API，
不依赖 `modelscope.msdatasets` 子模块。若某台机器能访问 Hugging Face，可下载课程指定数据：

```bash
python scripts/download_data.py \
  --provider huggingface \
  --dataset nlile/24-game \
  --endpoint https://huggingface.co \
  --output /home/ma-user/work/data/nlile_24_game.jsonl
```

下载脚本只保存原始 JSONL，不会自动启动训练。若所有数据源都不可访问，请在其他联网机器
运行后通过 OBS 或浏览器上传文件。

将 ModelScope 数据划分为训练和验证集，用于确认训练闭环：

```bash
python scripts/prepare_train_data.py \
  --input-file /home/ma-user/work/data/game24_train.jsonl \
  --source cqupthzr/game24 \
  --output-dir data/processed \
  --val-ratio 0.1 \
  --seed 42
```

这批数据可能与下面的外部测试集重叠，因此不要用它在同一测试集上的结果作为正式报告指标。

先将 `test-time-compute/game-of-24` 的本地 CSV 转为统一测试集：

```bash
python scripts/prepare_data.py \
  --input-file /home/ma-user/work/data/game24.csv \
  --dataset test-time-compute/game-of-24 \
  --output data/processed/test.jsonl
```

正式实验使用 `nlile/24-game` 时，划分训练、验证和无解测试集，并排除与外部测试集重复
的数字组合：

```bash
python scripts/prepare_train_data.py \
  --input-file /home/ma-user/work/data/nlile_24_game.jsonl \
  --source nlile/24-game \
  --output-dir data/processed \
  --val-ratio 0.1 \
  --seed 42 \
  --test-file data/processed/test.jsonl
```

生成：

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/unsolvable_test.jsonl
data/processed/test.jsonl
```

## 4. 训练前基线

```bash
python scripts/evaluate_baseline.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/test.jsonl \
  --limit 20 \
  --output outputs/baseline_20.jsonl
```

输出明细和 `outputs/baseline_20.summary.json`。

## 5. 两步 Smoke Training

先用最小任务确认 CUDA、TRL、LoRA、生成和奖励都能运行：

```bash
python scripts/train_grpo.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/train.jsonl \
  --eval-data data/processed/validation.jsonl \
  --output outputs/grpo_2step \
  --limit 8 \
  --max-steps 2 \
  --num-generations 2 \
  --max-completion-length 128 \
  --eval-limit 5 \
  --run-eval
```

训练启动时会打印 Python、PyTorch、CUDA、Transformers、TRL、PEFT、GPU 和显存信息。
当前 PyTorch 2.1/T4 环境的 FP16 GRPO log-prob/KL 会出现 NaN，因此最小闭环默认使用
稳定的 FP32 LoRA 训练；基础模型被冻结，不会创建完整梯度和优化器状态。采样前还会清理并
重新归一化偶发的无效 logits。FP32 速度较慢，但优先保证课程实验能够正确更新参数。

## 6. 20 步最小闭环

```bash
python scripts/train_grpo.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/train.jsonl \
  --eval-data data/processed/validation.jsonl \
  --output outputs/grpo_smoke \
  --limit 20 \
  --max-steps 20 \
  --run-eval
```

默认每个 Prompt 生成 2 个回答、最大生成长度 128。奖励分别为答案提取 `0.1`、数字合法
`0.3`、正确结果 `1.0`。

## 7. 正式训练与续训

```bash
python scripts/train_grpo.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/train.jsonl \
  --eval-data data/processed/validation.jsonl \
  --output outputs/grpo_full \
  --limit 0 \
  --max-steps 200 \
  --learning-rate 1e-5 \
  --num-generations 2 \
  --max-completion-length 192 \
  --eval-limit 0 \
  --run-eval
```

从已有 checkpoint 继续：

```bash
python scripts/train_grpo.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/train.jsonl \
  --output outputs/grpo_full \
  --limit 0 \
  --max-steps 200 \
  --resume-from-checkpoint outputs/grpo_full/checkpoint-100
```

训练目录包含 LoRA adapter、tokenizer、checkpoint、`training_args.json`、
`train_metrics.jsonl`，启用 `--run-eval` 时还包含 `eval_results.jsonl` 和
`eval_summary.json`。

## 8. 完整、困难和无解集评测

完整测试集：

```bash
python scripts/evaluate_baseline.py \
  --model outputs/grpo_full \
  --data data/processed/test.jsonl \
  --limit 0 \
  --output outputs/grpo_full_test.jsonl
```

指定 rank 900–1000 的困难子集：

```bash
python scripts/evaluate_baseline.py \
  --model outputs/grpo_full \
  --data data/processed/test.jsonl \
  --rank-min 900 \
  --rank-max 1000 \
  --limit 0 \
  --output outputs/grpo_hard.jsonl
```

无解集（提取率可作为强行编造答案的参考）：

```bash
python scripts/evaluate_baseline.py \
  --model outputs/grpo_full \
  --data data/processed/unsolvable_test.jsonl \
  --limit 0 \
  --output outputs/grpo_unsolvable.jsonl
```

## 9. 生成报告图表

```bash
python scripts/plot_results.py \
  --train-metrics outputs/grpo_full/train_metrics.jsonl \
  --baseline-summary outputs/baseline_20.summary.json \
  --grpo-summary outputs/grpo_full/eval_summary.json \
  --output-dir outputs/grpo_full/figures
```

生成 reward、分项 reward、loss、训练前后对比 PNG，以及 `comparison_metrics.csv`。

## 本地检查

本地没有模型和 GPU 时只运行：

```bash
pip install -e ".[dev]"
python scripts/smoke_test.py
pytest -q
ruff check .
```
