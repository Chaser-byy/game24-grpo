# 基于 GRPO 的 24 点游戏求解

这是一个课程大作业项目，目标是让 Qwen2.5-1.5B-Instruct 学会根据四个数字生成等于
24 的算式，并使用可自动验证的奖励进行 GRPO 训练。

当前代码包含数据准备、Qwen 推理、基线评测和最小 LoRA GRPO 训练闭环：

```text
输入四个数字 -> 构造 Prompt -> Qwen 生成回答 -> 提取 <answer>
-> 验证数字和计算结果 -> 返回 0/1 奖励
```

## 目录

```text
game24/
  data.py       # 题目结构、JSONL 读写、去重和数据集重叠检查
  inference.py  # 单卡 Qwen 加载和生成
  prompts.py    # 构造模型 Prompt
  parser.py     # 提取 <answer> 中的算式
  verifier.py   # 检查数字使用和计算结果
  rewards.py    # 0/1 奖励
data/
  sample.jsonl  # 最小示例数据
scripts/
  smoke_test.py # 不需要模型的端到端示例
  infer_once.py # Qwen 单题推理
  prepare_data.py      # 下载或转换数据集
  evaluate_baseline.py # 批量基线评测
  train_grpo.py        # 单卡 LoRA GRPO 训练
tests/
  test_pipeline.py
```

## 安装与运行

需要 Python 3.10 或更高版本。当前核心流程只使用 Python 标准库。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/smoke_test.py
pytest -q
```

Smoke test 使用固定字符串模拟模型输出，因此不需要 GPU。

## Qwen 单题推理

在华为云 ModelArts 或实验室的 NVIDIA GPU 环境中，先使用镜像预装的 CUDA 和
PyTorch，再安装项目的轻量推理依赖。项目不会指定或重新安装某个 PyTorch/CUDA 版本。
项目固定使用 Transformers 4.46.3；安装项目依赖时不会主动安装或替换 PyTorch。

```bash
pip install -e ".[inference]"

# 使用本地模型目录
python scripts/infer_once.py \
  --model ./models/Qwen2.5-1.5B-Instruct \
  --numbers 1 3 4 6

# 或直接使用 Hugging Face 模型名称（首次运行会下载模型）
python scripts/infer_once.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --numbers 1 3 4 6
```

脚本使用 Qwen 聊天模板，只解码新生成的 token，然后复用项目的答案提取、验证和奖励
函数。默认使用确定性的贪心生成，可通过 `--max-new-tokens` 调整最大输出长度。需要采样时
添加 `--sample`，此时使用 `temperature=0.7` 和 `top_p=0.9`。

## 准备数据

数据脚本优先支持本地 CSV，同时保留 Hugging Face 在线加载。它会统一字段并按排序后的
四个数字去重，同一数字组合保留第一次出现的记录。官方 CSV 的 `Rank` 和
`Solved rate` 会一并保存，便于后续分析困难题。

```bash
# 华为云无法连接 Hugging Face 时，不需要安装 datasets
python scripts/prepare_data.py \
  --input-file /home/ma-user/work/data/game24.csv \
  --dataset test-time-compute/game-of-24 \
  --output data/processed/test.jsonl

# 能连接 Hugging Face 时再安装 datasets
pip install -e ".[data]"

python scripts/prepare_data.py \
  --dataset nlile/24-game \
  --output data/processed/nlile.jsonl

python scripts/prepare_data.py \
  --dataset test-time-compute/game-of-24 \
  --output data/processed/test.jsonl

```

本地 CSV 路径不会导入或访问 Hugging Face。可用 `--split` 选择在线数据集 split，
用 `--limit` 只转换前几条记录。

## 基线评测

在预装 CUDA 和 PyTorch 的 NVIDIA GPU 镜像中安装其余依赖：

```bash
pip install -e ".[inference,data]"
```

先跑 20 道题确认流程：

```bash
python scripts/evaluate_baseline.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/test.jsonl \
  --limit 20 \
  --output outputs/baseline_20.jsonl
```

完整评测使用 `--limit 0`：

```bash
python scripts/evaluate_baseline.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/test.jsonl \
  --limit 0 \
  --output outputs/baseline_full.jsonl
```

逐题结果保存在指定的 JSONL 文件中，汇总指标自动保存在同目录的
`baseline_20.summary.json` 或 `baseline_full.summary.json`。评测支持
`--max-new-tokens`、`--sample` 和 `--seed`。

## GRPO Smoke Training

训练环境使用已有的 PyTorch 2.1.0 + CUDA 11.8，再安装固定版本的训练依赖：

```bash
pip install -e ".[grpo]"
```

下面的命令用 20 条可解题目训练 20 步，默认每个 Prompt 生成 4 个回答：

```bash
python scripts/train_grpo.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/train.jsonl \
  --output outputs/grpo_smoke \
  --limit 20 \
  --max-steps 20
```

训练日志分别显示答案提取、数字使用和最终正确性奖励。LoRA adapter 保存在
`outputs/grpo_smoke/`，训练 checkpoint 也保存在该目录下。可调整
`--learning-rate`、`--num-generations`、`--max-completion-length` 和 `--seed`。

现有评测脚本可以直接读取 adapter 目录，并从 `adapter_config.json` 找到原始模型：

```bash
python scripts/evaluate_baseline.py \
  --model outputs/grpo_smoke \
  --data data/processed/test.jsonl \
  --limit 20 \
  --output outputs/grpo_smoke_eval.jsonl
```

当前 GRPO 脚本只面向单张 NVIDIA GPU，尚未在本地运行训练或验证 T4 显存占用。
