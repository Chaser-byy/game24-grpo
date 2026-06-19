# 基于 GRPO 的 24 点游戏求解

这是一个课程大作业项目，目标是让 Qwen2.5-1.5B-Instruct 学会根据四个数字生成等于
24 的算式，并使用可自动验证的奖励进行 GRPO 训练。

当前代码已包含最小闭环和 Qwen 单题推理入口，不包含 GRPO 训练：

```text
输入四个数字 -> 构造 Prompt -> Qwen 生成回答 -> 提取 <answer>
-> 验证数字和计算结果 -> 返回 0/1 奖励
```

## 目录

```text
game24/
  data.py       # 题目结构、JSONL 读写、去重和数据集重叠检查
  prompts.py    # 构造模型 Prompt
  parser.py     # 提取 <answer> 中的算式
  verifier.py   # 检查数字使用和计算结果
  rewards.py    # 0/1 奖励
data/
  sample.jsonl  # 最小示例数据
scripts/
  smoke_test.py # 不需要模型的端到端示例
  infer_once.py # Qwen 单题推理
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
函数。默认使用贪心生成，可通过 `--max-new-tokens` 调整最大输出长度。

## 后续接入 GRPO

安装与 GPU 环境兼容的 `datasets` 和 `trl`，然后把 `build_prompt` 与
`compute_reward` 接入 `trl.GRPOTrainer`。训练依赖暂不写入项目，避免提前引入
CUDA 和 PyTorch 版本冲突。

当前尚未实现真实数据集下载、GRPO 训练和测试集准确率评测。单题推理代码尚未在
NVIDIA GPU 和实际 Qwen 权重上运行验证。
