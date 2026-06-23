# 基于严格 RLVR/GRPO 的 24 点求解

本分支提供一套面向课程正式实验的实现：使用 Qwen2.5-1.5B-Instruct、TRL GRPO 和
LoRA，在无需奖励模型的情况下学习 24 点及 Countdown 动态目标算术。

核心原则：

- 最终正确性是严格的 `0/1` 可验证奖励，不使用“接近 24”奖励。
- 所有训练和评测共用“字符白名单 + AST + Fraction”验证器。
- 完整响应必须匹配 `<think>...</think><answer>...</answer>`。
- ToT 的 `indices 900:1000` 使用 Python 半开区间，恰好留出 100 道。
- baseline 与训练后模型必须在相同数据指纹上比较。
- 训练期间记录 reward、严格成功率、reward std、KL、completion length 和生成样本。

## 算法来源与边界

本实现采用以下思想：

- **DeepSeekMath / GRPO**：同一 prompt 采样一组 completion，使用组内相对奖励优化，
  并通过参考策略 KL 约束更新。
- **DeepSeek-R1 / RLVR**：使用无需奖励模型的规则奖励和 `<think>/<answer>` 输出协议。
- **TinyZero**：借鉴动态目标算术数据、规则验证、长推理、定期验证和定性 completion
  检查；没有复制其 veRL/Ray/vLLM 系统。
- **Tree of Thoughts**：使用其官方困难数据切片作为评测；当前训练和推理不是 ToT 搜索。
- **Logic-RL / open-r1**：借鉴格式奖励、结果奖励拆分和可复现实验组织。

GRPO 的张量级实现来自固定版本 `trl==0.15.2`。项目代码负责数据、prompt、奖励、验证、
监控和实验协议，不声称重新实现 TRL 内部优化器。

## 项目结构

```text
game24/
  data.py             # Game24/Countdown 统一数据模型与数据指纹
  parser.py           # 严格 R1 XML 响应解析
  verifier.py         # 字符白名单、AST、Fraction 精确验证
  rewards.py          # 共享奖励分解
  grpo_rewards.py     # TRL reward function 适配
  solver.py           # 无解题构造使用的精确动态规划求解器
  splits.py           # ToT 留出与无泄漏划分
  inference.py        # Qwen/LoRA 生成和多候选采样
  evaluation.py       # accuracy@1、pass@k、难度和无解评测
scripts/
  prepare_experiment_data.py # 正式 Game24 数据协议
  prepare_countdown_data.py  # Countdown OOD 扩展
  train_grpo.py              # 单 GPU LoRA-GRPO
  evaluate_baseline.py       # 基线/adapter 统一评测
  plot_results.py            # 训练曲线和同集前后对比
  summarize_experiments.py   # ID/OOD/无解结果汇总
configs/t4_grpo.json         # 单张 T4 的稳定配置
```

## 环境

推荐 Python 3.10、PyTorch 2.1 + CUDA 11.8 和单张 NVIDIA T4 16GB：

```bash
python -m venv /home/ma-user/work/venvs/game24
source /home/ma-user/work/venvs/game24/bin/activate
pip install -e ".[grpo,analysis]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

项目不主动安装或替换 PyTorch。模型建议放在持久化路径：

```bash
python scripts/download_model.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /home/ma-user/work/models/Qwen2.5-1.5B-Instruct
```

## 正式数据协议

当前官方 `nlile/24-game` 和 ToT CSV 含相同的 1362 个数字组合，只是排序不同。因此不能
把一个完整数据集训练后再把另一个完整数据集称作 OOD 测试，也不能删除“整个外部测试集”
的重叠，否则训练集会为空。

本项目采用：

1. ToT 零基索引 `[900:1000)` 作为 100 道未见困难测试。
2. 从 nlile 删除这 100 个组合，得到 `train_full=1262`。
3. 调参阶段从 1262 中固定抽取 100 道 ID validation，剩余 `train=1162`。
4. 超参数确定后可用 `train_full=1262` 重新训练，最终只评测困难留出集。
5. `1..13` 有重复四数组合共 1820 个；精确求解得到 1362 个可解、458 个无解，固定
   随机抽取其中 100 道作为无解测试。

准备本地原始文件后运行：

```bash
python scripts/prepare_experiment_data.py \
  --training-file /home/ma-user/work/data/nlile_24_game.jsonl \
  --ranked-file /home/ma-user/work/data/24.csv \
  --output-dir data/processed \
  --test-start 900 \
  --test-end 1000 \
  --validation-size 100 \
  --unsolvable-size 100 \
  --seed 42
```

期望输出：

```text
train.jsonl             1162  调参训练
validation_id.jsonl      100  周期验证
train_full.jsonl        1262  最终训练
test_hard.jsonl          100  唯一正式 Game24 困难测试
test_unsolvable.jsonl    100  瞎编/拒答测试
manifest.json                  数量、来源、切片和指纹
```

## 基线

所有前后对比必须使用相同的 `test_hard.jsonl`：

```bash
python scripts/evaluate_baseline.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/test_hard.jsonl \
  --output outputs/baseline_hard.jsonl \
  --num-samples 1
```

可额外测 stochastic pass@8：

```bash
python scripts/evaluate_baseline.py \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/test_hard.jsonl \
  --output outputs/baseline_hard_pass8.jsonl \
  --num-samples 8 \
  --sample
```

## 两阶段训练

先做两步硬件 smoke test：

```bash
python scripts/train_grpo.py \
  --config configs/t4_grpo.json \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/train.jsonl \
  --eval-data data/processed/validation_id.jsonl \
  --output outputs/grpo_smoke \
  --limit 16 \
  --eval-limit 8 \
  --max-steps 2
```

### Solver SFT warmup + GRPO

如果纯 GRPO 只学会了 `<think>/<answer>` 格式，但 `accuracy@1` 仍接近 0，说明严格正确性奖励
太稀疏，同组采样里很少出现正样本。此时可先用项目内精确 DP solver 生成训练题参考解，
做一个很短的 LoRA SFT warmup，再把 LoRA 合并成普通 CausalLM 作为 GRPO 初始策略。
这不改变最终评测协议；报告中应把它标注为 warm-start ablation。

```bash
python scripts/train_sft_warmup.py \
  --model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/train.jsonl \
  --output outputs/sft_warmup_4090 \
  --merged-output outputs/sft_warmup_4090_merged \
  --epochs 1 \
  --precision bf16 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5
```

随后从合并后的 warmup 模型继续 GRPO。RTX 4090 建议使用每题 16 个 generation，提高组内
发现正确表达式的概率：

```bash
python scripts/train_grpo.py \
  --config configs/rtx4090_grpo.json \
  --model outputs/sft_warmup_4090_merged \
  --data data/processed/train.jsonl \
  --eval-data data/processed/validation_id.jsonl \
  --output outputs/grpo_after_sft_4090
```

调参训练使用独立 ID validation：

```bash
python scripts/train_grpo.py \
  --config configs/t4_grpo.json \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/train.jsonl \
  --eval-data data/processed/validation_id.jsonl \
  --output outputs/grpo_tune \
  --epochs 1
```

确定超参数后使用全部 1262 道重新训练。不要再把其中的 validation 当作未见数据：

```bash
python scripts/train_grpo.py \
  --config configs/t4_grpo.json \
  --model /home/ma-user/work/models/Qwen2.5-1.5B-Instruct \
  --data data/processed/train_full.jsonl \
  --output outputs/grpo_final \
  --epochs 1
```

配置默认每题 4 个 generation、4 个 micro-batch 梯度累积、FP32 LoRA、显式 `beta=0.04`。
如果 T4 显存不足，先将 `num_generations` 降到 2；不要用缩短数据或更改测试集掩盖 OOM。
训练器只在无梯度 rollout 阶段临时切换到 eval mode 并启用 KV cache，随后恢复 train mode；
否则 gradient checkpointing 会让 T4 的逐 token 生成慢一个数量级。

## 训练监控

每个训练目录包含：

```text
run_config.json          完整超参数和依赖信息
train.log                运行日志
train_metrics.jsonl      reward、正确率、KL 等实时指标
completion_samples.jsonl 定期保存的定性生成样本
checkpoint-*             可恢复 checkpoint
adapter_config.json      最终 LoRA adapter
```

需要重点监控：

- `rewards/correctness_reward`：严格训练 solved rate，只有 0/1。
- `eval_rewards/correctness_reward`：周期 ID validation solved rate。
- `reward_std`：长期接近 0 表示同组回答没有差异，GRPO 几乎没有学习信号。
- `kl`：持续快速升高表示策略偏离基础模型，应降低学习率或增大 `beta`。
- `completion_length`：突然顶到上限通常意味着标签截断或无效长推理。
- `completion_samples.jsonl`：检查格式投机、重复模式和真实 self-verification。

## 正式评测

困难留出集 accuracy@1：

```bash
python scripts/evaluate_baseline.py \
  --model outputs/grpo_final \
  --data data/processed/test_hard.jsonl \
  --output outputs/grpo_final_hard.jsonl \
  --num-samples 1
```

无解题检查 `correct_abstention_rate` 和 `false_claim_rate`：

```bash
python scripts/evaluate_baseline.py \
  --model outputs/grpo_final \
  --data data/processed/test_unsolvable.jsonl \
  --output outputs/grpo_final_unsolvable.jsonl
```

生成同集前后对比和训练曲线：

```bash
python scripts/plot_results.py \
  --train-metrics outputs/grpo_tune/train_metrics.jsonl \
  --baseline-summary outputs/baseline_hard.summary.json \
  --grpo-summary outputs/grpo_final_hard.summary.json \
  --output-dir outputs/report_figures
```

脚本会验证两个 summary 的 `dataset_fingerprint`；不同测试集会直接报错。

## Countdown 任务 OOD

从本地 `Jiayi-Pan/Countdown-Tasks-3to4` 文件构造动态目标数据：

```bash
python scripts/prepare_countdown_data.py \
  --input-file /home/ma-user/work/data/countdown_3to4.jsonl \
  --output-dir data/processed/countdown \
  --validation-size 1024
```

零样本任务 OOD 评测：

```bash
python scripts/evaluate_baseline.py \
  --model outputs/grpo_final \
  --data data/processed/countdown/validation_ood.jsonl \
  --output outputs/grpo_final_countdown_ood.jsonl
```

也可以把 Countdown 训练数据作为第二个 `--data` 输入做多任务 GRPO；报告中必须将纯 Game24
和加入 Countdown 的结果分开，避免把额外训练数据带来的提升归因于算法。

## 本地验证

无模型/GPU 时运行：

```bash
pip install -e ".[dev]"
python scripts/smoke_test.py
python -m pytest -q
ruff check .
```

## 参考

- DeepSeekMath, arXiv:2402.03300
- DeepSeek-R1, arXiv:2501.12948
- Tree of Thoughts, arXiv:2305.10601
- Logic-RL, arXiv:2502.14768
- TinyZero: https://github.com/Jiayi-Pan/TinyZero
- TRL GRPO Trainer: https://huggingface.co/docs/trl/v0.15.2/grpo_trainer
- open-r1: https://github.com/huggingface/open-r1
