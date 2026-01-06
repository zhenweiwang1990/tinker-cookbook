# Rollout 并发控制

## 概述

CUA RL 训练系统现在支持全局 rollout 并发控制，可以限制同时运行的 rollout 任务数量。这对于控制资源使用（如 GBox 实例数量）非常有用。

## 配置参数

在 `CLIConfig` 中新增了 `max_concurrent_rollouts` 参数：

```python
max_concurrent_rollouts: int = 8  # 默认值为 8
```

## 使用方式

### 方式 1: 命令行参数

```bash
python tinker_cookbook/recipes/cua_rl/core/train.py \
  --max-concurrent-rollouts 8 \
  --model-name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --other-params...
```

### 方式 2: 配置对象

```python
from tinker_cookbook.recipes.cua_rl.core.train import CLIConfig, cli_main
import asyncio

config = CLIConfig(
    model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    max_concurrent_rollouts=8,  # 设置最大并发为 8
    group_size=4,
    groups_per_batch=2,
    # ... 其他配置
)

asyncio.run(cli_main(config))
```

## 影响范围

该并发控制会应用于所有 rollout 场景：

1. **Baseline 评估**: 训练开始前的基线评估
2. **训练 Rollout**: 每个 training step 的 rollout
3. **定期评估**: 训练过程中的定期评估 (eval_every)

所有场景共享同一个 semaphore，确保全局并发不超过设定值。

## 工作原理

系统使用 `asyncio.Semaphore` 来控制并发：

```python
# 在 cli_main 中创建全局 semaphore
_rollout_semaphore = asyncio.Semaphore(cli_config.max_concurrent_rollouts)

# 在每个 rollout 中使用
async with _rollout_semaphore:
    result = await do_actual_rollout(...)
```

## 示例场景

### 场景 1: 大规模评估任务

如果您有 20 个评估任务，但只想同时运行 5 个：

```bash
python tinker_cookbook/recipes/cua_rl/core/train.py \
  --max-concurrent-rollouts 5 \
  --eval-tasks '{"source_type": "demo_eval", "limit": 20}'
```

### 场景 2: 资源受限环境

如果您的 GBox 资源有限，可以降低并发数：

```bash
python tinker_cookbook/recipes/cua_rl/core/train.py \
  --max-concurrent-rollouts 2 \
  --group-size 4 \
  --groups-per-batch 2
```

这样最多同时运行 2 个 rollout（每个 rollout 内部仍有 4 个任务）。

## 性能考虑

- **较高并发** (如 16-32): 适合有充足 GBox 资源的情况，可以加快训练速度
- **中等并发** (如 8): 默认值，平衡速度和资源使用
- **较低并发** (如 2-4): 适合资源受限或调试场景

## 注意事项

1. `max_concurrent_rollouts` 控制的是 **rollout 级别** 的并发，每个 rollout 内部仍会根据 `group_size` 并发执行多个任务
2. 实际的 GBox 实例数 = `max_concurrent_rollouts × group_size`（最坏情况）
3. 建议根据可用的 GBox 配额来设置这个值

## 日志输出

启动时会在日志中看到：

```
[Concurrency] Set max concurrent rollouts to 8
```

这确认了并发控制已经生效。

