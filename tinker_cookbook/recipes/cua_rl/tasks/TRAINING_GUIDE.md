# 使用任务适配器进行训练

本指南说明如何使用任务适配器自动发现所有任务并开始训练。

## 快速开始

### 1. 使用任务适配器自动分割任务（推荐）

任务适配器会自动发现 `tasks/` 目录下的所有任务，并按固定种子（seed=42）随机分为 80% 训练集和 20% 评估集。

#### 训练命令

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "task_adapter", "split_type": "train", "seed": 42}' \
    --eval_tasks '{"source_type": "task_adapter", "split_type": "eval", "seed": 42}' \
    --group_size 4 \
    --groups_per_batch 10 \
    --learning_rate 1e-5 \
    --eval_every 10 \
    --save_every 10
```

#### 使用环境变量设置 API Key

```bash
export GBOX_API_KEY="your-gbox-api-key"
export TINKER_API_KEY="your-tinker-api-key"  # 通常与 GBOX_API_KEY 相同

python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "task_adapter", "split_type": "train", "seed": 42}' \
    --eval_tasks '{"source_type": "task_adapter", "split_type": "eval", "seed": 42}' \
    --group_size 4 \
    --groups_per_batch 10
```

### 2. 使用所有任务（不分割）

如果你想使用所有任务进行训练（不分割为训练/评估集）：

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "task_adapter"}' \
    --group_size 4 \
    --groups_per_batch 10
```

### 3. 自定义分割比例和种子

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "task_adapter", "split_type": "train", "train_ratio": 0.75, "seed": 123}' \
    --eval_tasks '{"source_type": "task_adapter", "split_type": "eval", "train_ratio": 0.75, "seed": 123}' \
    --group_size 4 \
    --groups_per_batch 10
```

### 4. 指定任务目录

如果任务不在默认位置：

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "task_adapter", "split_type": "train", "tasks_dir": "/path/to/tasks", "seed": 42}' \
    --eval_tasks '{"source_type": "task_adapter", "split_type": "eval", "tasks_dir": "/path/to/tasks", "seed": 42}' \
    --group_size 4 \
    --groups_per_batch 10
```

## 完整训练示例

### 基本训练配置

```bash
export GBOX_API_KEY="your-gbox-api-key"
export TINKER_API_KEY="your-tinker-api-key"

python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --lora_rank 32 \
    --tasks '{"source_type": "task_adapter", "split_type": "train", "seed": 42}' \
    --eval_tasks '{"source_type": "task_adapter", "split_type": "eval", "seed": 42}' \
    --group_size 4 \
    --groups_per_batch 10 \
    --learning_rate 1e-5 \
    --max_turns 20 \
    --eval_every 10 \
    --save_every 10 \
    --seed 42 \
    --log_dir "./logs"
```

### 高级配置

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --lora_rank 32 \
    --tasks '{"source_type": "task_adapter", "split_type": "train", "seed": 42}' \
    --eval_tasks '{"source_type": "task_adapter", "split_type": "eval", "seed": 42}' \
    --group_size 4 \
    --groups_per_batch 10 \
    --learning_rate 1e-5 \
    --max_tokens 2048 \
    --temperature 1.0 \
    --max_turns 20 \
    --eval_every 5 \
    --save_every 5 \
    --seed 42 \
    --wandb_project "cua-rl-training" \
    --wandb_name "airbnb-instagram-tasks" \
    --log_dir "./logs"
```

## 任务适配器配置选项

### TaskSourceConfig 参数

- `source_type`: 必须为 `"task_adapter"`
- `split_type`: `"train"` (训练集) 或 `"eval"` (评估集)，如果为 `None` 则使用所有任务
- `tasks_dir`: 任务目录路径（可选，默认自动检测）
- `train_ratio`: 训练集比例（默认 0.8，即 80% 训练，20% 评估）
- `seed`: 随机种子（默认 42，用于可重复的分割）
- `limit`: 限制任务数量（可选）

### 示例配置

```json
{
  "source_type": "task_adapter",
  "split_type": "train",
  "train_ratio": 0.8,
  "seed": 42
}
```

## 验证任务加载

在开始训练前，你可以先验证任务是否正确加载：

```python
from tinker_cookbook.recipes.cua_rl.tasks.task_adapter import TaskAdapter

adapter = TaskAdapter(seed=42)
train_tasks = adapter.get_train_tasks()
eval_tasks = adapter.get_eval_tasks()

print(f"训练任务: {len(train_tasks)}")
print(f"评估任务: {len(eval_tasks)}")

# 查看前几个训练任务
for i, task_info in enumerate(train_tasks[:5], 1):
    task = task_info["task_instance"]
    print(f"{i}. {task.name}: {task.description[:80]}...")
```

## 注意事项

1. **固定种子**: 使用相同的 `seed` 值可以确保每次运行都得到相同的训练/评估分割
2. **任务发现**: 任务适配器会自动发现所有包含 `create_task()` 函数的 `task.py` 文件
3. **验证**: 当前任务适配器加载的任务没有验证逻辑（`validation_query=None`），如果需要验证，需要在任务定义中实现
4. **GBox 模式**: 确保 `AdbClient` 使用 GBox 模式时，传入正确的 `gbox_client` 或 `gbox_box` 参数

## 故障排除

### 任务未发现

如果任务未被发现，检查：
1. 任务目录结构是否正确（每个任务应该有 `task.py` 文件）
2. `task.py` 中是否有 `create_task()` 函数
3. 模块路径是否正确

### API Key 错误

确保设置了环境变量：
```bash
export GBOX_API_KEY="your-key"
export TINKER_API_KEY="your-key"
```

或在命令中直接指定：
```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --gbox_api_key "your-key" \
    --tinker_api_key "your-key" \
    ...
```

