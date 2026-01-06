# Training with Task Adapter

This guide explains how to use the task adapter to automatically discover all tasks and start training.

## Quick Start

### 1. Using Task Adapter with Automatic Split (Recommended)

The task adapter automatically discovers all tasks in the `tasks/` directory and randomly splits them into 80% training set and 20% evaluation set using a fixed seed (seed=42).

#### Training Command

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

#### Setting API Keys via Environment Variables

```bash
export GBOX_API_KEY="your-gbox-api-key"
export TINKER_API_KEY="your-tinker-api-key"  # Usually same as GBOX_API_KEY

python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "task_adapter", "split_type": "train", "seed": 42}' \
    --eval_tasks '{"source_type": "task_adapter", "split_type": "eval", "seed": 42}' \
    --group_size 4 \
    --groups_per_batch 10
```

### 2. Using All Tasks (No Split)

If you want to use all tasks for training (without splitting into train/eval sets):

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "task_adapter"}' \
    --group_size 4 \
    --groups_per_batch 10
```

### 3. Custom Split Ratio and Seed

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "task_adapter", "split_type": "train", "train_ratio": 0.75, "seed": 123}' \
    --eval_tasks '{"source_type": "task_adapter", "split_type": "eval", "train_ratio": 0.75, "seed": 123}' \
    --group_size 4 \
    --groups_per_batch 10
```

### 4. Specifying Task Directory

If tasks are not in the default location:

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "task_adapter", "split_type": "train", "tasks_dir": "/path/to/tasks", "seed": 42}' \
    --eval_tasks '{"source_type": "task_adapter", "split_type": "eval", "tasks_dir": "/path/to/tasks", "seed": 42}' \
    --group_size 4 \
    --groups_per_batch 10
```

## Complete Training Examples

### Basic Training Configuration

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

### Advanced Configuration

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

## Task Adapter Configuration Options

### TaskSourceConfig Parameters

- `source_type`: Must be `"task_adapter"`
- `split_type`: `"train"` (training set) or `"eval"` (evaluation set), or `None` to use all tasks
- `tasks_dir`: Task directory path (optional, defaults to auto-detection)
- `train_ratio`: Training set ratio (default 0.8, i.e., 80% train, 20% eval)
- `seed`: Random seed (default 42, for reproducible splits)
- `limit`: Limit number of tasks (optional)

### Example Configuration

```json
{
  "source_type": "task_adapter",
  "split_type": "train",
  "train_ratio": 0.8,
  "seed": 42
}
```

## Verifying Task Loading

Before starting training, you can verify that tasks are loaded correctly:

```python
from tinker_cookbook.recipes.cua_rl.tasks.task_adapter import TaskAdapter

adapter = TaskAdapter(seed=42)
train_tasks = adapter.get_train_tasks()
eval_tasks = adapter.get_eval_tasks()

print(f"Training tasks: {len(train_tasks)}")
print(f"Evaluation tasks: {len(eval_tasks)}")

# View first few training tasks
for i, task_info in enumerate(train_tasks[:5], 1):
    task = task_info["task_instance"]
    print(f"{i}. {task.name}: {task.description[:80]}...")
```

## Notes

1. **Fixed Seed**: Using the same `seed` value ensures you get the same train/eval split on every run
2. **Task Discovery**: The task adapter automatically discovers all `task.py` files containing a `create_task()` function
3. **Validation**: Currently, tasks loaded by the task adapter have no validation logic (`validation_query=None`). If validation is needed, it must be implemented in the task definition
4. **GBox Mode**: Ensure `AdbClient` uses GBox mode by passing the correct `gbox_client` or `gbox_box` parameters

## Troubleshooting

### Tasks Not Discovered

If tasks are not discovered, check:
1. Task directory structure is correct (each task should have a `task.py` file)
2. `task.py` contains a `create_task()` function
3. Module paths are correct

### API Key Errors

Make sure environment variables are set:
```bash
export GBOX_API_KEY="your-key"
export TINKER_API_KEY="your-key"
```

Or specify directly in the command:
```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --gbox_api_key "your-key" \
    --tinker_api_key "your-key" \
    ...
```
