# Rollout Concurrency Control

## Overview

The CUA RL training system now supports global rollout concurrency control, which limits the number of rollout tasks running simultaneously. This is useful for controlling resource usage (e.g., GBox instance count).

## Configuration Parameter

A new `max_concurrent_rollouts` parameter has been added to `CLIConfig`:

```python
max_concurrent_rollouts: int = 8  # Default value is 8
```

## Usage

### Method 1: Command Line Argument

```bash
python tinker_cookbook/recipes/cua_rl/core/train.py \
  --max-concurrent-rollouts 8 \
  --model-name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --other-params...
```

### Method 2: Configuration Object

```python
from tinker_cookbook.recipes.cua_rl.core.train import CLIConfig, cli_main
import asyncio

config = CLIConfig(
    model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    max_concurrent_rollouts=8,  # Set max concurrency to 8
    group_size=4,
    groups_per_batch=2,
    # ... other configurations
)

asyncio.run(cli_main(config))
```

## Scope of Impact

This concurrency control applies to all rollout scenarios:

1. **Baseline Evaluation**: Baseline evaluation before training starts
2. **Training Rollouts**: Rollouts at each training step
3. **Periodic Evaluations**: Periodic evaluations during training (eval_every)

All scenarios share the same semaphore, ensuring global concurrency does not exceed the configured limit.

## How It Works

The system uses `asyncio.Semaphore` to control concurrency:

```python
# Create global semaphore in cli_main
_rollout_semaphore = asyncio.Semaphore(cli_config.max_concurrent_rollouts)

# Use in each rollout
async with _rollout_semaphore:
    result = await do_actual_rollout(...)
```

## Example Scenarios

### Scenario 1: Large-Scale Evaluation Tasks

If you have 20 evaluation tasks but only want to run 5 simultaneously:

```bash
python tinker_cookbook/recipes/cua_rl/core/train.py \
  --max-concurrent-rollouts 5 \
  --eval-tasks '{"source_type": "demo_eval", "limit": 20}'
```

### Scenario 2: Resource-Constrained Environment

If your GBox resources are limited, you can reduce concurrency:

```bash
python tinker_cookbook/recipes/cua_rl/core/train.py \
  --max-concurrent-rollouts 2 \
  --group-size 4 \
  --groups-per-batch 2
```

This will run at most 2 rollouts simultaneously (each rollout still has 4 tasks internally).

## Performance Considerations

- **High Concurrency** (e.g., 16-32): Suitable when you have ample GBox resources, can speed up training
- **Medium Concurrency** (e.g., 8): Default value, balances speed and resource usage
- **Low Concurrency** (e.g., 2-4): Suitable for resource-constrained environments or debugging

## Important Notes

1. `max_concurrent_rollouts` controls **rollout-level** concurrency; each rollout still executes multiple tasks concurrently based on `group_size`
2. Actual GBox instance count = `max_concurrent_rollouts × group_size` (worst case)
3. It's recommended to set this value based on your available GBox quota

## Log Output

At startup, you will see in the logs:

```
[Concurrency] Set max concurrent rollouts to 8
```

This confirms that concurrency control is in effect.

