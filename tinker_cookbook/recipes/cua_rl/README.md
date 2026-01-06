# CUA RL Recipe

This recipe implements RL training for CUA (Computer Use Agent) using Tinker and GBoxAgent.

## Overview

CUA RL trains a vision-language model to interact with device screens (Android/Linux) to complete tasks. The training uses:

- **Rollout**: GBoxAgent with Tinker's OpenAI-compatible API (supports vision via `input_text` and `input_image`)
- **Training**: Tinker's RL framework with GRPO algorithm (group-based advantage centering)
- **Vision Inputs**: Screenshots are converted from OpenAI format to Tinker's `ImageChunk` format for training
- **On-Policy RL**: The rollout model dynamically updates as training progresses using the current checkpoint

## Installation

### Dependencies

The recipe requires `gbox-agent` to be installed. Add it to your `pyproject.toml`:

```toml
dependencies = [
    # ... other dependencies ...
    "gbox-agent @ git+https://github.com/your-org/gbox-agent.git",
]
```

Or install directly:

```bash
pip install git+https://github.com/your-org/gbox-agent.git
```

### Environment Variables

Set the following environment variables:

```bash
export GBOX_API_KEY="your-gbox-api-key"
export TINKER_API_KEY="your-tinker-api-key"  # Usually same as GBOX_API_KEY
```

Note: `TINKER_API_KEY` is used for Tinker's OpenAI-compatible API during rollout.

## Usage

### Basic Training

```bash
uv run python -m tinker_cookbook.recipes.cua_rl.train 
```

### Task Configuration

Tasks are configured using `TaskSourceConfig` dict(s) via the `tasks` parameter. This provides flexible task loading from various sources.

#### Using Demo Tasks

```python
# All demo training tasks
tasks = {"source_type": "demo_training"}

# Filter by category
tasks = {
    "source_type": "demo_training",
    "category": "settings"
}

# Filter by difficulty
tasks = {
    "source_type": "demo_training",
    "difficulty": "easy"
}

# Combine filters with limit
tasks = {
    "source_type": "demo_training",
    "category": "settings",
    "difficulty": "easy",
    "limit": 10,
    "seed": 42  # For reproducible sampling
}
```

#### Using Specific Task IDs

```python
tasks = {
    "source_type": "ids",
    "task_ids": ["train_01_open_settings", "train_02_enable_wifi"]
}
```

#### Custom Tasks

```python
tasks = {
    "source_type": "custom",
    "custom_tasks": [
        "Open the Settings app",
        "Enable WiFi",
        "Set brightness to maximum"
    ]
}
```

#### Multiple Sources

```python
# Combine multiple sources
tasks = [
    {"source_type": "demo_training", "category": "settings", "limit": 20},
    {"source_type": "demo_training", "category": "navigation", "limit": 10}
]
```

#### Evaluation Tasks

Evaluation tasks are configured separately using `eval_tasks` with the same format:

```python
eval_tasks = {"source_type": "demo_eval"}
```

See `TASK_CONFIG_QUICKSTART.md` and `task_config_examples.md` for more examples.

### Configuration Options

#### Model Configuration

- `model_name`: Model to train (e.g., `Qwen/Qwen3-VL-30B-A3B-Instruct`)
  - **Also used for rollout**: The rollout model dynamically updates as training progresses (on-policy RL)
  - Uses Tinker's OpenAI-compatible API with checkpoint paths (e.g., `tinker://.../sampler_weights/000080`)
- `lora_rank`: LoRA rank for training (default: 32)
- `renderer_name`: Renderer name (auto-detected if not specified)
- `load_checkpoint_path`: Path to checkpoint to resume training from (optional)

#### GBox Configuration

- `gbox_api_key`: GBox API key (or use `GBOX_API_KEY` env var)
- `tinker_api_key`: Tinker API key for OpenAI-compatible API (or use `TINKER_API_KEY` env var, defaults to `gbox_api_key`)
- `box_type`: Type of GBox environment (`android` or `linux`, default: `android`)
- `max_turns`: Maximum number of turns per task (default: 20)

#### Task Configuration

- `tasks`: TaskSourceConfig dict(s) for training (required)
  - Single dict: `{"source_type": "demo_training"}`
  - List of dicts: `[{"source_type": "demo_training"}, {"source_type": "demo_eval"}]`
- `eval_tasks`: Optional TaskSourceConfig dict(s) for evaluation
- `seed`: Random seed for task shuffling (default: 0)

#### Training Hyperparameters

- `group_size`: Number of parallel environments per group (default: 4)
- `groups_per_batch`: Number of groups per batch (default: 10)
- `learning_rate`: Learning rate (default: 1e-5)
- `max_tokens`: Maximum tokens per generation (default: 2048)
- `temperature`: Sampling temperature (default: 1.0)
- `kl_penalty_coef`: KL penalty coefficient (default: 0.0)
- `num_substeps`: Number of substeps per batch (default: 1)

#### Async Configuration

- `max_steps_off_policy`: Maximum steps to run off-policy before syncing (optional)
  - If set, enables async training where rollout and training can overlap
  - See Tinker's async training docs for details

#### Logging

- `log_path`: Path to log directory (default: `/tmp/tinker-examples/cua_rl/{run_name}`)
- `log_dir`: Base directory for log path (alternative to `log_path`)
- `wandb_project`: Weights & Biases project name
- `wandb_name`: Weights & Biases run name
- `eval_every`: Evaluate every N batches (default: 10)
- `save_every`: Save checkpoint every N batches (default: 10)
- `num_groups_to_log`: Number of groups to log in detail (default: 1)
- `compute_post_kl`: Compute post-training KL divergence (default: False)
- `behavior_if_log_dir_exists`: What to do if log dir exists (`"ask"`, `"overwrite"`, `"skip"`, default: `"ask"`)

#### Service Configuration

- `base_url`: Tinker service base URL (optional, uses default if not specified)

## Architecture

### Rollout Process

1. **GBoxAgent Rollout**: Uses GBoxAgent to run tasks with Tinker's OpenAI-compatible API
   - Input: Task description + screenshots (via `input_text` and `input_image`)
   - Output: Task completion result (success/failure)
   - Model: Current training checkpoint via `tinker://.../sampler_weights/XXXXX` path

2. **On-Policy RL**: The rollout model dynamically updates as training progresses
   - Checkpoint path is extracted from the `SamplingClient` in the policy
   - GBoxAgent uses Tinker's OpenAI-compatible API endpoint
   - Base URL: `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`
   - Model name: Checkpoint path (e.g., `tinker://.../sampler_weights/000080`)

3. **Message Conversion**: OpenAI Responses API format → Tinker Message format
   - `input_text` → `TextPart`
   - `input_image` → `ImagePart`
   - Handled by `vision_utils.convert_openai_responses_to_message()`

### Training Process

1. **Data Preparation**: Convert rollout results to `TrajectoryGroup`
   - Each trajectory contains observations (task + screenshots) and actions
   - Rewards based on task success (1.0 if successful, 0.0 otherwise)

2. **Vision Inputs**: Convert messages to `ModelInput` with `ImageChunk`
   - Uses `Qwen3VLRenderer` or similar vision renderer
   - Screenshots are converted to `ImageChunk` with proper token counts

3. **GRPO Training**: Group-based advantage centering
   - Advantages are centered within each group
   - Uses PPO loss with importance sampling

### Task Loading System

Tasks are loaded using the flexible `TaskSourceConfig` system:

- **Demo Tasks**: Pre-defined tasks from `demo_tasks.py` with categories and difficulties
- **Task IDs**: Load specific tasks by ID
- **Files**: Load from text files (one task per line)
- **Custom**: Pass task descriptions directly
- **Multiple Sources**: Combine multiple sources in a single training run

The task loader supports filtering by:
- Category: `system`, `navigation`, `settings`, `app`, `input`
- Difficulty: `easy`, `medium`, `hard`
- Limit: Limit number of tasks (with optional random sampling)

See `task_loader.py` and `TASK_CONFIG_QUICKSTART.md` for details.

## Vision Inputs Handling

The recipe handles vision inputs according to [Tinker's vision inputs guide](https://tinker-docs.thinkingmachines.ai/training-sampling#vision-inputs):

1. **Rollout**: GBoxAgent uses OpenAI Responses API format:
   ```python
   {
       "role": "user",
       "content": [
           {"type": "input_text", "text": "Turn 1/20. Analyze the screenshot..."},
           {"type": "input_image", "image_url": "data:image/png;base64,..."}
       ]
   }
   ```

2. **Training**: Converted to Tinker format:
   ```python
   ModelInput(chunks=[
       EncodedTextChunk(tokens=...),
       ImageChunk(data=image_bytes, format="jpeg", expected_tokens=...),
       EncodedTextChunk(tokens=...),
   ])
   ```

The conversion is handled automatically by the renderer (e.g., `Qwen3VLRenderer`).

## Dynamic Rollout Model (On-Policy RL)

The rollout model **dynamically updates** as training progresses:

- **Rollout**: Uses the current training checkpoint via Tinker's OpenAI-compatible API
  - Checkpoint path format: `tinker://.../sampler_weights/000080`
  - Automatically extracted from the `SamplingClient` in the policy
  - Updates after each training step (when `save_every > 0`)
- **Training**: Uses the same model being trained

This ensures **on-policy RL** - the rollout always uses the latest trained model, not a fixed model.

According to [Tinker's OpenAI-compatible API docs](https://tinker-docs.thinkingmachines.ai/compatible-apis/openai):
- Base URL: `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`
- Model name: Tinker checkpoint path (e.g., `tinker://.../sampler_weights/000080`)
- API key: Tinker API key

## Reward Function

Currently, the recipe uses a simple binary reward function:
- **1.0**: Task completed successfully
- **0.0**: Task failed or not completed

The reward is based on the `task_success` field from the GBoxAgent rollout result.

See `reward.py` for reward function implementations and extensibility.

## Examples

### Training with Demo Tasks

```python
from tinker_cookbook.recipes.cua_rl.train import CLIConfig, cli_main
import asyncio

config = CLIConfig(
    model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    tasks={"source_type": "demo_training", "category": "settings"},
    gbox_api_key="your-key",
    tinker_api_key="your-key",
    group_size=4,
    groups_per_batch=10,
    learning_rate=1e-5,
)

asyncio.run(cli_main(config))
```

### Training with Multiple Task Sources

```python
config = CLIConfig(
    model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    tasks=[
        {"source_type": "demo_training", "category": "settings", "limit": 20},
        {"source_type": "demo_training", "category": "navigation", "limit": 10}
    ],
    eval_tasks={"source_type": "demo_eval"},
    gbox_api_key="your-key",
    group_size=4,
    groups_per_batch=10,
)
```

### Training with Custom Tasks

```python
config = CLIConfig(
    model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
    tasks={
        "source_type": "custom",
        "custom_tasks": [
            "Open Settings and enable Wi-Fi",
            "Send a message to Alice",
            "Take a screenshot",
        ]
    },
    gbox_api_key="your-key",
    group_size=4,
)
```

### CLI Usage

```bash
# Basic training
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "demo_training"}' \
    --group_size 4 \
    --groups_per_batch 10

# With evaluation tasks
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "demo_training"}' \
    --eval_tasks '{"source_type": "demo_eval"}' \
    --eval_every 5 \
    --save_every 5

# Filtered tasks
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --tasks '{"source_type": "demo_training", "category": "settings", "difficulty": "easy", "limit": 20}' \
    --group_size 4
```

## Limitations

1. **Simplified Trajectories**: Currently, the recipe creates simplified trajectories. Full conversation history extraction from GBoxAgent is a future improvement.

2. **Vision Renderer**: Requires a vision-capable renderer (e.g., `Qwen3VLRenderer`). Text-only renderers won't work.

3. **GBoxAgent Dependency**: Requires `gbox-agent` package to be installed and accessible.

4. **Binary Rewards**: Currently uses simple binary rewards (1.0/0.0). More sophisticated reward shaping can be added via `reward.py`.

## Database Integration

The recipe automatically records all training data to a PostgreSQL database for monitoring and analysis.

### Database Setup

The database is automatically initialized when training starts. You can configure it via environment variables:

```bash
export DATABASE_URL=postgresql://training_user:training_password@localhost:5432/training_db
```

Or use individual variables:
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=training_db
export POSTGRES_USER=training_user
export POSTGRES_PASSWORD=training_password
```

### Database Migration

The project uses Alembic for database migrations. The database schema is automatically migrated on initialization.

**Rebuild database** (if needed):
```bash
uv run python -m tinker_cookbook.recipes.cua_rl.rebuild_database
```

**Create new migration** (when models change):
```bash
cd tinker_cookbook/recipes/cua_rl
uv run alembic revision --autogenerate -m "describe changes"
uv run alembic upgrade head
```

See `DATABASE_USAGE.md` for more details.

### Training Monitor

A web-based training monitor is available to visualize training progress in real-time. See `training-monitor/` directory for setup instructions.

The monitor displays:
- Training sessions and their status
- Steps with group and rollout information
- Evaluations and baseline assessments
- Real-time updates (auto-refreshes every 10 seconds)

## References

- [Tinker Training and Sampling Guide](https://tinker-docs.thinkingmachines.ai/training-sampling)
- [Tinker Vision Inputs](https://tinker-docs.thinkingmachines.ai/training-sampling#vision-inputs)
- [Tinker Compatible APIs](https://tinker-docs.thinkingmachines.ai/compatible-apis/openai)
- [Tinker RL Guide](https://tinker-docs.thinkingmachines.ai/rl)
- [GBoxAgent Documentation](https://github.com/your-org/gbox-agent)
- Task Configuration: See `TASK_CONFIG_QUICKSTART.md` and `task_config_examples.md`
- Database Usage: See `DATABASE_USAGE.md` and `DATABASE_IMPLEMENTATION.md`

