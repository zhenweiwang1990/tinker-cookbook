# CUA RL Task Configuration Guide

This guide shows how to configure tasks for CUA RL training using the flexible task loading system.

## Quick Start

### 1. Use All Demo Training Tasks

```python
# In train.py or via CLI
tasks = {"source_type": "demo_training"}
```

### 2. Use Specific Task IDs

```python
tasks = {
    "source_type": "ids",
    "task_ids": ["train_01_open_settings", "train_02_enable_wifi", "train_03_max_brightness"]
}
```

### 3. Filter by Category

```python
# Only settings tasks
tasks = {
    "source_type": "demo_training",
    "category": "settings"
}
```

### 4. Filter by Difficulty

```python
# Only easy tasks
tasks = {
    "source_type": "demo_training",
    "difficulty": "easy"
}
```

### 5. Combine Filters

```python
# Easy settings tasks, limit to 10
tasks = {
    "source_type": "demo_training",
    "category": "settings",
    "difficulty": "easy",
    "limit": 10,
    "seed": 42  # For reproducible sampling
}
```

### 6. Multiple Sources

```python
# Combine training and eval tasks
tasks = [
    {"source_type": "demo_training", "category": "settings", "limit": 20},
    {"source_type": "demo_training", "category": "navigation", "limit": 10}
]
```

### 6b. Configure Evaluation Dataset

Evaluation dataset is configured separately using `eval_tasks`:

```python
# In train.py
tasks = {"source_type": "demo_training"}  # Training dataset
eval_tasks = {"source_type": "demo_eval"}  # Evaluation dataset
```

The evaluation dataset will be run every `eval_every` steps to monitor model performance.

### 7. Load from File

```python
# Load tasks from a file (one task description per line)
tasks = {
    "source_type": "file",
    "file_path": "path/to/tasks.txt"
}
```

### 8. Custom Tasks

```python
# Custom task descriptions via TaskSourceConfig
tasks = {
    "source_type": "custom",
    "custom_tasks": [
        "Open the Settings app",
        "Enable WiFi",
        "Set brightness to maximum"
    ]
}
```

## Configuration Reference

### TaskSourceConfig

```python
@dataclass
class TaskSourceConfig:
    source_type: str  # Required: "demo_training", "demo_eval", "demo_all", "ids", "file", "custom"
    task_ids: Optional[List[str]] = None  # For "ids" source_type
    category: Optional[str] = None  # Filter: "system", "navigation", "settings", "app", "input"
    difficulty: Optional[str] = None  # Filter: "easy", "medium", "hard"
    file_path: Optional[str] = None  # For "file" source_type
    custom_tasks: Optional[List[str]] = None  # For "custom" source_type
    limit: Optional[int] = None  # Limit number of tasks
    seed: Optional[int] = None  # Random seed for sampling when limit is used
```

## Examples

### Example 1: Training on Easy Tasks Only

```python
# train.py
tasks = {
    "source_type": "demo_training",
    "difficulty": "easy",
    "limit": 30
}
```

### Example 2: Mix of Categories

```python
tasks = [
    {"source_type": "demo_training", "category": "settings", "limit": 15},
    {"source_type": "demo_training", "category": "navigation", "limit": 10},
    {"source_type": "demo_training", "category": "app", "limit": 10}
]
```

### Example 3: Specific Task IDs

```python
tasks = {
    "source_type": "ids",
    "task_ids": [
        "train_01_open_settings",
        "train_02_enable_wifi",
        "train_05_airplane_mode",
        "train_07_go_home"
    ]
}
```

### Example 4: Progressive Training

```python
# Start with easy tasks, then add harder ones
tasks = [
    {"source_type": "demo_training", "difficulty": "easy"},
    {"source_type": "demo_training", "difficulty": "medium", "limit": 20}
]
```

## CLI Usage

When using via command line with `chz`, you can pass tasks as JSON:

```bash
python train.py \
    --tasks '{"source_type": "demo_training", "category": "settings"}' \
    --group_size 4 \
    --groups_per_batch 10
```

For multiple sources:

```bash
python train.py \
    --tasks '[{"source_type": "demo_training"}, {"source_type": "demo_eval"}]' \
    --group_size 4
```

## Extending with New Datasets

To add a new dataset source:

1. Add your tasks to a new module (e.g., `custom_tasks.py`)
2. Create a loader function that returns `List[CUATask]`
3. Update `task_loader.py` to support your new source type:

```python
# In task_loader.py
elif config.source_type == "custom_dataset":
    from tinker_cookbook.recipes.cua_rl.custom_tasks import get_custom_tasks
    tasks = get_custom_tasks()
    # Apply filters, etc.
```

Then use it:

```python
tasks = {"source_type": "custom_dataset"}
```

