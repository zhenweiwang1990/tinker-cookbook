# Tasks Directory

This directory contains task definitions for CUA RL training.

## Structure

- `airbnb/` - Airbnb app tasks (32 tasks)
- `demo/` - Demo Android system tasks (32 tasks)
- `instagram/` - Instagram app tasks
- `adb.py` - ADB client (supports both local ADB and GBox command mode)
- `task_adapter.py` - Task adapter for splitting tasks into train/eval sets
- `config.py` - Configuration utilities

## ADB Client with GBox Support

The `AdbClient` class now supports both local ADB commands and GBox command execution.

### Local ADB Mode (default)

```python
from tinker_cookbook.recipes.cua_rl.tasks.adb import AdbClient

# Uses local ADB commands
adb = AdbClient()
adb.install("app.apk")
adb.launch("com.example.app")
```

### GBox Command Mode

```python
from tinker_cookbook.recipes.cua_rl.tasks.adb import AdbClient
from tinker_cookbook.recipes.cua_rl.cua_gbox_client import CuaGBoxClient

# Create GBox client and box
gbox_client = CuaGBoxClient(api_key=os.environ["GBOX_API_KEY"])
await gbox_client.create_box(box_type="android")

# Use AdbClient with GBox
adb = AdbClient(gbox_box=gbox_client._box)
# or
adb = AdbClient(gbox_client=gbox_client)

# Commands are executed via GBox SDK
adb.install("app.apk")
adb.launch("com.example.app")
result = adb.run_sqlite_query("com.example.app", "databases/app.db", "SELECT * FROM users")
```

## Task Adapter

The `TaskAdapter` class automatically discovers all tasks and splits them into training and evaluation sets.

### Basic Usage

```python
from tinker_cookbook.recipes.cua_rl.tasks.task_adapter import TaskAdapter

# Create adapter with default settings (80% train, 20% eval, seed=42)
adapter = TaskAdapter()

# Get task descriptions
train_descriptions = adapter.get_train_descriptions()
eval_descriptions = adapter.get_eval_descriptions()

print(f"Training tasks: {len(train_descriptions)}")
print(f"Evaluation tasks: {len(eval_descriptions)}")
```

### Custom Split

```python
# Custom split ratio and seed
adapter = TaskAdapter(train_ratio=0.75, seed=123)

train_descriptions = adapter.get_train_descriptions()
eval_descriptions = adapter.get_eval_descriptions()
```

### Get Task Instances

```python
# Get full task metadata
train_tasks = adapter.get_train_tasks()
eval_tasks = adapter.get_eval_tasks()

for task_info in train_tasks:
    task = task_info["task_instance"]
    print(f"Task: {task.name}")
    print(f"Description: {task.description}")
    validator = task.get_validator()
    pre_hook = task.get_pre_hook()
```

### Convenience Function

```python
from tinker_cookbook.recipes.cua_rl.tasks.task_adapter import get_tasks_train_eval

train_descriptions, eval_descriptions = get_tasks_train_eval(
    train_ratio=0.8,
    seed=42
)
```

## Task Datasets

### Airbnb Tasks
- **Total**: 32 tasks (Task01-Task32)
- **Domain**: Airbnb app interactions
- **Validation**: SQL database queries
- **Features**: Some tasks include pre-hooks for database seeding
- **Split**: Train/eval split determined by `TaskAdapter` configuration

### Demo Tasks  
- **Total**: 32 tasks (Task01-Task32)
- **Domain**: Android system settings and file operations
- **Validation**: ADB shell commands and system state checks
- **Split**: Train/eval split determined by `TaskAdapter` configuration
- **Categories**:
  - Settings adjustments (brightness, timeout, airplane mode, WiFi, battery saver, DND)
  - File operations (create folders in Downloads/Documents)
  - App management (uninstall apps, check storage)
  - System tasks (clear notifications, check system info)

## Task Structure

Each task directory should contain:
- `task.py` - Task definition with `create_task()` function
- `validator.py` - Validation logic
- `pre_hook.py` (optional) - Pre-execution setup

Example task structure:
```
tasks/
  airbnb/
    01_i_plan_to_go_to_united/
      task.py          # Defines Task01 with create_task()
      validator.py     # Defines Task01Validator
      pre_hook.py     # Optional pre-execution setup
  demo/
    01_open_settings/
      task.py          # Defines TrainTask01 with create_task()
      validator.py     # Defines TrainTask01Validator
    config.py          # Package name configuration
```

