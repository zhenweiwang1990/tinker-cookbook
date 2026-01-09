# Task Datasets Comparison

This document compares the two available task datasets for CUA RL training.

## Overview

| Dataset | Tasks | Domain | Validation Method |
|---------|-------|--------|-------------------|
| **Airbnb** | 32 | Airbnb app | SQL queries |
| **Demo** | 32 | Android system | ADB shell commands |

## Dataset Details

### Airbnb Dataset

**Location**: `tasks/airbnb/`

**Structure**:
- 32 tasks (Task01-Task32)
- Each task in separate directory: `XX_task_description/`
- Files per task: `task.py`, `validator.py`, optional `pre_hook.py`
- Configuration: `config.py` (package name: `com.airbnb.android`)

**Task Types**:
- Search and save listings with specific criteria
- Booking management and modifications
- User profile and settings management
- Message and communication tasks
- Advanced filtering and calculations

**Validation**:
- SQL queries against Airbnb app database
- Database path: `/data/data/com.airbnb.android/databases/airbnbSQLiteSQLite.db`
- Some tasks require pre-hooks to seed database

**Example Task**:
```
01_i_plan_to_go_to_united/
  task.py         # Task01 class
  validator.py    # SQL-based validation
```

---

### Demo Dataset

**Location**: `tasks/demo/`

**Structure**:
- 32 tasks (Task01-Task32)
- Each task in separate directory: `XX_task_description/`
- Files per task: `task.py`, `validator.py`
- Configuration: `config.py` (package name: `com.android.settings`)

**Task Types**:
1. **Settings Adjustments** (15 tasks)
   - Screen brightness (min/max)
   - Screen timeout (30s to 30min)
   - Airplane mode
   - WiFi control
   - Battery saver
   - Do Not Disturb
   - Auto time settings

2. **File Operations** (10 tasks)
   - Create folders in Downloads
   - Create folders in Documents
   - Download files from web

3. **App Management** (5 tasks)
   - Uninstall apps (Instagram, Facebook)
   - Check app storage size (Chrome, Instagram, GBOX Keyboard)

4. **System Tasks** (2 tasks)
   - Clear notifications
   - Check system information

**Validation**:
- ADB shell commands (`settings get`, `dumpsys`, `test -e`, etc.)
- System state checks (brightness, timeout, mode toggles)
- File existence verification
- App installation status

**Example Task**:
```
01_open_settings/
  task.py         # Task01 class
  validator.py    # ADB shell-based validation
```

---

## Task Naming Convention

Both datasets follow the same structure (matching Airbnb's pattern):

```python
# Directory name: {number}_{description}
01_open_settings/

# Class name: Task{number}
class Task01(Task):
    name: str = "01_open_settings"
    description: str = "..."
    
# Validator class: Task{number}Validator
class Task01Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        ...

# Factory function
def create_task() -> Task:
    return Task01()
```

---

## Train/Eval Split

**Both datasets use dynamic splitting** via `TaskAdapter`:

```python
from tinker_cookbook.recipes.cua_rl.tasks.task_adapter import TaskAdapter

# Default: 80% train, 20% eval, seed=42
adapter = TaskAdapter()

# Or custom split
adapter = TaskAdapter(train_ratio=0.75, seed=123)

train_descriptions = adapter.get_train_descriptions()
eval_descriptions = adapter.get_eval_descriptions()
```

The split is **not hardcoded** in the task code - it's determined at training time by configuration.

---

## Usage Recommendation

### Use Airbnb Dataset When:
- Testing complex app interactions
- Working with database-backed validation
- Need realistic, production-like tasks
- Evaluating multi-step workflows in a real app

### Use Demo Dataset When:
- Getting started with CUA RL
- Testing system-level Android interactions
- Need simpler, more predictable tasks
- Debugging validation logic
- Working without external app dependencies

---

## Extending the Datasets

To add new tasks to either dataset:

1. Create a new directory: `{next_number}_{description}/`
2. Add `task.py` with `Task{N}` class and `create_task()` function
3. Add `validator.py` with `Task{N}Validator` class
4. (Optional) Add `pre_hook.py` for setup steps

Example:
```bash
cd tasks/demo/
mkdir 33_new_task/

# Create task.py
cat > 33_new_task/task.py << 'EOF'
from __future__ import annotations
from dataclasses import dataclass
from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task33Validator

@dataclass
class Task33(Task):
    name: str = "33_new_task"
    description: str = """Your task description here."""
    
    def run(self, adb_client: AdbClient) -> bool:
        return True
    
    def get_validator(self):
        return Task33Validator()
    
    def get_pre_hook(self):
        return None

def create_task() -> Task:
    return Task33()
EOF

# Create validator.py
cat > 33_new_task/validator.py << 'EOF'
from __future__ import annotations
from ... import config
from ....adb import AdbClient

class Task33Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Your validation logic here
        return True
EOF
```

---

## Summary

Both datasets provide 32 well-structured tasks with:
- ✅ Unified naming convention (TaskXX)
- ✅ Separate directories per task
- ✅ Clean validator pattern
- ✅ Dynamic train/eval splitting
- ✅ Extensible structure

Choose based on your use case: **Airbnb** for complex app testing, **Demo** for system-level basics.

