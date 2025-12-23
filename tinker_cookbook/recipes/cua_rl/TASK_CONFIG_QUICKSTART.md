# CUA RL Task Configuration - Quick Start

## 最简单的使用方式

### 使用所有 demo 训练任务

```python
# 在 train.py 中
tasks = {"source_type": "demo_training"}
```

### 使用特定类别的任务

```python
# 只使用 settings 类别的任务
tasks = {
    "source_type": "demo_training",
    "category": "settings"
}
```

### 使用特定难度的任务

```python
# 只使用简单任务
tasks = {
    "source_type": "demo_training",
    "difficulty": "easy"
}
```

### 组合筛选条件

```python
# 简单难度的 settings 任务，限制 10 个
tasks = {
    "source_type": "demo_training",
    "category": "settings",
    "difficulty": "easy",
    "limit": 10,
    "seed": 42  # 用于可重复的采样
}
```

### 使用多个数据源

```python
# 组合训练和评估任务
tasks = [
    {"source_type": "demo_training", "category": "settings", "limit": 20},
    {"source_type": "demo_eval", "limit": 10}
]
```

### 使用特定任务 ID

```python
tasks = {
    "source_type": "ids",
    "task_ids": ["train_01_open_settings", "train_02_enable_wifi"]
}
```

## 配置参数说明

- `source_type`: 数据源类型
  - `"demo_training"`: 使用 `demo_tasks.py` 中的训练任务
  - `"demo_eval"`: 使用 `demo_tasks.py` 中的评估任务
  - `"demo_all"`: 使用所有 demo 任务
  - `"ids"`: 通过任务 ID 指定（需要提供 `task_ids`）
  - `"file"`: 从文件加载（需要提供 `file_path`）
  - `"custom"`: 自定义任务列表（需要提供 `custom_tasks`）

- `category`: 任务类别筛选（可选）
  - `"system"`, `"navigation"`, `"settings"`, `"app"`, `"input"`

- `difficulty`: 难度筛选（可选）
  - `"easy"`, `"medium"`, `"hard"`

- `limit`: 限制任务数量（可选）
- `seed`: 随机种子，用于可重复采样（可选）

## 配置评估数据集

评估数据集通过 `eval_tasks` 配置，格式与 `tasks` 相同：

```python
# 在 train.py 中
tasks = {"source_type": "demo_training"}  # 训练数据集
eval_tasks = {"source_type": "demo_eval"}  # 评估数据集
```

评估数据集会在每个 `eval_every` 步骤运行一次，用于监控模型性能。

## 从文件加载任务

如果需要从文件加载任务，使用 `file` source_type：

```python
tasks = {
    "source_type": "file",
    "file_path": "path/to/tasks.txt"
}
```

## 扩展新数据集

要添加新的数据集源，只需：

1. 在 `task_loader.py` 中添加新的 `source_type` 处理逻辑
2. 实现对应的加载函数

详见 `task_loader.py` 和 `task_config_examples.md`。

