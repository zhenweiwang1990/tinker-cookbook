# CUA RL Training Database Usage

## 概述

CUA RL训练系统使用 PostgreSQL 数据库记录整个训练过程的详细信息。数据库会自动在训练启动时初始化，并在训练过程中记录所有相关信息。

## 数据库配置

### 方式 1: 使用 DATABASE_URL 环境变量

```bash
export DATABASE_URL=postgresql://training_user:training_password@localhost:5432/training_db
python -m tinker_cookbook.recipes.cua_rl.train --model_name "..." --log_path "/path/to/logs"
```

### 方式 2: 使用独立的 PostgreSQL 环境变量

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=training_db
export POSTGRES_USER=training_user
export POSTGRES_PASSWORD=training_password
python -m tinker_cookbook.recipes.cua_rl.train --model_name "..." --log_path "/path/to/logs"
```

如果未设置这些变量，系统会使用默认值（匹配 training-monitor 的 docker-compose.yml 配置）。

### 方式 3: 与 training-monitor 共享数据库

如果使用 training-monitor 的 Docker Compose，可以直接使用相同的数据库：

```bash
export DATABASE_URL=postgresql://training_user:training_password@localhost:5432/training_db
python -m tinker_cookbook.recipes.cua_rl.train --model_name "..." --log_path "/path/to/logs"
```

## 自动记录的内容

### 1. 训练会话 (Training)
- 训练配置（模型、超参数等）
- 训练状态和进度
- 开始/结束时间

### 2. 任务和验证器 (Task & Validator)
- 所有加载的任务自动保存到数据库
- 任务的验证器信息也会保存
- 后续可以直接从数据库加载任务

### 3. 训练步骤 (Step)
- 每个训练步骤的详细信息
- 滚动和训练的进度
- 指标和损失值

### 4. 滚动 (Rollout)
- 每次滚动的完整信息
- 任务执行结果
- 验证结果
- 奖励和指标

### 5. 回合、动作、观察 (Turn, Action, Obs)
- 每个回合的详细信息
- 所有动作和观察

### 6. 评估 (Eval & Baseline)
- 评估结果
- 基线评估结果

## 使用方式

### 基本使用

数据库集成是自动的。只需正常启动训练：

```bash
export DATABASE_URL=postgresql://training_user:training_password@localhost:5432/training_db
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --log_path "/path/to/logs"
```

数据库会自动创建和更新。

### 查询数据库

使用 Python 查询数据库：

```python
from tinker_cookbook.recipes.cua_rl.database import init_database, get_session
from tinker_cookbook.recipes.cua_rl.database_dao import (
    get_training_by_run_name,
    list_rollouts_by_step,
    get_status_history,
)

# 初始化数据库连接
init_database("postgresql://training_user:training_password@localhost:5432/training_db")

# 查询训练记录
with get_session() as session:
    training = get_training_by_run_name(session, "cua_rl-...")
    print(f"Training ID: {training.id}")
    print(f"Status: {training.status}")
    print(f"Progress: {training.progress_percent}%")
    
    # 查询某个步骤的所有滚动
    steps = training.steps
    if steps:
        step = steps[0]
        rollouts = list_rollouts_by_step(session, step.id)
        print(f"Step {step.step} has {len(rollouts)} rollouts")
    
    # 查询状态历史
    history = get_status_history(session, "training", training.id)
    for h in history:
        print(f"{h.changed_at}: {h.old_status} -> {h.new_status}")
```

## 数据库表结构

详细的表结构请参考 `database_schema.md`。

主要表：
- `training` - 训练会话
- `task` - 任务
- `validator` - 验证器
- `step` - 训练步骤
- `group` - 滚动组（每个 step/eval/baseline 包含多个 group）
- `rollout` - 滚动（属于某个 group）
- `turn` - 回合
- `action` - 动作
- `obs` - 观察
- `validation` - 验证结果
- `environment` - 环境信息
- `eval` - 评估
- `baseline` - 基线评估
- `status_history` - 状态历史

## 从数据库加载训练进度

训练启动时会自动检查数据库，如果找到相同 `run_name` 的训练记录，会：
1. 加载训练状态
2. 更新状态为 "initializing"
3. 继续训练

## 注意事项

1. **数据库会话管理**: 数据库会话在训练过程中会自动管理，无需手动处理
2. **性能**: 数据库操作是异步的，不会显著影响训练性能
3. **错误处理**: 如果数据库操作失败，会记录警告但不会中断训练
4. **并发**: PostgreSQL 完全支持并发访问，适合生产环境

## 数据库迁移

项目使用 Alembic 进行数据库迁移管理。

### 重建数据库（从零开始）

如果需要重建数据库以应用最新的模型结构：

```bash
uv run python -m tinker_cookbook.recipes.cua_rl.rebuild_database
```

这会：
1. 删除所有现有表
2. 根据当前模型重新创建所有表
3. 创建并标记初始 Alembic 迁移

### 创建新的迁移

当模型结构改变时，创建新的迁移：

```bash
cd tinker_cookbook/recipes/cua_rl
uv run alembic revision --autogenerate -m "描述你的更改"
uv run alembic upgrade head
```

### 应用迁移

数据库初始化时会自动运行 Alembic 迁移。也可以手动运行：

```bash
cd tinker_cookbook/recipes/cua_rl
uv run alembic upgrade head
```

### 查看迁移历史

```bash
cd tinker_cookbook/recipes/cua_rl
uv run alembic history
uv run alembic current
```

## 依赖

需要安装 SQLAlchemy、PostgreSQL 驱动和 Alembic：
```bash
pip install sqlalchemy>=2.0.0 psycopg2-binary alembic>=1.13.0
```

或者安装完整的CUA依赖：
```bash
pip install -e ".[cua]"
```

