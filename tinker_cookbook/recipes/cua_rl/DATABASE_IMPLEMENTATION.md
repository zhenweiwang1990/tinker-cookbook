# CUA RL 数据库实现总结

## 实现完成情况

✅ **所有核心功能已完成实现**

### 1. 数据库基础设施
- ✅ SQLAlchemy ORM 模型定义 (`database_models.py`)
- ✅ 数据库初始化和会话管理 (`database.py`)
- ✅ 完整的 DAO 层 (`database_dao.py`)
- ✅ 全局数据库上下文 (`database_context.py`)
- ✅ Alembic 数据库迁移 (`alembic/`)
- ✅ 数据库重建脚本 (`rebuild_database.py`)

### 2. 数据记录模块
- ✅ 任务和验证器记录 (`database_task_loader.py`)
- ✅ Rollout 记录 (`database_rollout.py`)
- ✅ Step 记录 (`database_step.py`)
- ✅ Eval 和 Baseline 记录 (`database_eval.py`)
- ✅ 训练循环 Hook (`database_training_hooks.py`)

### 3. 集成点
- ✅ 训练启动时初始化数据库 (`train.py`)
- ✅ 任务加载时自动保存 (`task_loader.py`, `cua_env.py`)
- ✅ Rollout 过程中记录 (`rollout.py`)
- ✅ 训练步骤记录 (通过 hook)
- ✅ 评估和基线记录 (通过 wrapper)

## 数据库位置

数据库由 PostgreSQL 服务器管理，位置取决于 PostgreSQL 配置。

## 自动记录的数据

### 训练会话 (Training)
- 训练配置和超参数
- 训练状态和进度
- 开始/结束时间

### 任务和验证器 (Task & Validator)
- 所有加载的任务自动保存
- 验证器信息保存
- 支持从数据库加载任务

### 训练步骤 (Step)
- 每个步骤的详细信息
- 滚动和训练的进度
- 指标和损失值

### 滚动组 (Group)
- 每个 step/eval/baseline 包含多个 group
- Group 状态和进度
- Group 级别的统计信息（rollout 数量、成功率等）

### 滚动 (Rollout)
- 完整的滚动信息
- 任务执行结果
- 验证结果
- 奖励和指标
- 支持 step/eval/baseline 三种来源
- 属于某个 group（通过 group_id 关联）

### 回合、动作、观察 (Turn, Action, Obs)
- 每个回合的详细信息
- 所有动作和观察

### 评估 (Eval & Baseline)
- 评估结果
- 基线评估结果

## 使用方式

数据库集成是完全自动的，无需额外配置。只需正常启动训练：

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --log_path "/path/to/logs"
```

数据库会在 PostgreSQL 服务器中自动创建和更新。

## 查询示例

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

## 文件结构

```
tinker_cookbook/recipes/cua_rl/
├── database_models.py          # SQLAlchemy 模型定义
├── database.py                 # 数据库初始化和会话管理
├── database_dao.py             # DAO 层 (所有表的 CRUD)
├── database_context.py         # 全局数据库上下文
├── database_task_loader.py     # 任务和验证器数据库操作
├── database_rollout.py          # Rollout 数据库操作
├── database_step.py             # Step 数据库操作
├── database_eval.py             # Eval 和 Baseline 数据库操作
├── database_training_hooks.py   # 训练循环 Hook
├── database_schema.md           # 数据库表结构文档
├── DATABASE_USAGE.md           # 使用说明
├── DATABASE_IMPLEMENTATION.md  # 实现总结 (本文件)
├── rebuild_database.py         # 数据库重建脚本
├── alembic.ini                 # Alembic 配置文件
└── alembic/                    # Alembic 迁移目录
    ├── env.py                  # Alembic 环境配置
    └── versions/               # 迁移脚本目录
```

## 技术栈

- **ORM**: SQLAlchemy 2.0+
- **数据库**: PostgreSQL
- **集成方式**: 
  - 自动集成（任务加载、训练启动）
  - Hook/Wrapper（rollout、step、eval）

## 数据库配置

通过环境变量配置：

```bash
# 方式 1: 使用 DATABASE_URL
export DATABASE_URL=postgresql://user:password@host:port/dbname

# 方式 2: 使用独立的 POSTGRES_* 变量
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=training_db
export POSTGRES_USER=training_user
export POSTGRES_PASSWORD=training_password
```

## 注意事项

1. **数据库会话管理**: 使用提供的 context manager 和全局上下文
2. **错误处理**: 数据库操作失败不会中断训练，只会记录警告
3. **性能**: 数据库操作是异步的，对训练性能影响最小
4. **并发**: PostgreSQL 完全支持并发访问，适合生产环境

## 数据库迁移

项目使用 Alembic 进行数据库迁移管理。数据库初始化时会自动运行迁移。

### 重建数据库

如果需要重建数据库（删除所有表并重新创建）：

```bash
uv run python -m tinker_cookbook.recipes.cua_rl.rebuild_database
```

### 创建新迁移

当模型结构改变时：

```bash
cd tinker_cookbook/recipes/cua_rl
uv run alembic revision --autogenerate -m "描述更改"
uv run alembic upgrade head
```

## 依赖

需要安装 SQLAlchemy、PostgreSQL 驱动和 Alembic：
```bash
pip install sqlalchemy>=2.0.0 psycopg2-binary alembic>=1.13.0
```

或安装完整的 CUA 依赖：
```bash
pip install -e ".[cua]"
```

## 下一步

所有核心功能已完成。可以：
1. 测试数据库功能
2. 根据需要添加更多查询功能
3. 创建数据分析脚本

