# 进度追踪系统 (Progress Tracking System)

## 概述

本进度追踪系统为 CUA RL 训练提供统一的状态管理和进度计算，所有层级都基于 **turns** 来进行进度评估和时间预测。

## 层级结构

```
Training (训练)
  ├─ Baseline (基准评估, 1个)
  │   └─ Groups (任务数量个)
  │       └─ Rollouts (每个Group包含group_size个)
  │           └─ Turns (每个Rollout最多max_turns个)
  ├─ Steps (N个训练步骤)
  │   └─ Groups (每个Step包含任务数量个)
  │       └─ Rollouts (每个Group包含group_size个)
  │           └─ Turns (每个Rollout最多max_turns个)
  └─ Evals (N/eval_every 个评估)
      └─ Groups (每个Eval包含任务数量个)
          └─ Rollouts (每个Group包含group_size个)
              └─ Turns (每个Rollout最多max_turns个)
```

## 状态类型

每个层级支持以下状态：

- `pending`: 待执行
- `running`: 执行中
- `completed`: 已完成
- `failed`: 失败

## 进度计算

### 基于 Turns 的进度

所有进度都基于 turns 来计算，因为 turn 是最小的可测量执行单元：

```python
# Rollout 进度
progress = (completed_turns / max_turns) * 100%

# Group 进度
progress = (sum(rollout_completed_turns) / sum(rollout_total_turns)) * 100%

# Step 进度 (分两阶段)
rollout_progress = (completed_turns / total_turns) * 80%  # Rollout阶段占80%
training_progress = 20% if training_completed else 0%     # Training阶段占20%
progress = rollout_progress + training_progress

# Eval 进度
progress = (completed_turns / total_turns) * 100%

# Baseline 进度
progress = (completed_turns / total_turns) * 100%

# Training 进度 (加权平均)
baseline_weight = 10%
steps_weight = 80%
evals_weight = 10%
progress = baseline_progress * 0.1 + avg_step_progress * 0.8 + avg_eval_progress * 0.1
```

### 时间估算

系统会基于历史 turn 的平均耗时来预测总时间和剩余时间：

```python
avg_turn_time = total_time / completed_turns  # 平均每个turn的耗时（秒）
estimated_total_time = total_turns * avg_turn_time  # 预计总耗时
estimated_remaining_time = remaining_turns * avg_turn_time  # 预计剩余时间
```

时间估算的优先级（从最精确到最粗略）：
1. Rollout 层级：基于该 rollout 已完成 turns 的实际耗时
2. Group 层级：基于该 group 内所有 rollouts 的平均 turn 耗时
3. Step/Eval/Baseline 层级：基于该实体内所有 groups 的平均 turn 耗时
4. Training 层级：基于所有已完成 baseline/steps/evals 的平均 turn 耗时
5. 默认值：30秒/turn (如果没有历史数据)

## 数据库字段

所有实体（Training, Baseline, Eval, Step, Group）都包含以下进度追踪字段：

```python
progress_percent: Float  # 进度百分比 (0-100)
status: String  # 状态 (pending/running/completed/failed)
avg_turn_time: Float  # 平均每个turn的耗时（秒）
estimated_total_time: Float  # 预计总耗时（秒）
estimated_remaining_time: Float  # 预计剩余时间（秒）
current_turn: Integer  # 当前执行到的turn（仅Rollout）
num_turns: Integer  # 实际完成的turns数量（仅Rollout）
max_turns: Integer  # 最大turns数量（仅Rollout）
```

## 使用示例

### 初始化进度追踪器

```python
from tinker_cookbook.recipes.cua_rl.database.progress_tracker import ProgressTracker
from tinker_cookbook.recipes.cua_rl.database.database import get_session

with get_session() as session:
    tracker = ProgressTracker(session)
```

### 更新 Rollout 进度

```python
# 在每个 turn 开始或结束时更新
stats = tracker.update_rollout_progress(
    rollout_id=rollout_db_id,
    current_turn=turn_num,
    max_turns=20,
    status="running",
    turn_time=5.2,  # 本次turn耗时（可选）
)

print(f"Progress: {stats.progress_percent:.1f}%")
print(f"ETA: {tracker.format_time_estimate(stats.estimated_remaining_time)}")
```

### 更新 Group 进度

```python
# Group 进度会自动从其包含的 rollouts 聚合
stats = tracker.update_group_progress(group_id=group_db_id)
```

### 更新 Step 进度

```python
# Step 进度会自动从其包含的 groups 聚合
stats = tracker.update_step_progress(step_id=step_db_id)
```

### 更新 Training 进度

```python
# Training 进度会自动从 baseline/steps/evals 聚合
stats = tracker.update_training_progress(training_id=training_id)
```

## 自动级联更新

当更新子实体的进度时，系统会自动向上级联更新父实体的进度：

```
更新 Rollout → 自动更新 Group → 自动更新 Step/Eval/Baseline → 自动更新 Training
```

例如：
```python
# 只需要更新 rollout 进度
tracker.update_rollout_progress(rollout_id, current_turn, max_turns)

# 系统会自动：
# 1. 更新 Rollout 的进度和时间估算
# 2. 聚合所有 Rollouts，更新 Group 的进度
# 3. 聚合所有 Groups，更新 Step/Eval/Baseline 的进度
# 4. 聚合所有 Steps/Evals/Baseline，更新 Training 的进度
```

## RolloutRecorder 集成

`RolloutRecorder` 已经集成了进度追踪器，在每个 turn 开始时自动更新进度：

```python
# 在 start_turn() 时自动更新进度
def start_turn(self, turn_num: int):
    # ... 创建 turn 记录 ...
    
    # 自动更新进度（包括级联更新）
    progress_stats = self.progress_tracker.update_rollout_progress(
        rollout_id=self.rollout_db_id,
        current_turn=turn_num,
        max_turns=self.max_turns,
        status="running",
    )
    
    logger.debug(
        f"Turn {turn_num}, progress={progress_stats.progress_percent:.1f}%, "
        f"ETA={self.progress_tracker.format_time_estimate(progress_stats.estimated_remaining_time)}"
    )
```

## 数据库迁移

添加新字段需要运行数据库迁移：

```bash
cd tinker_cookbook/recipes/cua_rl
alembic upgrade head
```

迁移脚本位于：`alembic/versions/add_progress_tracking_fields.py`

## 性能考虑

1. **缓存机制**: `ProgressTracker` 会缓存已计算的平均 turn 时间，避免重复查询
2. **批量更新**: 在并发 rollout 场景下，进度更新会在每个 rollout 独立进行
3. **事务管理**: 所有进度更新都在数据库事务中完成，确保一致性

## 时间格式化

系统提供了人性化的时间格式化函数：

```python
tracker.format_time_estimate(seconds)
# 示例输出:
# 45s (少于1分钟)
# 3.5m (3.5分钟)
# 2.1h (2.1小时)
# 1.5d (1.5天)
```

## 调试和监控

可以通过日志查看进度更新：

```python
logger.debug(
    f"[Progress] Rollout {rollout_id}: "
    f"{stats.completed_turns}/{stats.total_turns} turns, "
    f"progress={stats.progress_percent:.1f}%, "
    f"avg_turn_time={stats.avg_turn_time:.1f}s, "
    f"ETA={tracker.format_time_estimate(stats.estimated_remaining_time)}"
)
```

## 示例场景

### 场景 1: 训练启动时的 Baseline 评估

```
Training (0%)
  └─ Baseline (0%)
      └─ Groups (0% each, 10 groups for 10 tasks)
          └─ Rollouts (0% each, 4 rollouts per group for group_size=4)
              └─ Turns (0/20 per rollout)

随着 rollouts 执行：
- 每个 turn 完成后，Rollout 进度从 0% → 5% → 10% → ... → 100%
- Group 进度聚合所有 rollouts
- Baseline 进度聚合所有 groups
- Training 进度: baseline_progress * 0.1 (因为 baseline 占10%)
```

### 场景 2: 训练步骤执行

```
Training (10% from baseline)
  ├─ Step 0 (0%)
  │   └─ Groups (0% each, 10 groups for 10 tasks)
  │       └─ Rollouts (0% each, 4 rollouts per group)
  │           └─ Turns (0/20 per rollout)

执行过程：
- Rollout phase: Groups 完成 rollouts → Step 进度 0% → 80%
- Training phase: 模型训练 → Step 进度 80% → 100%
- Training 总进度: 10% (baseline) + avg_steps * 0.8
```

### 场景 3: 定期评估

```
Training (50% from baseline + steps)
  └─ Eval at Step 10 (0%)
      └─ Groups (0% each, 10 groups for 10 tasks)
          └─ Rollouts (0% each, 1 rollout per group for eval)
              └─ Turns (0/20 per rollout)

执行过程：
- Eval rollouts 完成 → Eval 进度 0% → 100%
- Training 总进度: 10% (baseline) + 40% (steps) + 5% (evals)
```

## 注意事项

1. **Turn 计数**: Rollout 的 `current_turn` 是 0-indexed，但代表已完成的 turns 数量
2. **并发安全**: 每个 rollout 使用独立的数据库 session，避免并发冲突
3. **失败处理**: 如果 rollout 失败，其已完成的 turns 仍然计入进度
4. **默认值**: 如果没有历史数据，使用 30秒/turn 作为默认估算

