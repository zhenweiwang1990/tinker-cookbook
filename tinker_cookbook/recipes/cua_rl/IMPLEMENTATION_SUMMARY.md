# 进度追踪系统实现总结

## 实现概述

本次实现了一个完整的、基于 turns 的进度追踪系统，能够正确记录和展示 Training、Baseline、Eval、Step、Group 和 Rollout 等所有层级的状态和进度信息。

## 主要特性

### 1. 统一的进度计算
- **基于 Turns**: 所有进度都以 turn 为基本单元计算，因为 turn 是最小的可测量执行单元
- **自动聚合**: 子实体的进度自动向上聚合到父实体
- **状态管理**: 支持 pending、running、completed、failed 状态

### 2. 时间估算
- **平均耗时**: 基于历史数据计算每个 turn 的平均耗时
- **预计总时间**: 根据总 turns 数量和平均耗时估算总时间
- **剩余时间**: 根据剩余 turns 数量估算剩余时间
- **层级估算**: 支持从 Rollout 到 Training 的所有层级

### 3. 层级结构

```
Training (包含 N 个 Step + N/eval_every 个 Eval + 1 个 Baseline)
  ├─ Baseline (1个)
  │   └─ Groups (Task数量个)
  │       └─ Rollouts (group_size个)
  │           └─ Turns (max_turns个)
  ├─ Steps (N个)
  │   └─ Groups (Task数量个)
  │       └─ Rollouts (group_size个)
  │           └─ Turns (max_turns个)
  └─ Evals (N/eval_every个)
      └─ Groups (Task数量个)
          └─ Rollouts (1个，因为eval每个task只run一次)
              └─ Turns (max_turns个)
```

## 新增文件

### 1. `database/progress_tracker.py`
**核心进度追踪模块**，提供了：
- `ProgressTracker` 类：统一的进度追踪接口
- `ProgressStats` 数据类：进度统计信息
- 各层级的进度更新方法：
  - `update_rollout_progress()` - 更新 Rollout 进度
  - `update_group_progress()` - 更新 Group 进度（聚合 Rollouts）
  - `update_step_progress()` - 更新 Step 进度（聚合 Groups）
  - `update_eval_progress()` - 更新 Eval 进度（聚合 Groups）
  - `update_baseline_progress()` - 更新 Baseline 进度（聚合 Groups）
  - `update_training_progress()` - 更新 Training 进度（聚合所有）
- 时间估算功能：
  - `_calculate_avg_turn_time()` - 计算平均 turn 耗时（带缓存）
  - `format_time_estimate()` - 格式化时间显示
- 自动级联更新：子实体更新时自动更新父实体

### 2. `alembic/versions/add_progress_tracking_fields.py`
**数据库迁移脚本**，添加了以下字段到 Training、Baseline、Eval、Step、Group 表：
- `avg_turn_time`: 平均每个 turn 的耗时（秒）
- `estimated_total_time`: 预计总耗时（秒）
- `estimated_remaining_time`: 预计剩余时间（秒）

### 3. `PROGRESS_TRACKING.md`
**详细文档**，包含：
- 系统概述和层级结构
- 进度计算公式
- 时间估算方法
- 使用示例和最佳实践
- 性能考虑
- 调试和监控指南

## 修改的文件

### 1. `database/database_models.py`
**为所有实体模型添加进度追踪字段**：
- Training 模型
- Baseline 模型
- Eval 模型
- Step 模型
- Group 模型

### 2. `database/rollout_recorder.py`
**集成进度追踪器**：
- 在 `__init__` 中初始化 `ProgressTracker`
- 在 `start_rollout()` 中接收 `max_turns` 参数
- 在 `start_turn()` 中调用 `update_rollout_progress()`，自动更新进度和级联更新
- 添加进度和 ETA 日志

### 3. `core/rollout.py`
**传递 max_turns 参数**：
- 在 `_run_single_env_rollout()` 函数签名中添加 `max_turns` 参数
- 在调用 `rollout_recorder.start_rollout()` 时传递 `max_turns`
- 在调用 `_run_single_env_rollout()` 时传递 `max_turns`

### 4. `core/train.py`
**初始化进度追踪器**：
- 在训练开始时创建 `ProgressTracker` 实例
- 为整个训练会话提供统一的进度追踪

## 进度计算公式

### Rollout 进度
```python
progress = (current_turn / max_turns) * 100%
```

### Group 进度
```python
total_turns = sum(rollout.max_turns for all rollouts)
completed_turns = sum(rollout.num_turns for completed rollouts + 
                     rollout.current_turn for running rollouts)
progress = (completed_turns / total_turns) * 100%
```

### Step 进度（两阶段）
```python
# 阶段1: Rollout (80%)
rollout_progress = (completed_turns / total_turns) * 80%

# 阶段2: Training (20%)
training_progress = 20% if training_completed else 0%

# 总进度
progress = rollout_progress + training_progress
```

### Eval/Baseline 进度
```python
# 与 Group 相同，聚合所有 groups 的 turns
progress = (completed_turns / total_turns) * 100%
```

### Training 进度（加权）
```python
baseline_progress = baseline.progress_percent * 0.1  # 10%
steps_progress = avg(step.progress_percent) * 0.8     # 80%
evals_progress = avg(eval.progress_percent) * 0.1     # 10%

progress = baseline_progress + steps_progress + evals_progress
```

## 时间估算

### 平均 Turn 耗时计算
优先级从高到低：
1. **Rollout 层级**: `rollout.rollout_time / rollout.num_turns`
2. **Group 层级**: 聚合该 group 所有 completed rollouts 的平均值
3. **Step/Eval/Baseline 层级**: 聚合该实体所有 groups 的平均值
4. **Training 层级**: 聚合所有 baseline/steps/evals 的平均值
5. **默认值**: 30 秒/turn（无历史数据时）

### 时间预测
```python
avg_turn_time = total_time / completed_turns
estimated_total_time = total_turns * avg_turn_time
estimated_remaining_time = remaining_turns * avg_turn_time
```

## 自动级联更新流程

当更新 Rollout 进度时：
```
1. update_rollout_progress(rollout_id, current_turn, max_turns)
   ↓
2. 计算 Rollout 的进度和时间估算
   ↓
3. 自动调用 update_group_progress(group_id)
   ↓
4. 聚合所有 Rollouts，更新 Group 进度
   ↓
5. 根据 source_type 自动调用：
   - update_step_progress(step_id) 或
   - update_eval_progress(eval_id) 或
   - update_baseline_progress(baseline_id)
   ↓
6. 聚合所有 Groups，更新 Step/Eval/Baseline 进度
   ↓
7. 自动调用 update_training_progress(training_id)
   ↓
8. 聚合所有 Steps/Evals/Baseline，更新 Training 进度
```

## 使用方式

### 训练代码中
```python
# train.py 中初始化
from tinker_cookbook.recipes.cua_rl.database.progress_tracker import ProgressTracker

progress_tracker = ProgressTracker(db_session)
```

### Rollout 过程中
```python
# rollout_recorder.py 中自动更新
def start_turn(self, turn_num: int):
    # ... 创建 turn 记录 ...
    
    # 自动更新进度（包括所有级联更新）
    progress_stats = self.progress_tracker.update_rollout_progress(
        rollout_id=self.rollout_db_id,
        current_turn=turn_num,
        max_turns=self.max_turns,
        status="running",
    )
    
    # 日志输出
    logger.debug(
        f"Turn {turn_num}, progress={progress_stats.progress_percent:.1f}%, "
        f"ETA={self.progress_tracker.format_time_estimate(progress_stats.estimated_remaining_time)}"
    )
```

## 数据库迁移

运行以下命令应用迁移：
```bash
cd tinker_cookbook/recipes/cua_rl
alembic upgrade head
```

这会添加所有必需的进度追踪字段到数据库。

## 性能优化

1. **缓存机制**: `ProgressTracker` 缓存已计算的平均 turn 时间
2. **独立 Session**: 每个 rollout 使用独立的数据库 session，避免并发冲突
3. **批量计算**: 进度更新在事务中完成，确保一致性
4. **按需更新**: 只在必要时才计算和更新进度

## 监控和调试

系统会输出详细的进度日志：
```
[RolloutRecorder] Started turn 5 (ID=123), progress=25.0%, ETA=3.5m
[Progress] Group 0: 40 completed turns / 100 total, avg_turn_time=7.2s
[Progress] Step 0: 320 completed turns / 800 total, progress=40.0%
[Progress] Training: 1520 completed turns / 4000 total, progress=38.0%
```

## 前端展示支持

数据库现在包含所有必要的字段，前端可以直接查询展示：
- `progress_percent`: 进度百分比
- `status`: 当前状态
- `estimated_remaining_time`: 剩余时间（秒）
- `avg_turn_time`: 平均 turn 耗时（用于速度显示）

## 测试建议

1. **单个 Rollout**: 验证 turn-by-turn 进度更新
2. **Group 聚合**: 验证多个 rollouts 的进度聚合正确
3. **Step 两阶段**: 验证 rollout 和 training 阶段的进度分配
4. **Training 加权**: 验证 baseline/steps/evals 的权重计算
5. **时间估算**: 验证在不同数据量下的时间预测准确性

## 未来优化方向

1. **动态权重**: 根据实际执行情况动态调整 step 的 rollout/training 阶段权重
2. **学习曲线**: 考虑 turn 耗时的趋势变化，提高时间预测准确性
3. **并发优化**: 在高并发场景下进一步优化数据库查询
4. **缓存策略**: 实现更智能的缓存失效策略
5. **实时推送**: 集成 WebSocket 实时推送进度更新到前端

## 总结

本次实现提供了一个完整、可靠、基于 turns 的进度追踪系统，具有以下优势：

1. ✅ **统一标准**: 所有层级都基于 turns 计算，易于理解和维护
2. ✅ **自动聚合**: 子实体更新自动级联到父实体，无需手动管理
3. ✅ **精确估算**: 基于历史数据的时间估算，越用越准
4. ✅ **完整状态**: 支持 pending/running/completed/failed 全状态管理
5. ✅ **性能优化**: 缓存和独立 session 确保高并发场景下的性能
6. ✅ **易于监控**: 详细的日志和结构化的数据库字段便于监控和调试
7. ✅ **前端友好**: 所有必要信息都在数据库中，前端直接查询即可展示

