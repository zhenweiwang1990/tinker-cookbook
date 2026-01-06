# Env Build 实时保存功能

## 问题描述

在 rollout 执行过程中（Turn 1 已完成，但整个 rollout 还未结束），前端无法看到 Env Build 信息。

**原因**: `trajectory_data_json` 只在 rollout 完全结束后才保存到数据库，导致执行过程中无法看到 env_build 数据。

## 解决方案

### 1. 立即保存 Env Build 数据

在 `tinker_cua_agent.py` 中，当 env_build 完成后立即保存到数据库：

```python
# Complete env build logging
if self.rollout_logger:
    self.rollout_logger.log_env_build_complete(...)
    
    # 立即保存 env_build 数据到数据库
    if self.rollout_recorder is not None:
        env_build_data = {
            "execution_details": {
                "env_build": self.rollout_logger.trajectory_data.get("env_build", {})
            }
        }
        trajectory_data_json = json.dumps(env_build_data, default=str)
        self.rollout_recorder.update(trajectory_data_json=trajectory_data_json)
```

### 2. 数据合并逻辑

在 `rollout.py` 中，rollout 结束时保存完整数据前，先检查是否已有 env_build 数据：

```python
# 获取数据库中已有的 trajectory_data_json（可能包含早期保存的 env_build）
db_rollout = get_rollout_by_rollout_id(db_session, rollout_id)
if db_rollout and db_rollout.trajectory_data_json:
    existing_data = json.loads(db_rollout.trajectory_data_json)
    # 如果已有 env_build 但新数据中没有，保留旧的
    if (existing_data.get("execution_details", {}).get("env_build") and 
        not combined_trajectory_data.get("execution_details", {}).get("env_build")):
        combined_trajectory_data["execution_details"]["env_build"] = existing_data["execution_details"]["env_build"]
```

## 数据流

```
1. Env Build 开始
   ↓
2. 记录各个阶段（Box Creation, APK Installation, Prehook）
   ↓
3. Env Build 完成
   ↓
4. 立即保存 env_build 到数据库
   trajectory_data_json = {"execution_details": {"env_build": {...}}}
   ↓
5. 前端可以立即看到 Env Build 信息
   ↓
6. Turn 1, Turn 2, ... 执行
   ↓
7. Rollout 完成
   ↓
8. 保存完整的 trajectory_data_json
   - 如果 rollout_logger.trajectory_data 中有 env_build，使用它
   - 如果没有，保留数据库中已有的 env_build
   trajectory_data_json = {
       "training_data": [...],
       "execution_details": {
           "env_build": {...},  // 可能来自早期保存或 rollout_logger
           "turns": [...],
           ...
       }
   }
```

## 修改的文件

1. **tinker_cookbook/recipes/cua_rl/agent/tinker_cua_agent.py**
   - 在 `log_env_build_complete()` 后立即保存 env_build 数据

2. **tinker_cookbook/recipes/cua_rl/core/rollout.py**
   - 添加数据合并逻辑，保留早期保存的 env_build 数据

## 测试

1. 启动一个新的 rollout
2. 在 Turn 1 完成后，立即刷新前端页面
3. 应该能看到 "Env Build" 标签页，包含：
   - Box Creation 阶段
   - APK Installation 阶段
   - Prehook Execution 阶段（如果有）
   - 每个阶段的耗时和详细信息

## 注意事项

1. **数据库事务**: env_build 数据保存使用 `rollout_recorder.update()`，会自动提交事务
2. **错误处理**: 如果保存失败，只记录警告，不影响 rollout 继续执行
3. **数据一致性**: rollout 结束时的完整保存会确保最终数据的完整性
4. **向后兼容**: 即使旧的 rollout 没有早期保存，也能正常工作

## 性能影响

- 增加一次数据库写入操作（env_build 完成时）
- 写入的数据量很小（通常 < 5KB）
- 对整体性能影响可忽略

