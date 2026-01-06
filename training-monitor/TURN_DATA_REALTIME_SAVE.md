# Turn 数据实时保存功能

## 问题描述

在 rollout 执行过程中，即使 Turn 1 已经完成，前端仍然看不到 Turn 1 的 Action 解析数据（action_results, tool_executions 等）。

**原因**: Turn 的详细数据存储在 `rollout_logger.trajectory_data["turns"]` 中，但和 env_build 一样，只在 rollout 完全结束后才保存到数据库。

## 解决方案

### 1. 创建辅助方法保存 Turn 数据

在 `TinkerCuaAgent` 类中添加 `_save_turn_data_to_db()` 方法：

```python
def _save_turn_data_to_db(self, turn: int):
    """Save the completed turn's data to database immediately."""
    if not self.rollout_logger or not self.rollout_recorder:
        return
    
    # 从 rollout_logger.trajectory_data 获取已完成的 turn 数据
    completed_turn_data = ...
    
    # 从数据库读取现有的 trajectory_data_json
    existing_data = ...
    
    # 将新 turn 数据添加到 execution_details.turns 数组
    if turn not in existing turns:
        existing_data['execution_details']['turns'].append(completed_turn_data)
    
    # 保存更新后的数据
    self.rollout_recorder.update(trajectory_data_json=...)
```

### 2. 在每次 end_turn 后调用

在所有 `self.rollout_logger.end_turn(turn)` 调用后立即调用：

```python
self.rollout_logger.end_turn(turn)
# 立即保存 turn 数据到数据库
self._save_turn_data_to_db(turn)
```

有 3 处需要修改（3 种不同的执行路径）。

## Turn 数据结构

每个 turn 的数据包含：

```python
{
    "turn_num": 1,
    "max_turns": 20,
    "start_time": 1234567890.123,
    "end_time": 1234567892.456,
    "duration": 2.333,
    "screenshot_uri": "path/to/screenshot.png",
    "model_response": "I will tap on the search button...",
    "parse_success": true,
    "tool_calls": [...],
    "action_results": [
        {
            "action_type": "tap",
            "target": "search button",
            "coordinates": {"x": 100, "y": 200},
            "coord_time": 0.5,
            "exec_time": 1.2,
            "total_time": 1.7
        }
    ],
    "tool_executions": [...]
}
```

## 数据流

```
1. Turn 开始
   ↓
2. 截图 → log_screenshot()
   ↓
3. 模型推理 → log_model_inference()
   ↓
4. 工具调用解析 → log_tool_calls()
   ↓
5. Action 执行 → log_action()
   ↓
6. Turn 结束 → end_turn()
   - 数据保存到 rollout_logger.trajectory_data["turns"]
   ↓
7. 立即保存到数据库 → _save_turn_data_to_db()
   - 读取现有 trajectory_data_json
   - 添加新 turn 到 execution_details.turns
   - 保存更新后的 JSON
   ↓
8. 前端可以立即看到 Turn 数据（包括 Action 解析）
```

## 与 Env Build 的配合

现在整个 rollout 的数据保存时机：

1. **Env Build 完成时**: 立即保存 `execution_details.env_build`
2. **每个 Turn 完成时**: 立即保存 `execution_details.turns[n]`
3. **Rollout 完成时**: 保存完整的 `trajectory_data_json`，包括：
   - `training_data`: 用于训练的 token-level 数据
   - `execution_details.env_build`: 环境构建信息（已有）
   - `execution_details.turns`: 所有 turns 信息（已有）
   - `execution_details.adb_validation`: 验证信息

## 数据合并

Rollout 完成时的保存逻辑已经支持数据合并（在 `rollout.py` 中）：

```python
# 获取数据库中已有的 trajectory_data_json
existing_data = json.loads(db_rollout.trajectory_data_json)

# 如果已有 env_build 但新数据中没有，保留旧的
if existing_data.get("execution_details", {}).get("env_build"):
    combined_trajectory_data["execution_details"]["env_build"] = existing_data["execution_details"]["env_build"]

# turns 数据同理 - rollout_logger 中应该有完整的 turns 数据
```

## 修改的文件

1. **tinker_cookbook/recipes/cua_rl/agent/tinker_cua_agent.py**
   - 添加 `_save_turn_data_to_db()` 方法
   - 在 3 处 `end_turn()` 调用后添加 `_save_turn_data_to_db()` 调用

2. **tinker_cookbook/recipes/cua_rl/core/rollout.py**
   - 已有数据合并逻辑，支持保留早期保存的数据

## 前端展示

现在前端可以实时看到：

1. **Env Build Tab**: 
   - 环境构建完成后立即可见
   - 显示各个阶段、耗时、状态

2. **Turn N Tabs**:
   - Turn 完成后立即可见
   - 显示：
     - 模型响应文本
     - Action 解析结果
     - Tool 执行详情
     - 坐标信息
     - Before/After 截图

3. **Tab 耗时徽章**:
   - 每个 Tab 显示该阶段的耗时
   - 灰色徽章，不影响布局

## 性能影响

- 每个 turn 完成时增加一次数据库写入
- 写入的数据量小（通常 < 10KB per turn）
- 使用 `rollout_recorder.update()` 自动提交事务
- 对整体性能影响可忽略
- 大幅提升用户体验（可实时查看 rollout 进度）

## 错误处理

- 如果保存失败，只记录警告，不影响 rollout 继续执行
- 使用 try-except 包裹所有数据库操作
- Rollout 结束时的完整保存作为最终保障

## 测试

1. 启动新的 rollout
2. 等待 Turn 1 完成
3. 刷新前端页面
4. 检查：
   - ✅ "Env Build" 标签可见且有数据
   - ✅ "Turn 1" 标签可见且有数据
   - ✅ Turn 1 页面显示：
     - Model Response
     - Action Details (action_type, coordinates, timing)
     - Before/After 截图
   - ✅ 所有 Tab 显示耗时徽章

