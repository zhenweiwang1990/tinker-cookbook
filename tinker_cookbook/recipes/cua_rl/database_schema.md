# CUA RL Training Database Schema

## 表结构设计

### 1. training (训练会话表)
记录整个训练会话的信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 训练ID | AUTOINCREMENT |
| run_name | TEXT | 运行名称 | NOT NULL, UNIQUE |
| log_path | TEXT | 日志路径 | NOT NULL |
| model_name | TEXT | 模型名称 | NOT NULL |
| lora_rank | INTEGER | LoRA rank | |
| learning_rate | REAL | 学习率 | |
| batch_size | INTEGER | 批次大小 | |
| group_size | INTEGER | 组大小 | |
| groups_per_batch | INTEGER | 每批次组数 | |
| max_tokens | INTEGER | 最大token数 | |
| temperature | REAL | 温度参数 | |
| kl_penalty_coef | REAL | KL惩罚系数 | |
| num_substeps | INTEGER | 子步骤数 | |
| max_turns | INTEGER | 最大回合数 | |
| seed | INTEGER | 随机种子 | |
| box_type | TEXT | GBox类型 (android/linux) | |
| renderer_name | TEXT | 渲染器名称 | |
| wandb_project | TEXT | WandB项目名 | |
| wandb_name | TEXT | WandB运行名 | |
| status | TEXT | 当前状态 (pending/initializing/running/completed/failed/paused/cancelled) | DEFAULT 'pending' |
| progress_percent | REAL | 进度百分比 (0-100) | DEFAULT 0.0 |
| current_step | INTEGER | 当前步骤号 | |
| total_steps | INTEGER | 总步骤数 | |
| current_phase | TEXT | 当前阶段 (initialization/rollout/training/evaluation/checkpointing) | |
| status_message | TEXT | 状态消息 | |
| error_message | TEXT | 错误信息 | |
| start_time | TIMESTAMP | 开始时间 | |
| end_time | TIMESTAMP | 结束时间 | |
| last_heartbeat | TIMESTAMP | 最后心跳时间 | DEFAULT CURRENT_TIMESTAMP |
| config_json | TEXT | 完整配置JSON | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新时间 | DEFAULT CURRENT_TIMESTAMP |

### 2. baseline (基线评估表)
记录基线评估的信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 基线ID | AUTOINCREMENT |
| training_id | INTEGER | 训练ID | NOT NULL, REFERENCES training(id) |
| model_path | TEXT | 模型路径 | NOT NULL |
| status | TEXT | 状态 (pending/running/completed/failed/cancelled) | DEFAULT 'pending' |
| progress_percent | REAL | 进度百分比 (0-100) | DEFAULT 0.0 |
| current_task_index | INTEGER | 当前任务索引 | |
| total_tasks | INTEGER | 总任务数 | |
| completed_tasks | INTEGER | 已完成任务数 | |
| current_phase | TEXT | 当前阶段 (initialization/rollout/validation/aggregation) | |
| status_message | TEXT | 状态消息 | |
| error_message | TEXT | 错误信息 | |
| start_time | TIMESTAMP | 开始时间 | |
| end_time | TIMESTAMP | 结束时间 | |
| eval_time | TIMESTAMP | 评估完成时间 | |
| success_rate | REAL | 成功率 | |
| avg_reward | REAL | 平均奖励 | |
| avg_turns | REAL | 平均回合数 | |
| successful_tasks | INTEGER | 成功任务数 | |
| metrics_json | TEXT | 详细指标JSON | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新时间 | DEFAULT CURRENT_TIMESTAMP |

### 3. eval (评估表)
记录训练过程中的评估信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 评估ID | AUTOINCREMENT |
| training_id | INTEGER | 训练ID | NOT NULL, REFERENCES training(id) |
| step | INTEGER | 训练步骤号 | NOT NULL |
| model_path | TEXT | 模型路径 | NOT NULL |
| status | TEXT | 状态 (pending/running/completed/failed/cancelled) | DEFAULT 'pending' |
| progress_percent | REAL | 进度百分比 (0-100) | DEFAULT 0.0 |
| current_task_index | INTEGER | 当前任务索引 | |
| total_tasks | INTEGER | 总任务数 | |
| completed_tasks | INTEGER | 已完成任务数 | |
| current_phase | TEXT | 当前阶段 (initialization/rollout/validation/aggregation) | |
| status_message | TEXT | 状态消息 | |
| error_message | TEXT | 错误信息 | |
| start_time | TIMESTAMP | 开始时间 | |
| end_time | TIMESTAMP | 结束时间 | |
| eval_time | TIMESTAMP | 评估完成时间 | |
| success_rate | REAL | 成功率 | |
| avg_reward | REAL | 平均奖励 | |
| avg_turns | REAL | 平均回合数 | |
| successful_tasks | INTEGER | 成功任务数 | |
| metrics_json | TEXT | 详细指标JSON | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新时间 | DEFAULT CURRENT_TIMESTAMP |
| UNIQUE(training_id, step) | | | |

### 4. task (任务表)
记录所有任务的信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 任务ID | AUTOINCREMENT |
| task_id | TEXT | 任务唯一标识符 | NOT NULL, UNIQUE |
| name | TEXT | 任务名称 | NOT NULL |
| description | TEXT | 任务描述 | NOT NULL |
| difficulty | TEXT | 难度 (easy/medium/hard) | |
| category | TEXT | 类别 (system/navigation/settings/app/input) | |
| max_steps | INTEGER | 最大步骤数 | |
| validation_type | TEXT | 验证类型 (state/screenshot/api) | |
| validation_query | TEXT | 验证查询 | |
| expected_result | TEXT | 预期结果 | |
| tags | TEXT | 标签列表 (JSON数组) | |
| prerequisites | TEXT | 前置条件 (JSON数组) | |
| app_name | TEXT | 应用名称 (airbnb/instagram等) | |
| source_type | TEXT | 来源类型 (demo_training/demo_eval/task_adapter等) | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新时间 | DEFAULT CURRENT_TIMESTAMP |

### 5. validator (验证器表)
记录验证器的信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 验证器ID | AUTOINCREMENT |
| task_id | INTEGER | 任务ID | NOT NULL, REFERENCES task(id) |
| validator_type | TEXT | 验证器类型 | NOT NULL |
| validation_query | TEXT | 验证查询 | |
| validation_method | TEXT | 验证方法 | |
| config_json | TEXT | 验证器配置JSON | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |

### 6. step (训练步骤表)
记录每个训练步骤的信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 步骤ID | AUTOINCREMENT |
| training_id | INTEGER | 训练ID | NOT NULL, REFERENCES training(id) |
| step | INTEGER | 步骤号 | NOT NULL |
| batch | INTEGER | 批次号 | |
| status | TEXT | 状态 (pending/rollout_collecting/rollout_running/training/completed/failed) | DEFAULT 'pending' |
| progress_percent | REAL | 进度百分比 (0-100) | DEFAULT 0.0 |
| current_phase | TEXT | 当前阶段 (rollout_collection/rollout_execution/training/checkpointing) | |
| rollout_progress | TEXT | 滚动进度 (JSON: {completed_groups, total_groups, completed_envs, total_envs}) | |
| training_progress | TEXT | 训练进度 (JSON: {completed_substeps, total_substeps}) | |
| status_message | TEXT | 状态消息 | |
| error_message | TEXT | 错误信息 | |
| start_time | TIMESTAMP | 开始时间 | |
| end_time | TIMESTAMP | 结束时间 | |
| rollout_start_time | TIMESTAMP | 滚动开始时间 | |
| rollout_end_time | TIMESTAMP | 滚动结束时间 | |
| training_start_time | TIMESTAMP | 训练开始时间 | |
| training_end_time | TIMESTAMP | 训练结束时间 | |
| learning_rate | REAL | 当前学习率 | |
| model_path | TEXT | 模型路径 | |
| checkpoint_path | TEXT | 检查点路径 | |
| loss | REAL | 损失值 | |
| kl_divergence | REAL | KL散度 | |
| policy_gradient_norm | REAL | 策略梯度范数 | |
| reward_mean | REAL | 平均奖励 | |
| reward_std | REAL | 奖励标准差 | |
| num_trajectories | INTEGER | 轨迹数量 | |
| num_tokens | INTEGER | Token数量 | |
| metrics_json | TEXT | 详细指标JSON | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新时间 | DEFAULT CURRENT_TIMESTAMP |
| UNIQUE(training_id, step) | | | |

### 7. rollout (滚动表)
记录每次滚动的信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 滚动ID | AUTOINCREMENT |
| source_type | TEXT | 来源类型 (step/eval/baseline) | NOT NULL |
| step_id | INTEGER | 步骤ID (当source_type='step'时) | REFERENCES step(id) |
| eval_id | INTEGER | 评估ID (当source_type='eval'时) | REFERENCES eval(id) |
| baseline_id | INTEGER | 基线ID (当source_type='baseline'时) | REFERENCES baseline(id) |
| rollout_id | TEXT | 滚动唯一标识符 | NOT NULL, UNIQUE |
| batch | INTEGER | 批次号 | |
| group | INTEGER | 组号 | |
| env_index | INTEGER | 环境索引 | |
| task_id | INTEGER | 任务ID | NOT NULL, REFERENCES task(id) |
| model_path | TEXT | 模型路径 | NOT NULL |
| is_eval | INTEGER | 是否为评估 (0/1) | DEFAULT 0 |
| status | TEXT | 状态 (pending/env_creation/agent_init/running/completed/failed/cancelled) | DEFAULT 'pending' |
| progress_percent | REAL | 进度百分比 (0-100) | DEFAULT 0.0 |
| current_phase | TEXT | 当前阶段 (env_creation/agent_initialization/task_execution/validation/cleanup) | |
| current_turn | INTEGER | 当前回合号 | |
| status_message | TEXT | 状态消息 | |
| error_message | TEXT | 错误信息 | |
| start_time | TIMESTAMP | 开始时间 | |
| end_time | TIMESTAMP | 结束时间 | |
| env_creation_time | TIMESTAMP | 环境创建时间 | |
| agent_init_time | TIMESTAMP | Agent初始化时间 | |
| task_start_time | TIMESTAMP | 任务开始时间 | |
| task_end_time | TIMESTAMP | 任务结束时间 | |
| validation_time | TIMESTAMP | 验证时间 | |
| rollout_time | REAL | 滚动耗时(秒) | |
| task_completed | INTEGER | 任务是否完成 (0/1) | |
| task_success | INTEGER | 任务是否成功 (0/1) | |
| agent_reported_success | INTEGER | Agent报告成功 (0/1) | |
| validation_passed | INTEGER | 验证是否通过 (0/1) | |
| num_turns | INTEGER | 回合数 | |
| max_turns | INTEGER | 最大回合数 | |
| reward | REAL | 奖励值 | |
| temperature | REAL | 温度参数 | |
| num_total_actions | INTEGER | 总动作数 | |
| consecutive_repeated_actions | INTEGER | 连续重复动作数 | |
| parse_errors | INTEGER | 解析错误数 | |
| tool_name_errors | INTEGER | 工具名错误数 | |
| tool_arg_errors | INTEGER | 工具参数错误数 | |
| runtime_errors | INTEGER | 运行时错误数 | |
| ran_out_of_turns | INTEGER | 是否用完回合 (0/1) | |
| attempted_completion | INTEGER | 是否尝试完成 (0/1) | |
| turn_first_success | INTEGER | 首次成功回合号 | |
| turn_task_completed | INTEGER | 任务完成回合号 | |
| errors | TEXT | 错误列表 (JSON数组) | |
| summary_json | TEXT | 摘要信息JSON | |
| trajectory_path | TEXT | 轨迹文件路径 | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新时间 | DEFAULT CURRENT_TIMESTAMP |

**约束说明**:
- `source_type` 必须为 'step'、'eval' 或 'baseline' 之一
- 当 `source_type='step'` 时，`step_id` 必须非空，`eval_id` 和 `baseline_id` 必须为空
- 当 `source_type='eval'` 时，`eval_id` 必须非空，`step_id` 和 `baseline_id` 必须为空
- 当 `source_type='baseline'` 时，`baseline_id` 必须非空，`step_id` 和 `eval_id` 必须为空
- 此约束需要在应用层或数据库触发器层面实现

### 8. turn (回合表)
记录每个回合的信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 回合ID | AUTOINCREMENT |
| rollout_id | INTEGER | 滚动ID | NOT NULL, REFERENCES rollout(id) |
| turn | INTEGER | 回合号 | NOT NULL |
| start_time | TIMESTAMP | 开始时间 | NOT NULL, DEFAULT CURRENT_TIMESTAMP |
| end_time | TIMESTAMP | 结束时间 | |
| turn_time | REAL | 回合耗时(秒) | |
| reward | REAL | 奖励值 | |
| episode_done | INTEGER | 是否结束 (0/1) | |
| metrics_json | TEXT | 指标JSON | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| UNIQUE(rollout_id, turn) | | | |

### 9. action (动作表)
记录每个动作的信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 动作ID | AUTOINCREMENT |
| turn_id | INTEGER | 回合ID | NOT NULL, REFERENCES turn(id) |
| action_type | TEXT | 动作类型 | |
| tool_name | TEXT | 工具名称 | |
| tool_args | TEXT | 工具参数 (JSON) | |
| tokens | TEXT | Token列表 (JSON数组) | |
| logprobs | TEXT | Log概率列表 (JSON数组) | |
| num_tokens | INTEGER | Token数量 | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |

### 10. obs (观察表)
记录每个观察的信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 观察ID | AUTOINCREMENT |
| turn_id | INTEGER | 回合ID | NOT NULL, REFERENCES turn(id) |
| obs_type | TEXT | 观察类型 (screenshot/text/multimodal) | |
| screenshot_uri | TEXT | 截图URI | |
| text_content | TEXT | 文本内容 | |
| model_input_json | TEXT | ModelInput JSON | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |

### 11. validation (验证表)
记录验证结果。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 验证ID | AUTOINCREMENT |
| rollout_id | INTEGER | 滚动ID | NOT NULL, REFERENCES rollout(id) |
| validator_id | INTEGER | 验证器ID | REFERENCES validator(id) |
| validation_time | TIMESTAMP | 验证时间 | NOT NULL, DEFAULT CURRENT_TIMESTAMP |
| validation_query | TEXT | 验证查询 | |
| expected_result | TEXT | 预期结果 | |
| actual_result | TEXT | 实际结果 | |
| success | INTEGER | 是否成功 (0/1) | NOT NULL |
| execution_time | REAL | 执行时间(秒) | |
| error_message | TEXT | 错误信息 | |
| details_json | TEXT | 详细信息JSON | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |

### 12. environment (环境表)
记录环境信息。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 环境ID | AUTOINCREMENT |
| rollout_id | INTEGER | 滚动ID | NOT NULL, REFERENCES rollout(id) |
| env_type | TEXT | 环境类型 (android/linux) | NOT NULL |
| status | TEXT | 状态 (pending/creating/running/terminated/error) | DEFAULT 'pending' |
| gbox_id | TEXT | GBox ID | |
| box_type | TEXT | Box类型 | |
| creation_time | TIMESTAMP | 创建时间 | |
| termination_time | TIMESTAMP | 终止时间 | |
| status_message | TEXT | 状态消息 | |
| error_message | TEXT | 错误信息 | |
| config_json | TEXT | 环境配置JSON | |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | 更新时间 | DEFAULT CURRENT_TIMESTAMP |

### 13. status_history (状态历史表)
记录所有有状态实体的状态变化历史。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INTEGER PRIMARY KEY | 历史ID | AUTOINCREMENT |
| entity_type | TEXT | 实体类型 (training/baseline/eval/step/rollout/environment) | NOT NULL |
| entity_id | INTEGER | 实体ID | NOT NULL |
| old_status | TEXT | 旧状态 | |
| new_status | TEXT | 新状态 | NOT NULL |
| progress_percent | REAL | 进度百分比 | |
| status_message | TEXT | 状态消息 | |
| metadata_json | TEXT | 元数据JSON | |
| changed_at | TIMESTAMP | 变更时间 | NOT NULL, DEFAULT CURRENT_TIMESTAMP |
| created_at | TIMESTAMP | 创建时间 | DEFAULT CURRENT_TIMESTAMP |

## 索引设计

为了提高查询性能，创建以下索引：

1. `training(run_name)` - 快速查找训练会话
2. `training(status)` - 按状态查询训练
3. `training(status, last_heartbeat)` - 查找活跃的训练会话
4. `baseline(training_id)` - 查找训练的所有基线评估
5. `baseline(status)` - 按状态查询基线评估
6. `eval(training_id, step)` - 查找特定步骤的评估
7. `eval(status)` - 按状态查询评估
8. `task(task_id)` - 快速查找任务
9. `validator(task_id)` - 查找任务的验证器
10. `step(training_id, step)` - 快速查找步骤
11. `step(status)` - 按状态查询步骤
12. `rollout(source_type, step_id)` - 查找步骤的所有滚动
13. `rollout(source_type, eval_id)` - 查找评估的所有滚动
14. `rollout(source_type, baseline_id)` - 查找基线评估的所有滚动
15. `rollout(task_id)` - 查找任务的所有滚动
16. `rollout(rollout_id)` - 快速查找滚动
17. `rollout(status)` - 按状态查询滚动
18. `turn(rollout_id, turn)` - 快速查找回合
19. `action(turn_id)` - 查找回合的所有动作
20. `obs(turn_id)` - 查找回合的所有观察
21. `validation(rollout_id)` - 查找滚动的验证结果
22. `environment(rollout_id)` - 查找滚动的环境信息
23. `environment(status)` - 按状态查询环境
24. `status_history(entity_type, entity_id)` - 查找实体的状态历史
25. `status_history(entity_type, entity_id, changed_at)` - 按时间排序的状态历史

## 关系图

```
training (1) ──< (N) step
training (1) ──< (N) baseline
training (1) ──< (N) eval
task (1) ──< (N) validator
task (1) ──< (N) rollout
step (1) ──< (N) rollout  (source_type='step')
eval (1) ──< (N) rollout  (source_type='eval')
baseline (1) ──< (N) rollout  (source_type='baseline')
rollout (1) ──< (N) turn
rollout (1) ──< (1) validation
rollout (1) ──< (1) environment
turn (1) ──< (N) action
turn (1) ──< (N) obs
validator (1) ──< (N) validation
* (1) ──< (N) status_history  (所有有状态的实体)
```

**注意**: rollout 表通过 `source_type` 字段区分来源，并通过对应的 `step_id`、`eval_id` 或 `baseline_id` 关联到不同的实体。每个 rollout 只能关联到一种来源类型。

## 状态说明

### Training 状态
- `pending`: 待开始
- `initializing`: 初始化中
- `running`: 运行中
- `completed`: 已完成
- `failed`: 失败
- `paused`: 暂停
- `cancelled`: 已取消

### Baseline/Eval 状态
- `pending`: 待开始
- `running`: 运行中
- `completed`: 已完成
- `failed`: 失败
- `cancelled`: 已取消

### Step 状态
- `pending`: 待开始
- `rollout_collecting`: 收集滚动中
- `rollout_running`: 滚动执行中
- `training`: 训练中
- `completed`: 已完成
- `failed`: 失败

### Rollout 状态
- `pending`: 待开始
- `env_creation`: 环境创建中
- `agent_init`: Agent初始化中
- `running`: 运行中
- `completed`: 已完成
- `failed`: 失败
- `cancelled`: 已取消

### Environment 状态
- `pending`: 待创建
- `creating`: 创建中
- `running`: 运行中
- `terminated`: 已终止
- `error`: 错误

## 进展跟踪说明

### Training 进展
- `progress_percent`: 基于当前步骤/总步骤计算
- `current_step`: 当前执行的步骤号
- `total_steps`: 总步骤数（从配置或历史数据获取）
- `current_phase`: 当前阶段（initialization/rollout/training/evaluation/checkpointing）

### Step 进展
- `progress_percent`: 基于当前阶段和子任务计算
- `rollout_progress`: JSON格式，包含 `{completed_groups, total_groups, completed_envs, total_envs}`
- `training_progress`: JSON格式，包含 `{completed_substeps, total_substeps}`
- `current_phase`: 当前阶段（rollout_collection/rollout_execution/training/checkpointing）

### Rollout 进展
- `progress_percent`: 基于当前回合/最大回合计算
- `current_turn`: 当前执行的回合号
- `current_phase`: 当前阶段（env_creation/agent_initialization/task_execution/validation/cleanup）

### Baseline/Eval 进展
- `progress_percent`: 基于已完成任务/总任务计算
- `current_task_index`: 当前任务索引
- `total_tasks`: 总任务数
- `completed_tasks`: 已完成任务数
- `current_phase`: 当前阶段（initialization/rollout/validation/aggregation）

