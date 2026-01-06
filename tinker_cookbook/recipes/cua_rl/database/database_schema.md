# CUA RL Training Database Schema

## Table Structure Design

### 1. training (Training Session Table)
Records information for the entire training session.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Training ID | AUTOINCREMENT |
| run_name | TEXT | Run name | NOT NULL, UNIQUE |
| log_path | TEXT | Log path | NOT NULL |
| model_name | TEXT | Model name | NOT NULL |
| lora_rank | INTEGER | LoRA rank | |
| learning_rate | REAL | Learning rate | |
| batch_size | INTEGER | Batch size | |
| group_size | INTEGER | Group size | |
| groups_per_batch | INTEGER | Groups per batch | |
| max_tokens | INTEGER | Maximum tokens | |
| temperature | REAL | Temperature parameter | |
| kl_penalty_coef | REAL | KL penalty coefficient | |
| num_substeps | INTEGER | Number of substeps | |
| max_turns | INTEGER | Maximum turns | |
| seed | INTEGER | Random seed | |
| box_type | TEXT | GBox type (android/linux) | |
| renderer_name | TEXT | Renderer name | |
| wandb_project | TEXT | WandB project name | |
| wandb_name | TEXT | WandB run name | |
| status | TEXT | Current status (pending/initializing/running/completed/failed/paused/cancelled) | DEFAULT 'pending' |
| progress_percent | REAL | Progress percentage (0-100) | DEFAULT 0.0 |
| current_step | INTEGER | Current step number | |
| total_steps | INTEGER | Total number of steps | |
| current_phase | TEXT | Current phase (initialization/rollout/training/evaluation/checkpointing) | |
| status_message | TEXT | Status message | |
| error_message | TEXT | Error message | |
| start_time | TIMESTAMP | Start time | |
| end_time | TIMESTAMP | End time | |
| last_heartbeat | TIMESTAMP | Last heartbeat time | DEFAULT CURRENT_TIMESTAMP |
| config_json | TEXT | Complete configuration JSON | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Update time | DEFAULT CURRENT_TIMESTAMP |

### 2. baseline (Baseline Evaluation Table)
Records information for baseline evaluations.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Baseline ID | AUTOINCREMENT |
| training_id | INTEGER | Training ID | NOT NULL, REFERENCES training(id) |
| model_path | TEXT | Model path | NOT NULL |
| status | TEXT | Status (pending/running/completed/failed/cancelled) | DEFAULT 'pending' |
| progress_percent | REAL | Progress percentage (0-100) | DEFAULT 0.0 |
| current_task_index | INTEGER | Current task index | |
| total_tasks | INTEGER | Total number of tasks | |
| completed_tasks | INTEGER | Number of completed tasks | |
| current_phase | TEXT | Current phase (initialization/rollout/validation/aggregation) | |
| status_message | TEXT | Status message | |
| error_message | TEXT | Error message | |
| start_time | TIMESTAMP | Start time | |
| end_time | TIMESTAMP | End time | |
| eval_time | TIMESTAMP | Evaluation completion time | |
| success_rate | REAL | Success rate | |
| avg_reward | REAL | Average reward | |
| avg_turns | REAL | Average number of turns | |
| successful_tasks | INTEGER | Number of successful tasks | |
| metrics_json | TEXT | Detailed metrics JSON | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Update time | DEFAULT CURRENT_TIMESTAMP |

### 3. eval (Evaluation Table)
Records evaluation information during training.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Evaluation ID | AUTOINCREMENT |
| training_id | INTEGER | Training ID | NOT NULL, REFERENCES training(id) |
| step | INTEGER | Training step number | NOT NULL |
| model_path | TEXT | Model path | NOT NULL |
| status | TEXT | Status (pending/running/completed/failed/cancelled) | DEFAULT 'pending' |
| progress_percent | REAL | Progress percentage (0-100) | DEFAULT 0.0 |
| current_task_index | INTEGER | Current task index | |
| total_tasks | INTEGER | Total number of tasks | |
| completed_tasks | INTEGER | Number of completed tasks | |
| current_phase | TEXT | Current phase (initialization/rollout/validation/aggregation) | |
| status_message | TEXT | Status message | |
| error_message | TEXT | Error message | |
| start_time | TIMESTAMP | Start time | |
| end_time | TIMESTAMP | End time | |
| eval_time | TIMESTAMP | Evaluation completion time | |
| success_rate | REAL | Success rate | |
| avg_reward | REAL | Average reward | |
| avg_turns | REAL | Average number of turns | |
| successful_tasks | INTEGER | Number of successful tasks | |
| metrics_json | TEXT | Detailed metrics JSON | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Update time | DEFAULT CURRENT_TIMESTAMP |
| UNIQUE(training_id, step) | | | |

### 4. task (Task Table)
Records information for all tasks.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Task ID | AUTOINCREMENT |
| task_id | TEXT | Task unique identifier | NOT NULL, UNIQUE |
| name | TEXT | Task name | NOT NULL |
| description | TEXT | Task description | NOT NULL |
| difficulty | TEXT | Difficulty (easy/medium/hard) | |
| category | TEXT | Category (system/navigation/settings/app/input) | |
| max_steps | INTEGER | Maximum number of steps | |
| validation_type | TEXT | Validation type (state/screenshot/api) | |
| validation_query | TEXT | Validation query | |
| expected_result | TEXT | Expected result | |
| tags | TEXT | Tag list (JSON array) | |
| prerequisites | TEXT | Prerequisites (JSON array) | |
| app_name | TEXT | App name (airbnb/instagram, etc.) | |
| source_type | TEXT | Source type (demo_training/demo_eval/task_adapter, etc.) | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Update time | DEFAULT CURRENT_TIMESTAMP |

### 5. validator (Validator Table)
Records validator information.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Validator ID | AUTOINCREMENT |
| task_id | INTEGER | Task ID | NOT NULL, REFERENCES task(id) |
| validator_type | TEXT | Validator type | NOT NULL |
| validation_query | TEXT | Validation query | |
| validation_method | TEXT | Validation method | |
| config_json | TEXT | Validator configuration JSON | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |

### 6. step (Training Step Table)
Records information for each training step.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Step ID | AUTOINCREMENT |
| training_id | INTEGER | Training ID | NOT NULL, REFERENCES training(id) |
| step | INTEGER | Step number | NOT NULL |
| batch | INTEGER | Batch number | |
| status | TEXT | Status (pending/rollout_collecting/rollout_running/training/completed/failed) | DEFAULT 'pending' |
| progress_percent | REAL | Progress percentage (0-100) | DEFAULT 0.0 |
| current_phase | TEXT | Current phase (rollout_collection/rollout_execution/training/checkpointing) | |
| rollout_progress | TEXT | Rollout progress (JSON: {completed_groups, total_groups, completed_envs, total_envs}) | |
| training_progress | TEXT | Training progress (JSON: {completed_substeps, total_substeps}) | |
| status_message | TEXT | Status message | |
| error_message | TEXT | Error message | |
| start_time | TIMESTAMP | Start time | |
| end_time | TIMESTAMP | End time | |
| rollout_start_time | TIMESTAMP | Rollout start time | |
| rollout_end_time | TIMESTAMP | Rollout end time | |
| training_start_time | TIMESTAMP | Training start time | |
| training_end_time | TIMESTAMP | Training end time | |
| learning_rate | REAL | Current learning rate | |
| model_path | TEXT | Model path | |
| checkpoint_path | TEXT | Checkpoint path | |
| loss | REAL | Loss value | |
| kl_divergence | REAL | KL divergence | |
| policy_gradient_norm | REAL | Policy gradient norm | |
| reward_mean | REAL | Average reward | |
| reward_std | REAL | Reward standard deviation | |
| num_trajectories | INTEGER | Number of trajectories | |
| num_tokens | INTEGER | Number of tokens | |
| metrics_json | TEXT | Detailed metrics JSON | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Update time | DEFAULT CURRENT_TIMESTAMP |
| UNIQUE(training_id, step) | | | |

### 7. rollout (Rollout Table)
Records information for each rollout.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Rollout ID | AUTOINCREMENT |
| source_type | TEXT | Source type (step/eval/baseline) | NOT NULL |
| step_id | INTEGER | Step ID (when source_type='step') | REFERENCES step(id) |
| eval_id | INTEGER | Evaluation ID (when source_type='eval') | REFERENCES eval(id) |
| baseline_id | INTEGER | Baseline ID (when source_type='baseline') | REFERENCES baseline(id) |
| rollout_id | TEXT | Rollout unique identifier | NOT NULL, UNIQUE |
| batch | INTEGER | Batch number | |
| group | INTEGER | Group number | |
| env_index | INTEGER | Environment index | |
| task_id | INTEGER | Task ID | NOT NULL, REFERENCES task(id) |
| model_path | TEXT | Model path | NOT NULL |
| is_eval | INTEGER | Is evaluation (0/1) | DEFAULT 0 |
| status | TEXT | Status (pending/env_creation/agent_init/running/completed/failed/cancelled) | DEFAULT 'pending' |
| progress_percent | REAL | Progress percentage (0-100) | DEFAULT 0.0 |
| current_phase | TEXT | Current phase (env_creation/agent_initialization/task_execution/validation/cleanup) | |
| current_turn | INTEGER | Current turn number | |
| status_message | TEXT | Status message | |
| error_message | TEXT | Error message | |
| start_time | TIMESTAMP | Start time | |
| end_time | TIMESTAMP | End time | |
| env_creation_time | TIMESTAMP | Environment creation time | |
| agent_init_time | TIMESTAMP | Agent initialization time | |
| task_start_time | TIMESTAMP | Task start time | |
| task_end_time | TIMESTAMP | Task end time | |
| validation_time | TIMESTAMP | Validation time | |
| rollout_time | REAL | Rollout duration (seconds) | |
| task_completed | INTEGER | Task completed (0/1) | |
| task_success | INTEGER | Task successful (0/1) | |
| agent_reported_success | INTEGER | Agent reported success (0/1) | |
| validation_passed | INTEGER | Validation passed (0/1) | |
| num_turns | INTEGER | Number of turns | |
| max_turns | INTEGER | Maximum number of turns | |
| reward | REAL | Reward value | |
| temperature | REAL | Temperature parameter | |
| num_total_actions | INTEGER | Total number of actions | |
| consecutive_repeated_actions | INTEGER | Number of consecutive repeated actions | |
| parse_errors | INTEGER | Number of parse errors | |
| tool_name_errors | INTEGER | Number of tool name errors | |
| tool_arg_errors | INTEGER | Number of tool argument errors | |
| runtime_errors | INTEGER | Number of runtime errors | |
| ran_out_of_turns | INTEGER | Ran out of turns (0/1) | |
| attempted_completion | INTEGER | Attempted completion (0/1) | |
| turn_first_success | INTEGER | First successful turn number | |
| turn_task_completed | INTEGER | Turn when task was completed | |
| errors | TEXT | Error list (JSON array) | |
| summary_json | TEXT | Summary information JSON | |
| trajectory_path | TEXT | Trajectory file path | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Update time | DEFAULT CURRENT_TIMESTAMP |

**Constraint Notes**:
- `source_type` must be one of 'step', 'eval', or 'baseline'
- When `source_type='step'`, `step_id` must be non-null, `eval_id` and `baseline_id` must be null
- When `source_type='eval'`, `eval_id` must be non-null, `step_id` and `baseline_id` must be null
- When `source_type='baseline'`, `baseline_id` must be non-null, `step_id` and `eval_id` must be null
- These constraints need to be implemented at the application layer or via database triggers

### 8. turn (Turn Table)
Records information for each turn.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Turn ID | AUTOINCREMENT |
| rollout_id | INTEGER | Rollout ID | NOT NULL, REFERENCES rollout(id) |
| turn | INTEGER | Turn number | NOT NULL |
| start_time | TIMESTAMP | Start time | NOT NULL, DEFAULT CURRENT_TIMESTAMP |
| end_time | TIMESTAMP | End time | |
| turn_time | REAL | Turn duration (seconds) | |
| reward | REAL | Reward value | |
| episode_done | INTEGER | Episode done (0/1) | |
| model_response | TEXT | Model response text | |
| metrics_json | TEXT | Metrics JSON | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |
| UNIQUE(rollout_id, turn) | | | |

### 9. action (Action Table)
Records information for each action.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Action ID | AUTOINCREMENT |
| turn_id | INTEGER | Turn ID | NOT NULL, REFERENCES turn(id) |
| action_type | TEXT | Action type | |
| tool_name | TEXT | Tool name | |
| tool_args | TEXT | Tool arguments (JSON) | |
| tokens | TEXT | Token list (JSON array) | |
| logprobs | TEXT | Log probability list (JSON array) | |
| num_tokens | INTEGER | Number of tokens | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |

### 10. obs (Observation Table)
Records information for each observation.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Observation ID | AUTOINCREMENT |
| turn_id | INTEGER | Turn ID | NOT NULL, REFERENCES turn(id) |
| obs_type | TEXT | Observation type (screenshot/text/multimodal) | |
| screenshot_uri | TEXT | Screenshot URI | |
| text_content | TEXT | Text content | |
| model_input_json | TEXT | ModelInput JSON | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |

### 11. validation (Validation Table)
Records validation results.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Validation ID | AUTOINCREMENT |
| rollout_id | INTEGER | Rollout ID | NOT NULL, REFERENCES rollout(id) |
| validator_id | INTEGER | Validator ID | REFERENCES validator(id) |
| validation_time | TIMESTAMP | Validation time | NOT NULL, DEFAULT CURRENT_TIMESTAMP |
| validation_query | TEXT | Validation query | |
| expected_result | TEXT | Expected result | |
| actual_result | TEXT | Actual result | |
| success | INTEGER | Success (0/1) | NOT NULL |
| execution_time | REAL | Execution time (seconds) | |
| error_message | TEXT | Error message | |
| details_json | TEXT | Detailed information JSON | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |

### 12. environment (Environment Table)
Records environment information.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | Environment ID | AUTOINCREMENT |
| rollout_id | INTEGER | Rollout ID | NOT NULL, REFERENCES rollout(id) |
| env_type | TEXT | Environment type (android/linux) | NOT NULL |
| status | TEXT | Status (pending/creating/running/terminated/error) | DEFAULT 'pending' |
| gbox_id | TEXT | GBox ID | |
| box_type | TEXT | Box type | |
| creation_time | TIMESTAMP | Creation time | |
| termination_time | TIMESTAMP | Termination time | |
| status_message | TEXT | Status message | |
| error_message | TEXT | Error message | |
| config_json | TEXT | Environment configuration JSON | |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Update time | DEFAULT CURRENT_TIMESTAMP |

### 13. status_history (Status History Table)
Records status change history for all stateful entities.

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| id | INTEGER PRIMARY KEY | History ID | AUTOINCREMENT |
| entity_type | TEXT | Entity type (training/baseline/eval/step/rollout/environment) | NOT NULL |
| entity_id | INTEGER | Entity ID | NOT NULL |
| old_status | TEXT | Old status | |
| new_status | TEXT | New status | NOT NULL |
| progress_percent | REAL | Progress percentage | |
| status_message | TEXT | Status message | |
| metadata_json | TEXT | Metadata JSON | |
| changed_at | TIMESTAMP | Change time | NOT NULL, DEFAULT CURRENT_TIMESTAMP |
| created_at | TIMESTAMP | Creation time | DEFAULT CURRENT_TIMESTAMP |

## Index Design

To improve query performance, create the following indexes:

1. `training(run_name)` - Fast lookup of training sessions
2. `training(status)` - Query training by status
3. `training(status, last_heartbeat)` - Find active training sessions
4. `baseline(training_id)` - Find all baseline evaluations for a training
5. `baseline(status)` - Query baseline evaluations by status
6. `eval(training_id, step)` - Find evaluation for a specific step
7. `eval(status)` - Query evaluations by status
8. `task(task_id)` - Fast lookup of tasks
9. `validator(task_id)` - Find validators for a task
10. `step(training_id, step)` - Fast lookup of steps
11. `step(status)` - Query steps by status
12. `rollout(source_type, step_id)` - Find all rollouts for a step
13. `rollout(source_type, eval_id)` - Find all rollouts for an evaluation
14. `rollout(source_type, baseline_id)` - Find all rollouts for a baseline evaluation
15. `rollout(task_id)` - Find all rollouts for a task
16. `rollout(rollout_id)` - Fast lookup of rollouts
17. `rollout(status)` - Query rollouts by status
18. `turn(rollout_id, turn)` - Fast lookup of turns
19. `action(turn_id)` - Find all actions for a turn
20. `obs(turn_id)` - Find all observations for a turn
21. `validation(rollout_id)` - Find validation results for a rollout
22. `environment(rollout_id)` - Find environment information for a rollout
23. `environment(status)` - Query environments by status
24. `status_history(entity_type, entity_id)` - Find status history for an entity
25. `status_history(entity_type, entity_id, changed_at)` - Status history sorted by time

## Relationship Diagram

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
* (1) ──< (N) status_history  (all stateful entities)
```

**Note**: The rollout table distinguishes sources via the `source_type` field and associates with different entities via the corresponding `step_id`, `eval_id`, or `baseline_id`. Each rollout can only be associated with one source type.

## Status Descriptions

### Training Status
- `pending`: Pending start
- `initializing`: Initializing
- `running`: Running
- `completed`: Completed
- `failed`: Failed
- `paused`: Paused
- `cancelled`: Cancelled

### Baseline/Eval Status
- `pending`: Pending start
- `running`: Running
- `completed`: Completed
- `failed`: Failed
- `cancelled`: Cancelled

### Step Status
- `pending`: Pending start
- `rollout_collecting`: Collecting rollouts
- `rollout_running`: Executing rollouts
- `training`: Training
- `completed`: Completed
- `failed`: Failed

### Rollout Status
- `pending`: Pending start
- `env_creation`: Creating environment
- `agent_init`: Initializing agent
- `running`: Running
- `completed`: Completed
- `failed`: Failed
- `cancelled`: Cancelled

### Environment Status
- `pending`: Pending creation
- `creating`: Creating
- `running`: Running
- `terminated`: Terminated
- `error`: Error

## Progress Tracking Notes

### Training Progress
- `progress_percent`: Calculated based on current step / total steps
- `current_step`: Currently executing step number
- `total_steps`: Total number of steps (obtained from configuration or historical data)
- `current_phase`: Current phase (initialization/rollout/training/evaluation/checkpointing)

### Step Progress
- `progress_percent`: Calculated based on current phase and subtasks
- `rollout_progress`: JSON format, contains `{completed_groups, total_groups, completed_envs, total_envs}`
- `training_progress`: JSON format, contains `{completed_substeps, total_substeps}`
- `current_phase`: Current phase (rollout_collection/rollout_execution/training/checkpointing)

### Rollout Progress
- `progress_percent`: Calculated based on current turn / max turns
- `current_turn`: Currently executing turn number
- `current_phase`: Current phase (env_creation/agent_initialization/task_execution/validation/cleanup)

### Baseline/Eval Progress
- `progress_percent`: Calculated based on completed tasks / total tasks
- `current_task_index`: Current task index
- `total_tasks`: Total number of tasks
- `completed_tasks`: Number of completed tasks
- `current_phase`: Current phase (initialization/rollout/validation/aggregation)
