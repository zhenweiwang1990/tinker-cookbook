"""
Custom rollout function for CUA RL training.

This module provides a custom rollout function that integrates GBoxAgent
with the Tinker RL training framework. It uses TinkerCuaAgent to
allow GBoxAgent to use the current training model for rollout (on-policy RL).
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, cast

import tinker

from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.recipes.cua_rl.agent.cua_env import CUAEnv
from tinker_cookbook.recipes.cua_rl.utils.vision_utils import convert_openai_responses_to_message
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.recipes.cua_rl.core.rollout_logger import RolloutLogger

logger = logging.getLogger(__name__)

# Global variables to store rollout context for trajectory saving
_rollout_output_dir: Optional[str] = None
_rollout_step: Optional[int] = None
_rollout_batch: Optional[int] = None

def set_rollout_output_dir(output_dir: Optional[str]):
    """Set the output directory for saving trajectories."""
    global _rollout_output_dir
    _rollout_output_dir = output_dir

def set_rollout_context(step: Optional[int] = None, batch: Optional[int] = None):
    """Set the current training step and batch for trajectory saving."""
    global _rollout_step, _rollout_batch
    _rollout_step = step
    _rollout_batch = batch


async def _run_single_env_rollout(
    env: CUAEnv,
    env_idx: int,
    total_envs: int,
    model_path: str,
    policy: TokenCompleter,
    trajectory_step: int | None,
    trajectory_batch: int | None,
    group_num: int,
    output_dir: str | None,
    is_eval: bool,
    db_session = None,  # Optional database session
    source_type: str | None = None,  # 'step', 'eval', or 'baseline'
    step_id: int | None = None,
    eval_id: int | None = None,
    baseline_id: int | None = None,
    group_id: int | None = None,  # Database group ID
) -> tuple[Trajectory, float, dict, dict]:
    """
    Run rollout for a single environment.
    
    Returns:
        tuple of (trajectory, final_reward, metrics, summary_dict)
    """
    # Create rollout logger for this environment
    # Generate unique rollout_id as UUID for reliable tracking
    import uuid
    import sys
    rollout_id = str(uuid.uuid4())
    
    # Debug: Log function entry - use print directly to bypass logger
    print(f"[Rollout DEBUG] _run_single_env_rollout called: env_idx={env_idx}, rollout_id={rollout_id}", file=sys.stderr, flush=True)
    logger.info(f"[Rollout DEBUG] _run_single_env_rollout called: env_idx={env_idx}, rollout_id={rollout_id}")
    sys.stdout.flush()
    sys.stderr.flush()
    
    rollout_logger = RolloutLogger(
        rollout_id=rollout_id,
        step=trajectory_step,
        batch=trajectory_batch,
        group=group_num,
        rollout_index=env_idx,
    )
    
    # Debug: Verify logger was created - use print directly
    print(f"[Rollout DEBUG] RolloutLogger created, buffer size={len(rollout_logger.log_buffer)}", file=sys.stderr, flush=True)
    logger.info(f"[Rollout DEBUG] RolloutLogger created, buffer size={len(rollout_logger.log_buffer)}")
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Record rollout start in database if session is available
    # rollout_id is now a UUID generated above
    rollout_recorder = None
    
    # Only record to database if we have a valid source_type
    # source_type=None means this is a training rollout without a step_id (e.g., warmup)
    if db_session is not None and source_type is not None:
        try:
            from tinker_cookbook.recipes.cua_rl.database.rollout_recorder import RolloutRecorder
            
            # Get task info from env
            if not (hasattr(env, 'task') and env.task):
                logger.warning(f"[Rollout DB] Environment has no task attribute")
            else:
                task_id_str = getattr(env.task, 'id', None)
                task_description = getattr(env.task, 'description', '')
                
                if not task_id_str:
                    logger.warning(f"[Rollout DB] Task has no ID attribute: {env.task}")
                else:
                    # Create rollout recorder
                    rollout_recorder = RolloutRecorder(db_session, rollout_id)
                    
                    # Start rollout
                    success = rollout_recorder.start_rollout(
                        task_id_str=task_id_str,
                        task_description=task_description,
                        model_path=model_path,
                        env_type=getattr(env, 'box_type', 'android'),
                        source_type=source_type,
                        step_id=step_id,
                        eval_id=eval_id,
                        baseline_id=baseline_id,
                        batch=trajectory_batch,
                        group_num=group_num,
                        env_index=env_idx,
                        is_eval=is_eval,
                        group_id=group_id,
                        box_type=getattr(env, 'box_type', 'android'),
                    )
                    
                    if success:
                        # Update status to env_creation
                        rollout_recorder.update_status(
                            status="env_creation",
                            current_phase="env_creation",
                        )
                    else:
                        logger.error(f"[Rollout DB] Failed to start rollout recording")
                        rollout_recorder = None
                        
        except Exception as e:
            logger.warning(f"Failed to initialize rollout recorder: {e}", exc_info=True)
            rollout_recorder = None
    
    # Log environment-specific information
    # Debug: Test that log() is working
    import sys
    logger.info(f"[Rollout DEBUG] About to log to rollout_logger, buffer size={len(rollout_logger.log_buffer)}")
    sys.stdout.flush()
    
    rollout_logger.log("-" * 120)
    # Get task name if available
    task_name_str = ""
    if hasattr(env, 'task') and env.task:
        task_name = getattr(env.task, 'name', None) or getattr(env.task, 'id', None)
        if task_name:
            task_name_str = f" | Task: {task_name}"
    rollout_logger.log(f"ENVIRONMENT {env_idx + 1}/{total_envs} ROLLOUT START | Rollout ID: {rollout_id}{task_name_str}")
    rollout_logger.log("-" * 120)
    
    # Debug: Verify logs were added
    logger.info(f"[Rollout DEBUG] After logging, buffer size={len(rollout_logger.log_buffer)}")
    sys.stdout.flush()
    
    env_rollout_start = time.time()
    
    # Run TinkerCuaAgent rollout with current training model
    # Use Tinker's native API (supports multimodal inputs)
    # Get base_model_name from policy if available, otherwise use default
    base_model_name = (
        policy.model_name 
        if hasattr(policy, 'model_name') and policy.model_name is not None
        else "Qwen/Qwen3-VL-30B-A3B-Instruct"  # Default model name
    )
    
    # Force flush before starting rollout
    import sys
    logger.info(f"[Rollout] Starting rollout for env {env_idx}, model_path={model_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Update rollout status to running BEFORE starting execution
    if rollout_recorder is not None:
        rollout_recorder.update_status(
                status="running",
                current_phase="agent_initialization",
            )
    
    # Run rollout with proper error handling
    # Use try-finally to ensure flush() is always called, even if there's an exception
    rollout_result = None
    rollout_error = None
    try:
        rollout_result = await env.run_rollout_with_tinker_model(
            tinker_model_path=model_path,
            tinker_api_key=env.tinker_api_key,  # Use Tinker API key (not gbox_api_key!)
            base_model_name=base_model_name,
            renderer_name=env.renderer.name if hasattr(env.renderer, 'name') else None,
            rollout_logger=rollout_logger,
            rollout_recorder=rollout_recorder,  # Pass rollout_recorder instead of db_session
            rollout_id=rollout_id,  # rollout_id is now a UUID
        )
        
        logger.info(f"[Rollout] Rollout completed for env {env_idx}")
        sys.stdout.flush()
    except Exception as e:
        # Record rollout failure
        rollout_error = e
        logger.error(f"[Rollout] Rollout failed for env {env_idx}: {e}", exc_info=True)
        if rollout_recorder is not None:
            try:
                rollout_recorder.update_status(
                    status="failed",
                    status_message=f"Rollout failed: {str(e)}",
                )
            except Exception as db_error:
                logger.warning(f"[Rollout] Failed to record rollout failure in database: {db_error}")
        # Log error to rollout logger before re-raising
        if rollout_logger:
            rollout_logger.log(f"✗ Rollout failed with exception: {str(e)}", color="RED")
        # Re-raise the exception so it can be handled upstream
        raise
    finally:
        # ALWAYS flush logs, even if there was an exception
        # This ensures we see logs even when rollout fails
        try:
            # Debug: Check if we have logs before flushing - use print directly
            if rollout_logger.log_buffer:
                print(f"[Rollout] Flushing {len(rollout_logger.log_buffer)} log entries for rollout {rollout_logger.rollout_id}", file=sys.stderr, flush=True)
                logger.info(f"[Rollout] Flushing {len(rollout_logger.log_buffer)} log entries for rollout {rollout_logger.rollout_id}")
            else:
                print(f"[Rollout] WARNING: No logs in buffer when flushing rollout {rollout_logger.rollout_id}", file=sys.stderr, flush=True)
                logger.warning(f"[Rollout] WARNING: No logs in buffer when flushing rollout {rollout_logger.rollout_id}")
            sys.stdout.flush()
            sys.stderr.flush()
            rollout_logger.flush()
        except Exception as flush_error:
            # Even if flush fails, log it
            logger.error(f"[Rollout] Failed to flush logs: {flush_error}", exc_info=True)
            sys.stdout.flush()
            sys.stderr.flush()
    
    env_rollout_time = time.time() - env_rollout_start
    rollout_logger.log(f"✓ Environment rollout completed in {env_rollout_time:.2f}s")
    
    # Extract results (rollout_result should never be None here due to exception handling above)
    if rollout_result is None:
        # This should not happen, but handle it gracefully
        logger.error(f"[Rollout] rollout_result is None for env {env_idx}, this indicates a bug")
        rollout_result = {}
    
    task_success = rollout_result.get("task_success", False)
    task_completed = rollout_result.get("task_completed", False)
    num_turns = rollout_result.get("num_turns", 0)
    max_turns = rollout_result.get("max_turns", 15)
    errors = rollout_result.get("errors", [])
    
    # Get ADB validation result from rollout_logger if available
    # (validation is performed in cua_env.py before agent is closed)
    adb_validation_result = None
    validation_screenshot_uri = None
    if hasattr(rollout_logger, 'trajectory_data') and "adb_validation" in rollout_logger.trajectory_data:
        validation_data = rollout_logger.trajectory_data["adb_validation"]
        validation_screenshot_uri = validation_data.get("screenshot_uri")
        from tinker_cookbook.recipes.cua_rl.core.reward import ADBValidationResult
        adb_validation_result = ADBValidationResult(
            command=validation_data.get("command", ""),
            expected_result=validation_data.get("expected_result", ""),
            actual_result=validation_data.get("actual_result", ""),
            success=validation_data.get("success", False),
            execution_time=validation_data.get("execution_time", 0.0),
            validation_query=validation_data.get("validation_query", ""),
        )
    
    # Check if task has validator (required for all tasks)
    task_has_validator = False
    if hasattr(env, 'task') and env.task:
        # Check for _original_task with validator
        if hasattr(env.task, '_original_task') and env.task._original_task:
            if hasattr(env.task._original_task, 'get_validator'):
                validator = env.task._original_task.get_validator()
                if validator and hasattr(validator, 'validate'):
                    task_has_validator = True
        # Check for validation_query
        if not task_has_validator and env.task.validation_query:
            task_has_validator = True
    
    # All tasks must have validator - warn if missing
    if not task_has_validator:
        task_id = getattr(env.task, 'id', 'unknown') if hasattr(env, 'task') and env.task else 'unknown'
        task_name = getattr(env.task, 'name', 'unknown') if hasattr(env, 'task') and env.task else 'unknown'
        logger.warning(
            f"Task {task_id} ({task_name}) does not have a validator! "
            f"Setting task_success=False. All tasks must have a validator."
        )
        # Set task_success to False if no validator
        task_success = False
        adb_validation_result = None  # No validation result available
    
    # Compute reward using comprehensive_reward_function
    validation_start = time.time()
    from tinker_cookbook.recipes.cua_rl.core.reward import (
        comprehensive_reward_function,
        create_rollout_result_from_dict,
        CUARolloutResult,
    )
    
    # Create CUARolloutResult from rollout_result
    # Add validation info if available
    rollout_result_with_validation = rollout_result.copy()
    if adb_validation_result:
        rollout_result_with_validation["validation_passed"] = adb_validation_result.success
        rollout_result_with_validation["validation_details"] = {
            "command": adb_validation_result.command,
            "expected_result": adb_validation_result.expected_result,
            "actual_result": adb_validation_result.actual_result,
            "execution_time": adb_validation_result.execution_time,
            "validation_query": adb_validation_result.validation_query,
        }
    else:
        # No validation result - set to False
        rollout_result_with_validation["validation_passed"] = False
        rollout_result_with_validation["validation_details"] = None
    
    # Get task_id from env if available
    task_id = getattr(env.task, 'id', 'unknown') if hasattr(env, 'task') and env.task else 'unknown'
    
    # Create CUARolloutResult
    cua_result = create_rollout_result_from_dict(
        rollout_result_with_validation,
        task_id=task_id,
    )
    
    # Compute reward using comprehensive_reward_function
    reward = comprehensive_reward_function(cua_result, task=env.task if hasattr(env, 'task') else None)
    
    validation_time = time.time() - validation_start
    validation_method = "comprehensive_reward_function"
    
    # Set summary in rollout logger (after reward calculation)
    # All tasks must have validator - use validation_passed as the actual task_success
    # If no validator, task_success is already set to False with warning logged
    actual_task_success = (
        adb_validation_result.success if adb_validation_result else False
    )
    # Save original task_success (agent's self-reported success from finish tool)
    original_task_success = task_success
    # Save validation_passed (validator's result)
    validation_passed = adb_validation_result.success if adb_validation_result else False
    
    rollout_logger.set_summary({
        "task_success": actual_task_success,
        "task_completed": task_completed,
        "num_turns": num_turns,
        "rollout_time": env_rollout_time,
        "reward": reward,
        "validation_passed": validation_passed,
        "agent_reported_success": original_task_success,
    })
    
    # Log ADB validation details in table format (always shown, even if no validation)
    # This should be shown BEFORE the rollout summary
    rollout_logger.log_rollout_completion()
    
    # Log rollout summary in compact table format
    rollout_logger.log_rollout_summary_table(
        validation_passed=validation_passed,
        task_completed=task_completed,
        agent_reported_success=original_task_success,
        num_turns=num_turns,
        total_rollout_time=env_rollout_time,
        reward=reward,
        validation_method=validation_method,
        validation_time=validation_time,
    )
    
    # Get temperature from policy if available
    temperature = None
    if hasattr(policy, 'temperature'):
        temperature = policy.temperature
    elif hasattr(policy, 'sampling_params') and hasattr(policy.sampling_params, 'temperature'):
        temperature = policy.sampling_params.temperature
    
    # Get three comparison values for summary
    validation_passed = adb_validation_result.success if adb_validation_result else False
    original_task_success = task_success  # Agent's self-reported success
    
    # Create summary dict
    summary_dict = {
        "rollout_id": rollout_id,
        "task_description": env.task_description[:50] + "..." if len(env.task_description) > 50 else env.task_description,
        "task_completed": task_completed,
        "task_success": actual_task_success,  # Use validation_passed as ground truth
        "num_turns": num_turns,
        "reward": reward,
        "rollout_time": env_rollout_time,
        "temperature": temperature,
        "validation_passed": validation_passed,
        "agent_reported_success": original_task_success,
    }
    
    # Record rollout completion in database
    if rollout_recorder is not None:
        try:
            # Record validation if available
            if adb_validation_result:
                rollout_recorder.record_validation(
                    success=adb_validation_result.success,
                    validation_query=adb_validation_result.validation_query,
                    expected_result=adb_validation_result.expected_result,
                    actual_result=adb_validation_result.actual_result,
                    execution_time=adb_validation_result.execution_time,
                    error_message=None,
                    screenshot_uri=validation_screenshot_uri,
                    details_json={
                        "command": adb_validation_result.command,
                        "screenshot_uri": validation_screenshot_uri,
                    },
                )
            
            # Extract metrics from rollout_result
            num_total_actions = rollout_result.get('num_total_actions', 0)
            consecutive_repeated_actions = rollout_result.get('consecutive_repeated_actions', 0)
            parse_errors = rollout_result.get('parse_errors', 0)
            tool_name_errors = rollout_result.get('tool_name_errors', 0)
            tool_arg_errors = rollout_result.get('tool_arg_errors', 0)
            runtime_errors = rollout_result.get('runtime_errors', 0)
            ran_out_of_turns = rollout_result.get('ran_out_of_turns', False)
            attempted_completion = rollout_result.get('attempted_completion', False)
            turn_first_success = rollout_result.get('turn_first_success', -1)
            turn_task_completed = rollout_result.get('turn_task_completed', -1)
            
            # Record rollout completion
            rollout_recorder.complete_rollout(
                task_completed=task_completed,
                task_success=actual_task_success,
                agent_reported_success=original_task_success,
                validation_passed=validation_passed,
                num_turns=num_turns,
                reward=reward,
                rollout_time=env_rollout_time,
                errors=errors,
                summary_json=summary_dict,
                num_total_actions=num_total_actions,
                consecutive_repeated_actions=consecutive_repeated_actions,
                parse_errors=parse_errors,
                tool_name_errors=tool_name_errors,
                tool_arg_errors=tool_arg_errors,
                runtime_errors=runtime_errors,
                ran_out_of_turns=ran_out_of_turns,
                attempted_completion=attempted_completion,
                turn_first_success=turn_first_success,
                turn_task_completed=turn_task_completed,
                max_turns=max_turns,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"Failed to record rollout completion in database: {e}")
    
    # Save trajectory BEFORE flush (so logs are still in buffer)
    # Note: flush() is now called in the finally block above, so we don't need to call it here
    if output_dir:
        from pathlib import Path
        output_path = Path(output_dir)
        rollout_logger.save_trajectory(
            base_dir=output_path,
            step=trajectory_step,
            batch=trajectory_batch,
            group=env_idx,
            is_eval=is_eval,
        )
    
    # Extract full trajectory from env (token-level training)
    # Trajectory data is saved in env._trajectory_turns before agent cleanup
    trajectory_turns = getattr(env, '_trajectory_turns', [])
    
    if not trajectory_turns:
        # This can happen if:
        # 1. Model inference timed out on the first turn (before any trajectory data was saved)
        # 2. Model inference failed with exception on the first turn
        # 3. Rollout was cancelled/interrupted before completing any turn
        # 4. Some other early failure prevented any turns from completing
        logger.warning(
            f"[Rollout] No trajectory data available from agent (rollout_id={rollout_id}). "
            f"This indicates the rollout did not produce any turns or trajectory data was not saved. "
            f"Task: {env.task_description[:100]}... "
            f"Num turns in result: {num_turns}, Task completed: {task_completed}, "
            f"Result message: {rollout_result.get('result_message', 'N/A')[:200]}"
        )
        # Log additional debug info to help diagnose the issue
        if hasattr(env, '_agent') and env._agent:
            agent_result_msg = getattr(env._agent, 'result_message', 'N/A')
            logger.warning(
                f"[Rollout] Agent result_message: {agent_result_msg[:200] if agent_result_msg else 'N/A'}"
            )
        # Return empty trajectory instead of raising error
        # This allows the training loop to continue with other rollouts
        # Create empty trajectory with no transitions
        empty_trajectory = Trajectory(
            transitions=[],
            final_ob=tinker.ModelInput.empty(),
        )
        
        # Create metrics dict
        metrics = {
            "task_success": float(actual_task_success),
            "task_completed": float(task_completed),
            "num_turns": num_turns,
        }
        
        # Return 4-tuple to match expected format: (trajectory, final_reward, metrics, summary_dict)
        return empty_trajectory, reward, metrics, summary_dict
    
    # Save trajectory_data_json to database
    # Include both training data (trajectory_turns) and detailed execution info (rollout_logger.trajectory_data)
    if db_session is not None and rollout_id is not None:
        try:
            import json
            from tinker_cookbook.recipes.cua_rl.database.database_dao import update_rollout
            from tinker_cookbook.recipes.cua_rl.database.database_rollout import get_rollout_by_rollout_id
            
            # Convert trajectory_turns to JSON-serializable format (for training)
            trajectory_data_list = []
            for turn_num, observation, action_tokens, action_logprobs in trajectory_turns:
                # Convert observation (ModelInput) to dict
                obs_dict = None
                if observation:
                    try:
                        if hasattr(observation, 'to_dict'):
                            obs_dict = observation.to_dict()
                        elif isinstance(observation, dict):
                            obs_dict = observation
                        else:
                            obs_dict = str(observation)
                    except Exception as e:
                        logger.warning(f"Failed to serialize observation for turn {turn_num}: {e}")
                        obs_dict = str(observation)
                
                trajectory_data_list.append({
                    "turn": turn_num,
                    "observation": obs_dict,
                    "action_tokens": action_tokens,
                    "action_logprobs": action_logprobs,
                })
            
            # Combine training data with detailed execution info from rollout_logger
            combined_trajectory_data = {
                "training_data": trajectory_data_list,  # For training (token-level)
                "execution_details": None,  # Detailed execution info from RolloutLogger
            }
            
            # Add RolloutLogger's trajectory_data if available (contains turns, action_results, tool_executions, etc.)
            if hasattr(rollout_logger, 'trajectory_data') and rollout_logger.trajectory_data:
                try:
                    # Convert rollout_logger.trajectory_data to JSON-serializable format
                    execution_details = json.loads(json.dumps(rollout_logger.trajectory_data, default=str))
                    combined_trajectory_data["execution_details"] = execution_details
                except Exception as e:
                    logger.warning(f"Failed to serialize rollout_logger.trajectory_data: {e}")
            
            trajectory_data_json = json.dumps(combined_trajectory_data, default=str)
            
            # Update rollout with trajectory_data_json
            db_rollout = get_rollout_by_rollout_id(db_session, rollout_id)
            if db_rollout:
                update_rollout(
                    db_session,
                    db_rollout.id,
                    trajectory_data_json=trajectory_data_json,
                )
                db_session.commit()
                logger.debug(f"Saved trajectory_data_json (with execution details) to database for rollout {rollout_id}")
        except Exception as e:
            logger.warning(f"Failed to save trajectory_data_json to database: {e}")
            if db_session:
                db_session.rollback()
    
    transitions: list[Transition] = []
    
    # We have token-level trajectory data from the agent
    # Distribute reward across turns (uniform distribution for now)
    # In the future, we could use more sophisticated reward shaping
    num_trajectory_turns = len(trajectory_turns)
    
    # Distribute reward evenly across all turns
    # Alternatively, we could give all reward to the last turn, or use reward shaping
    per_turn_reward = reward / num_trajectory_turns if num_trajectory_turns > 0 else 0.0
    
    for turn_idx, (turn_num, observation, action_tokens, action_logprobs) in enumerate(trajectory_turns):
        # Create TokensWithLogprobs for this action
        # Ensure logprobs match tokens length: use empty list if tokens are empty, otherwise use provided logprobs
        if len(action_tokens) == 0:
            final_logprobs: list[float] | None = []
        elif action_logprobs and len(action_logprobs) == len(action_tokens):
            final_logprobs = action_logprobs
        else:
            # Mismatch: raise error since we require valid trajectory data
            raise ValueError(
                f"[Rollout] Turn {turn_num}: tokens length ({len(action_tokens)}) != logprobs length "
                f"({len(action_logprobs) if action_logprobs else 0}). "
                f"Token-level training requires matching tokens and logprobs."
            )
        
        action = TokensWithLogprobs(
            tokens=action_tokens,
            maybe_logprobs=final_logprobs
        )
        
        # Determine if this is the last turn
        is_last_turn = (turn_idx == num_trajectory_turns - 1)
        
        # Create transition
        transition = Transition(
            ob=observation,
            ac=action,
            reward=per_turn_reward,
            episode_done=is_last_turn,
            metrics={
                "task_success": float(task_success) if is_last_turn else 0.0,
                "task_completed": float(task_completed) if is_last_turn else 0.0,
                "turn": turn_num,
                "rollout_time": env_rollout_time if is_last_turn else 0.0,
            },
        )
        transitions.append(transition)
    
    # Create trajectory
    trajectory = Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.empty(),
    )
    
    metrics = {
        "task_success": float(task_success),
        "task_completed": float(task_completed),
        "num_turns": num_turns,
    }
    
    return trajectory, 0.0, metrics, summary_dict


async def do_cua_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    model_path: str | None = None,
    step: int | None = None,
    batch: int | None = None,
    group: int | None = None,
    output_dir: str | None = None,
    is_eval: bool = False,
    db_session = None,  # Optional database session for recording
    step_id: int | None = None,  # Database step ID
    eval_id: int | None = None,  # Database eval ID
    baseline_id: int | None = None,  # Database baseline ID
) -> TrajectoryGroup:
    # Debug: Log function entry - use print directly to bypass logger
    import sys
    print(f"[Rollout DEBUG] do_cua_group_rollout called: step={step}, batch={batch}, group={group}, is_eval={is_eval}", file=sys.stderr, flush=True)
    logger.info(f"[Rollout DEBUG] do_cua_group_rollout called: step={step}, batch={batch}, group={group}, is_eval={is_eval}")
    sys.stdout.flush()
    sys.stderr.flush()
    """
    Custom rollout function for CUA environments.
    
    This function:
    1. Creates environments from the builder
    2. Gets the current training model's checkpoint path from the policy (or uses provided model_path)
    3. Uses Tinker's native API to run GBoxAgent rollout with the training model
    4. Converts the rollout results to TrajectoryGroup format
    
    The rollout model dynamically updates as training progresses (on-policy RL).
    
    **Parallel Execution**: All environments in a group are executed in parallel using
    asyncio.gather, allowing multiple GBox instances to run simultaneously. This significantly
    speeds up rollout when you have multiple environments (controlled by group_size parameter).
    
    Args:
        env_group_builder: Builder for creating environments
        policy: TokenCompleter (should be TinkerTokenCompleter with the training model)
        model_path: Optional model path (tinker://...) to use for OpenAI-compatible API.
                    If not provided, will try to extract from policy's SamplingClient.
        
    Returns:
        TrajectoryGroup with rollout results
    """
    # Extract sampling client from policy
    if not isinstance(policy, TinkerTokenCompleter):
        raise ValueError(
            f"Expected TinkerTokenCompleter, got {type(policy)}. "
            "CUA RL requires using Tinker's model for rollout (on-policy RL)."
        )
    
    sampling_client = policy.sampling_client
    
    # If model_path was provided, validate and use it
    if model_path is not None:
        if not (isinstance(model_path, str) and model_path.startswith('tinker://')):
            raise ValueError(f"Invalid model_path format: {model_path}. Expected tinker://... format.")
    else:
        # Try to extract model_path from sampling client
        # The model_path is a tinker://... path that can be used with OpenAI-compatible API
        logger.warning("model_path not provided, attempting to extract from SamplingClient...")
        
        # Try multiple ways to access the model_path
        # Method 1: Direct attribute (most common for SamplingClient created with model_path)
        if hasattr(sampling_client, 'model_path') and sampling_client.model_path is not None:
            model_path = sampling_client.model_path
        # Method 2: Check holder's model_path
        elif hasattr(sampling_client, 'holder') and hasattr(sampling_client.holder, 'model_path'):
            model_path = sampling_client.holder.model_path
        # Method 3: Check holder's _model_path (private attribute)
        elif hasattr(sampling_client, 'holder') and hasattr(sampling_client.holder, '_model_path'):
            model_path = sampling_client.holder._model_path
        
        if model_path is None or not (isinstance(model_path, str) and model_path.startswith('tinker://')):
            # If we can't get model_path, log detailed debug info and raise an error
            debug_info = {
                "sampling_client_attrs": dir(sampling_client),
                "has_holder": hasattr(sampling_client, 'holder'),
            }
            if hasattr(sampling_client, 'holder'):
                debug_info["holder_attrs"] = dir(sampling_client.holder)
                # Try to get any path-like attributes
                holder = sampling_client.holder
                for attr in dir(holder):
                    if 'path' in attr.lower() or 'model' in attr.lower():
                        try:
                            val = getattr(holder, attr)
                            if isinstance(val, str):
                                debug_info[f"holder.{attr}"] = val
                        except:
                            pass
            
            logger.error(
                f"Cannot get model_path from SamplingClient (got: {model_path}). "
                f"Debug info: {debug_info}"
            )
            raise ValueError(
                "Cannot get model_path from SamplingClient. "
                "Please ensure the training loop always saves checkpoints with explicit paths "
                "or pass model_path explicitly to do_cua_group_rollout(). "
                f"Current model_path value: {model_path}"
            )
    
    # Create environments
    rollout_start_time = time.time()
    
    env_creation_start = time.time()
    envs = await env_group_builder.make_envs()
    env_creation_time = time.time() - env_creation_start
    
    # Validate all environments are CUAEnv
    for env_idx, env in enumerate(envs):
        if not isinstance(env, CUAEnv):
            raise ValueError(f"Expected CUAEnv, got {type(env)}")
    
    # Use provided step/batch or fall back to global variables
    trajectory_step = step if step is not None else _rollout_step
    trajectory_batch = batch if batch is not None else _rollout_batch
    trajectory_output_dir = output_dir or _rollout_output_dir or os.getenv("CUA_ROLLOUT_OUTPUT_DIR")
    
    # Log global setup information
    logger.info("=" * 120)
    logger.info(f"ROLLOUT START, with environment count: {len(envs)}")
    logger.info(f"Model path: {model_path}")
    logger.info("=" * 120)
    
    # Determine source_type and source IDs for database
    # Try to get from global context if not provided
    from tinker_cookbook.recipes.cua_rl.database.database_context import get_database_session, get_training_id
    global_db_session = get_database_session()
    if db_session is None and global_db_session is not None:
        db_session = global_db_session
    
    source_type = None
    step_id_db = None
    eval_id_db = None
    baseline_id_db = None
    
    # Determine source_type and IDs - this should work even if db_session is None
    # (we'll check db_session later when actually recording)
    # Priority: step_id > eval_id > baseline_id > context lookup
    if step_id is not None:
        source_type = "step"
        step_id_db = step_id
        logger.debug(f"[Rollout DB] Determined source_type='step' from step_id={step_id}")
    elif eval_id is not None:
        source_type = "eval"
        eval_id_db = eval_id
        logger.debug(f"[Rollout DB] Determined source_type='eval' from eval_id={eval_id}")
    elif baseline_id is not None:
        source_type = "baseline"
        baseline_id_db = baseline_id
        logger.debug(f"[Rollout DB] Determined source_type='baseline' from baseline_id={baseline_id}")
    elif is_eval:
        # This is an evaluation - try to get IDs from context
        # First try baseline (baseline evaluations happen before regular evaluations)
        if step is None:
            from tinker_cookbook.recipes.cua_rl.database.database_context import get_baseline_id
            baseline_id_from_context = get_baseline_id()
            if baseline_id_from_context:
                source_type = "baseline"
                baseline_id_db = baseline_id_from_context
                logger.info(f"[Rollout DB] Using baseline_id={baseline_id_db} from context for baseline rollout")
            else:
                # Try eval_id as fallback
                from tinker_cookbook.recipes.cua_rl.database.database_context import get_eval_id
                eval_id_from_context = get_eval_id()
                if eval_id_from_context:
                    source_type = "eval"
                    eval_id_db = eval_id_from_context
                    logger.info(f"[Rollout DB] Using eval_id={eval_id_db} from context for evaluation rollout")
                else:
                    logger.warning(f"[Rollout DB] is_eval=True, step=None, but both baseline_id and eval_id are None in context. Rollout will use 'unknown' as source_type.")
        else:
            # Regular evaluation - try to get eval_id from context
            from tinker_cookbook.recipes.cua_rl.database.database_context import get_eval_id
            eval_id_from_context = get_eval_id()
            if eval_id_from_context:
                source_type = "eval"
                eval_id_db = eval_id_from_context
                logger.info(f"[Rollout DB] Using eval_id={eval_id_db} from context for evaluation rollout")
            else:
                logger.warning(f"[Rollout DB] is_eval=True but eval_id is None and not found in context. Rollout will use 'unknown' as source_type.")
    else:
        # This is a training rollout but step_id is None
        # This can happen during warmup or if step hasn't been created yet
        # Skip database recording for this rollout
        logger.warning(f"[Rollout DB] Training rollout (is_eval=False) but step_id is None. Skipping database recording for this rollout.")
        source_type = None  # Explicitly set to None to skip DB recording
    
    # Now check if we have db_session for recording
    if db_session is None:
        db_session = global_db_session
    
    # Create or get group record at the start of group rollout
    group_id_db = None
    if db_session is not None and source_type and group is not None:
        try:
            from tinker_cookbook.recipes.cua_rl.database.database_dao import get_or_create_group, update_group
            group_obj = get_or_create_group(
                session=db_session,
                source_type=source_type,
                group_num=group,
                step_id=step_id_db,
                eval_id=eval_id_db,
                baseline_id=baseline_id_db,
                batch=trajectory_batch,
                status="running",
                current_phase="rollout",
            )
            group_id_db = group_obj.id
            db_session.commit()
            logger.info(f"[Rollout DB] Created/retrieved group {group} (group_id={group_id_db}) for {source_type}")
        except Exception as e:
            logger.warning(f"[Rollout DB] Failed to create/get group record: {e}")
            if db_session:
                db_session.rollback()
    
    # Run rollouts in parallel using asyncio.gather
    # This allows multiple GBox environments to execute simultaneously
    # IMPORTANT: Each environment must use its own database session to avoid
    # concurrent access issues (SQLAlchemy sessions are not thread-safe)
    rollout_tasks = []
    for env_idx, env in enumerate(envs):
        # Use provided group number if available
        # group should represent the group index (e.g., 0, 1, 2, ... for different groups in evaluation)
        # env_idx represents the environment index within this group (0, 1, 2, ... for environments in the same group)
        # CRITICAL: group_num is the GROUP number, env_idx is the ENV index within that group
        # They should NOT be confused - all envs in the same group share the same group_num
        if group is not None:
            group_num = group
        else:
            # Fallback: if group is None, we can't determine the correct group number
            # This should only happen if the caller didn't provide group number
            # Log a warning and use 0 as fallback (but this indicates a bug)
            logger.warning(
                f"[Rollout] group parameter is None for {len(envs)} environments. "
                f"Using 0 as fallback, but this may cause incorrect group numbering. "
                f"All environments will be recorded as group 0."
            )
            group_num = 0
        
        # Create a separate session for each environment to avoid concurrent access issues
        # SQLAlchemy sessions are not thread-safe and sharing a session across
        # parallel async tasks can cause data corruption and ID mismatches
        # CRITICAL: Each environment MUST have its own session to prevent data corruption
        env_db_session = None
        session_created = False
        if db_session is not None:
            from tinker_cookbook.recipes.cua_rl.database.database import get_session_direct
            try:
                env_db_session = get_session_direct()
                session_created = True
                logger.debug(f"[Rollout] Created separate DB session for env {env_idx}")
            except Exception as e:
                # CRITICAL FIX: DO NOT fall back to shared session - this causes data corruption
                # If we can't create a separate session, we must fail the rollout
                logger.error(
                    f"[Rollout] CRITICAL: Failed to create separate DB session for env {env_idx}: {e}. "
                    f"Cannot proceed with database recording to avoid data corruption. "
                    f"Rollout will continue without database recording."
                )
                # Set env_db_session to None to disable database recording for this rollout
                env_db_session = None
                session_created = False
        
        # Create a wrapper task that ensures session cleanup
        # Capture variables in closure to avoid issues with loop variable capture
        def make_rollout_task(env, env_idx, env_db_session, session_created_flag):
            async def run_with_session_cleanup():
                # Debug: Log task start - use print directly to bypass logger
                print(f"[Rollout DEBUG] Rollout task started for env {env_idx}", file=sys.stderr, flush=True)
                logger.info(f"[Rollout DEBUG] Rollout task started for env {env_idx}")
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    return await _run_single_env_rollout(
                        env=env,
                        env_idx=env_idx,
                        total_envs=len(envs),
                        model_path=model_path,
                        policy=policy,
                        trajectory_step=trajectory_step,
                        trajectory_batch=trajectory_batch,
                        group_num=group_num,
                        output_dir=trajectory_output_dir,
                        is_eval=is_eval,
                        db_session=env_db_session,  # Use separate session for each environment
                        source_type=source_type,
                        step_id=step_id_db,
                        eval_id=eval_id_db,
                        baseline_id=baseline_id_db,
                        group_id=group_id_db,  # Pass group_id so rollout belongs to the group
                    )
                finally:
                    # Close the session if it was created specifically for this environment
                    if session_created_flag and env_db_session is not None:
                        try:
                            env_db_session.close()
                            logger.debug(f"[Rollout] Closed DB session for env {env_idx}")
                        except Exception as e:
                            logger.warning(f"[Rollout] Error closing DB session for env {env_idx}: {e}")
            return run_with_session_cleanup
        
        task = make_rollout_task(env, env_idx, env_db_session, session_created)()
        rollout_tasks.append(task)
    
    # Execute all rollouts in parallel
    print(f"[Rollout DEBUG] About to execute {len(rollout_tasks)} rollout tasks in parallel", file=sys.stderr, flush=True)
    logger.info(f"[Rollout DEBUG] About to execute {len(rollout_tasks)} rollout tasks in parallel")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        results = await asyncio.gather(*rollout_tasks)
        print(f"[Rollout DEBUG] All {len(results)} rollouts completed", file=sys.stderr, flush=True)
        logger.info(f"[Rollout DEBUG] All {len(results)} rollouts completed")
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as e:
        # If any rollout fails, update group status to failed
        if db_session is not None and group_id_db is not None:
            try:
                from tinker_cookbook.recipes.cua_rl.database.database_dao import update_group
                update_group(
                    db_session,
                    group_id_db,
                    status="failed",
                    error_message=f"Group rollout failed: {str(e)}",
                    end_time=datetime.utcnow(),
                )
                db_session.commit()
            except Exception as db_error:
                logger.warning(f"[Rollout DB] Failed to update group status to failed: {db_error}")
                if db_session:
                    db_session.rollback()
        raise
    
    # Update group status after all rollouts complete
    if db_session is not None and group_id_db is not None:
        try:
            from tinker_cookbook.recipes.cua_rl.database.database_dao import get_group, update_group
            from datetime import datetime
            
            # Count successful and failed rollouts
            successful_rollouts = 0
            failed_rollouts = 0
            total_rewards = []
            
            for trajectory, final_reward, metrics, summary_dict in results:
                if summary_dict.get("task_success", False):
                    successful_rollouts += 1
                else:
                    failed_rollouts += 1
                if final_reward is not None:
                    total_rewards.append(final_reward)
            
            # Calculate metrics
            reward_mean = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
            if len(total_rewards) > 1:
                import statistics
                reward_std = statistics.stdev(total_rewards)
            else:
                reward_std = 0.0
            
            # Update group with completion status
            group = get_group(db_session, group_id_db)
            if group:
                completed_rollouts = (group.completed_rollouts or 0) + len(results)
                
                # If all expected rollouts are completed, mark group as completed
                if completed_rollouts >= (group.num_rollouts or len(results)):
                    update_group(
                        db_session,
                        group_id_db,
                        status="completed",
                        completed_rollouts=completed_rollouts,
                        success_count=successful_rollouts,
                        reward_mean=reward_mean,
                        reward_std=reward_std,
                        progress_percent=100.0,
                        end_time=datetime.utcnow(),
                    )
                else:
                    # Update progress but keep status as running
                    update_group(
                        db_session,
                        group_id_db,
                        completed_rollouts=completed_rollouts,
                        success_count=successful_rollouts,
                        reward_mean=reward_mean,
                        reward_std=reward_std,
                        progress_percent=(completed_rollouts / (group.num_rollouts or len(results))) * 100.0,
                    )
                db_session.commit()
        except Exception as e:
            logger.warning(f"[Rollout DB] Failed to update group completion status: {e}")
            if db_session:
                db_session.rollback()
    
    # Unpack results (results are already sorted by env_idx)
    trajectories = []
    final_rewards = []
    metrics_list = []
    rollout_summaries = []
    
    for trajectory, final_reward, metrics, summary_dict in results:
        trajectories.append(trajectory)
        final_rewards.append(final_reward)
        metrics_list.append(metrics)
        rollout_summaries.append(summary_dict)
    
    total_rollout_time = time.time() - rollout_start_time
    
    # Print group-level summary table (skip for evaluation mode)
    if not is_eval:
        logger.info("")
        logger.info("=" * 57)
        logger.info("ROLLOUT GROUP SUMMARY")
        logger.info("=" * 57)
    
    # Print table header (without Task Description since all tasks in a group are the same)
    # Column widths: #(4) + Completed(10) + Success(8) + Turns(6) + Reward(7) + Time(10) + Temp(6) + spaces(6) = 57
    header = f"{'#':<4} {'Completed':<10} {'Success':<8} {'Turns':<6} {'Reward':<7} {'Time (s)':<10} {'Temp':<6}"
    logger.info(header)
    logger.info("-" * 57)
    
    # Print each rollout's summary
    enable_color = os.environ.get("NO_COLOR", "").strip() == ""
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    
    for i, summary in enumerate(rollout_summaries):
        completed_str = "✓" if summary["task_completed"] else "✗"
        success_str = "✓" if summary["task_success"] else "✗"
        # Apply colors to success status
        if enable_color:
            if summary["task_success"]:
                success_str_colored = f"{GREEN}{success_str}{RESET}"
            else:
                success_str_colored = f"{RED}{success_str}{RESET}"
        else:
            success_str_colored = success_str
        
        temp_str = f"{summary['temperature']:.2f}" if summary['temperature'] is not None else "N/A"
        
        # Build row with proper alignment
        # For colored strings, we need to pad after the RESET code to ensure proper alignment
        # The display width of success_str_colored is 1 (just ✓ or ✗), but the string length includes ANSI codes
        if enable_color:
            # Pad the colored string to 8 characters display width (ANSI codes don't count)
            success_padded = success_str_colored + " " * (8 - 1)  # 8 display width - 1 char
        else:
            success_padded = f"{success_str_colored:<8}"
        
        row = (
            f"{i+1:<4} "
            f"{completed_str:<10} "
            f"{success_padded} "
            f"{summary['num_turns']:<6} "
            f"{summary['reward']:<7.1f} "
            f"{summary['rollout_time']:<10.2f} "
            f"{temp_str:<6}"
        )
        logger.info(row)
    
    # Print summary statistics (skip for evaluation mode)
    if not is_eval:
        logger.info("-" * 72)
        logger.info(f"Total environments: {len(envs)}")
        logger.info(f"Total rollout time: {total_rollout_time:.2f}s")
        logger.info(f"Average time per environment: {total_rollout_time / len(envs):.2f}s")
        # Count successes based on validator (ground truth)
        validation_success_count = sum(1 for s in rollout_summaries if s.get("validation_passed", False))
        logger.info(f"Validator success: {validation_success_count}/{len(envs)} ({100 * validation_success_count / len(envs):.1f}%)")
        # Count agent reported successes
        agent_success_count = sum(1 for s in rollout_summaries if s.get("agent_reported_success", False))
        logger.info(f"Agent reported success: {agent_success_count}/{len(envs)} ({100 * agent_success_count / len(envs):.1f}%)")
        total_turns = sum(m.get("num_turns", 0) for m in metrics_list)
        logger.info(f"Total turns across all environments: {total_turns}")
        logger.info(f"Average turns per environment: {total_turns / len(envs):.1f}")
        logger.info("=" * 72)
        
        # Check for learning signal: if all rewards are the same, there's no learning signal
        # Extract rewards from trajectories (each trajectory has transitions with rewards)
        if len(trajectories) > 0:
            # Get rewards from all transitions in all trajectories
            all_rewards = []
            for traj in trajectories:
                for transition in traj.transitions:
                    all_rewards.append(transition.reward)
            
            if len(all_rewards) > 0:
                unique_rewards = set(all_rewards)
                if len(unique_rewards) == 1:
                    # All rewards are the same - no learning signal
                    reward_value = list(unique_rewards)[0]
                    YELLOW = "\033[93m"
                    BOLD = "\033[1m"
                    RESET = "\033[0m"
                    if enable_color:
                        warning_msg = (
                            f"{YELLOW}{BOLD}{'=' * 80}{RESET}\n"
                            f"{YELLOW}{BOLD}⚠️  WARNING: NO LEARNING SIGNAL IN THIS GROUP ⚠️{RESET}\n"
                            f"{YELLOW}{BOLD}{'=' * 80}{RESET}\n"
                            f"{YELLOW}All {len(trajectories)} environments in this group have the same reward: {reward_value:.1f}{RESET}\n"
                            f"{YELLOW}This means there is no learning signal for policy gradient updates.{RESET}\n"
                            f"{YELLOW}Consider: increasing group_size, using different tasks, or adjusting the reward structure.{RESET}\n"
                            f"{YELLOW}{BOLD}{'=' * 80}{RESET}"
                        )
                    else:
                        warning_msg = (
                            f"{'=' * 80}\n"
                            f"⚠️  WARNING: NO LEARNING SIGNAL IN THIS GROUP ⚠️\n"
                            f"{'=' * 80}\n"
                            f"All {len(trajectories)} environments in this group have the same reward: {reward_value:.1f}\n"
                            f"This means there is no learning signal for policy gradient updates.\n"
                            f"Consider: increasing group_size, using different tasks, or adjusting the reward structure.\n"
                            f"{'=' * 80}"
                        )
                    logger.warning(warning_msg)
    
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=final_rewards,
        metrics_G=metrics_list,
    )

