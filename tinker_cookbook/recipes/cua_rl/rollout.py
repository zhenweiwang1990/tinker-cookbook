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
from tinker_cookbook.recipes.cua_rl.cua_env import CUAEnv
from tinker_cookbook.recipes.cua_rl.vision_utils import convert_openai_responses_to_message
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.recipes.cua_rl.rollout_logger import RolloutLogger

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
) -> tuple[Trajectory, float, dict, dict]:
    """
    Run rollout for a single environment.
    
    Returns:
        tuple of (trajectory, final_reward, metrics, summary_dict)
    """
    # Create rollout logger for this environment
    rollout_id = f"env_{env_idx}"
    rollout_logger = RolloutLogger(
        rollout_id=rollout_id,
        step=trajectory_step,
        batch=trajectory_batch,
        group=group_num,
        rollout_index=env_idx,
    )
    
    # Log environment-specific information
    rollout_logger.log("-" * 120)
    rollout_logger.log(f"ENVIRONMENT {env_idx + 1}/{total_envs} ROLLOUT START")
    rollout_logger.log("-" * 120)
    
    env_rollout_start = time.time()
    
    # Run TinkerCuaAgent rollout with current training model
    # Use Tinker's native API (supports multimodal inputs)
    # Get base_model_name from policy if available, otherwise use default
    base_model_name = (
        policy.model_name 
        if hasattr(policy, 'model_name') and policy.model_name is not None
        else "Qwen/Qwen3-VL-30B-A3B-Instruct"  # Default model name
    )
    
    rollout_result = await env.run_rollout_with_tinker_model(
        tinker_model_path=model_path,
        tinker_api_key=env.tinker_api_key,  # Use Tinker API key (not gbox_api_key!)
        base_model_name=base_model_name,
        renderer_name=env.renderer.name if hasattr(env.renderer, 'name') else None,
        rollout_logger=rollout_logger,
    )
    
    env_rollout_time = time.time() - env_rollout_start
    rollout_logger.log(f"✓ Environment rollout completed in {env_rollout_time:.2f}s")
    
    # Extract results
    task_success = rollout_result.get("task_success", False)
    task_completed = rollout_result.get("task_completed", False)
    num_turns = rollout_result.get("num_turns", 0)
    max_turns = rollout_result.get("max_turns", 15)
    errors = rollout_result.get("errors", [])
    
    # Get ADB validation result from rollout_logger if available
    # (validation is performed in cua_env.py before agent is closed)
    adb_validation_result = None
    if hasattr(rollout_logger, 'trajectory_data') and "adb_validation" in rollout_logger.trajectory_data:
        validation_data = rollout_logger.trajectory_data["adb_validation"]
        from tinker_cookbook.recipes.cua_rl.reward import ADBValidationResult
        adb_validation_result = ADBValidationResult(
            command=validation_data.get("command", ""),
            expected_result=validation_data.get("expected_result", ""),
            actual_result=validation_data.get("actual_result", ""),
            success=validation_data.get("success", False),
            execution_time=validation_data.get("execution_time", 0.0),
            validation_query=validation_data.get("validation_query", ""),
        )
    
    # Compute reward using comprehensive_reward_function
    validation_start = time.time()
    from tinker_cookbook.recipes.cua_rl.reward import (
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
    rollout_logger.set_summary({
        "task_success": task_success,
        "task_completed": task_completed,
        "num_turns": num_turns,
        "rollout_time": env_rollout_time,
        "reward": reward,
    })
    
    # Log ADB validation details in table format (always shown, even if no validation)
    # This should be shown BEFORE the rollout summary
    rollout_logger.log_rollout_completion()
    
    # Log rollout summary in compact table format
    rollout_logger.log_rollout_summary_table(
        task_success=task_success,
        task_completed=task_completed,
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
    
    # Create summary dict
    summary_dict = {
        "rollout_id": rollout_id,
        "task_description": env.task_description[:50] + "..." if len(env.task_description) > 50 else env.task_description,
        "task_completed": task_completed,
        "task_success": task_success,
        "num_turns": num_turns,
        "reward": reward,
        "rollout_time": env_rollout_time,
        "temperature": temperature,
    }
    
    # Save trajectory BEFORE flush (so logs are still in buffer)
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
    
    # Flush all logs at once for this environment (after saving)
    rollout_logger.flush()
    
    # Create a simplified trajectory
    # For now, we create a single transition with the task description as observation
    # and a dummy action. In the future, we could extract actual messages from GBoxAgent.
    
    # Build initial observation from task description
    # Note: This is a simplified version. In practice, we'd want to include
    # the actual conversation history with screenshots.
    initial_ob = env.renderer.build_generation_prompt([
        {"role": "user", "content": env.task_description}
    ])
    
    # Create a dummy action (empty tokens) since we don't have the actual agent response
    # In practice, we'd extract this from GBoxAgent's internal state
    dummy_action = TokensWithLogprobs(tokens=[], maybe_logprobs=None)
    
    # Create transition
    transition = Transition(
        ob=initial_ob,
        ac=dummy_action,
        reward=reward,
        episode_done=True,
        metrics={
            "task_success": float(task_success),
            "task_completed": float(task_completed),
            "num_turns": num_turns,
            "rollout_time": env_rollout_time,  # Store rollout time for evaluation results
        },
    )
    
    # Create trajectory
    trajectory = Trajectory(
        transitions=[transition],
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
) -> TrajectoryGroup:
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
    
    # Run rollouts in parallel using asyncio.gather
    # This allows multiple GBox environments to execute simultaneously
    rollout_tasks = []
    for env_idx, env in enumerate(envs):
        # Use provided group number if available, otherwise fall back to env_idx
        # group should represent the group index in the batch, not the env index within the group
        group_num = group if group is not None else env_idx
        task = _run_single_env_rollout(
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
        )
        rollout_tasks.append(task)
    
    # Execute all rollouts in parallel
    results = await asyncio.gather(*rollout_tasks)
    
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
        logger.info("-" * 57)
        logger.info(f"Total environments: {len(envs)}")
        logger.info(f"Total rollout time: {total_rollout_time:.2f}s")
        logger.info(f"Average time per environment: {total_rollout_time / len(envs):.2f}s")
        success_count = sum(1 for m in metrics_list if m.get("task_success", False))
        logger.info(f"Successful tasks: {success_count}/{len(envs)} ({100 * success_count / len(envs):.1f}%)")
        total_turns = sum(m.get("num_turns", 0) for m in metrics_list)
        logger.info(f"Total turns across all environments: {total_turns}")
        logger.info(f"Average turns per environment: {total_turns / len(envs):.1f}")
        logger.info("=" * 57)
        
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

