"""
Custom rollout function for CUA RL training.

This module provides a custom rollout function that integrates GBoxAgent
with the Tinker RL training framework. It uses TinkerAsyncOpenAIClient to
allow GBoxAgent to use the current training model for rollout (on-policy RL).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, cast

import tinker
from gbox_agent.agent import GBoxAgent

from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.recipes.cua_rl.cua_env import CUAEnv
from tinker_cookbook.recipes.cua_rl.vision_utils import convert_openai_responses_to_message
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)


async def do_cua_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
) -> TrajectoryGroup:
    """
    Custom rollout function for CUA environments.
    
    This function:
    1. Creates environments from the builder
    2. Gets the current training model's checkpoint path from the policy
    3. Uses Tinker's OpenAI-compatible API to run GBoxAgent rollout with the training model
    4. Converts the rollout results to TrajectoryGroup format
    
    The rollout model dynamically updates as training progresses (on-policy RL).
    
    Args:
        env_group_builder: Builder for creating environments
        policy: TokenCompleter (should be TinkerTokenCompleter with the training model)
        
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
    
    # Get the model path from the sampling client
    # The model_path is a tinker://... path that can be used with OpenAI-compatible API
    # According to Tinker docs, SamplingClient has a model_path attribute
    # Try multiple ways to get the model_path
    model_path = None
    
    # Method 1: Direct attribute access (most common)
    if hasattr(sampling_client, 'model_path'):
        model_path = sampling_client.model_path
    # Method 2: Check internal holder (if SamplingClient wraps a holder)
    elif hasattr(sampling_client, '_holder') and hasattr(sampling_client._holder, 'model_path'):
        model_path = sampling_client._holder.model_path
    # Method 3: Check holder attribute (alternative naming)
    elif hasattr(sampling_client, 'holder') and hasattr(sampling_client.holder, 'model_path'):
        model_path = sampling_client.holder.model_path
    # Method 4: Check __dict__ for any model_path-like attribute
    elif hasattr(sampling_client, '__dict__'):
        for key, value in sampling_client.__dict__.items():
            if 'model_path' in key.lower() and isinstance(value, str) and value.startswith('tinker://'):
                model_path = value
                break
    
    if model_path is None or not model_path.startswith('tinker://'):
        # If we can't get model_path, we need to save weights first to get a path
        # This happens when using base_model instead of model_path
        logger.warning(
            f"Cannot get model_path from SamplingClient (got: {model_path}). "
            "This may happen when using base_model. "
            "The rollout will use the base model via Tinker's OpenAI-compatible API, "
            "but it won't reflect training progress. "
            "Consider using save_weights_and_get_sampling_client() to get a checkpoint path."
        )
        # For now, we'll raise an error to make this explicit
        raise ValueError(
            "Cannot get model_path from SamplingClient. "
            "Please ensure the policy uses a SamplingClient created from a checkpoint path "
            "(e.g., via training_client.save_weights_and_get_sampling_client()). "
            f"SamplingClient attributes: {dir(sampling_client)}"
        )
    
    # Create environments
    envs = await env_group_builder.make_envs()
    
    # Run rollout for each environment
    trajectories = []
    final_rewards = []
    metrics_list = []
    
    for env in envs:
        if not isinstance(env, CUAEnv):
            raise ValueError(f"Expected CUAEnv, got {type(env)}")
        
        # Run GBoxAgent rollout with current training model
        # Use Tinker's OpenAI-compatible API endpoint and model_path
        rollout_result = await env.run_rollout_with_tinker_model(
            tinker_model_path=model_path,
            tinker_api_key=env.gbox_api_key,  # Use Tinker API key
        )
        
        # Extract results
        task_success = rollout_result.get("task_success", False)
        task_completed = rollout_result.get("task_completed", False)
        num_turns = rollout_result.get("num_turns", 0)
        
        # Reward is 1.0 if task succeeded, 0.0 otherwise
        reward = 1.0 if task_success else 0.0
        
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
        dummy_action = TokensWithLogprobs(tokens=[], logprobs=[])
        
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
            },
        )
        
        # Create trajectory
        trajectory = Trajectory(
            transitions=[transition],
            final_ob=tinker.ModelInput.empty(),
        )
        
        trajectories.append(trajectory)
        final_rewards.append(0.0)  # No additional group reward
        metrics_list.append({
            "task_success": float(task_success),
            "task_completed": float(task_completed),
            "num_turns": num_turns,
        })
    
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=final_rewards,
        metrics_G=metrics_list,
    )

