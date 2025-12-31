"""
Training hooks for database recording.

This module provides hooks to record training step information to the database.
"""

import logging
from typing import Any, Dict, Optional

from tinker_cookbook.recipes.cua_rl.database_context import get_database_session, get_training_id
from tinker_cookbook.recipes.cua_rl.database_step import (
    get_or_create_step,
    record_step_rollout_start,
    record_step_rollout_complete,
    record_step_training_start,
    record_step_completion,
)
from tinker_cookbook.recipes.cua_rl.database_rollout import list_rollouts_by_step

logger = logging.getLogger(__name__)


def record_step_before_rollout(step: int, batch: Optional[int] = None) -> Optional[int]:
    """
    Record step before rollout starts.
    Called before collecting trajectories.
    
    Returns:
        Database step ID, or None if database not available
    """
    session = get_database_session()
    training_id = get_training_id()
    
    if not session or not training_id:
        return None
    
    try:
        step_id = get_or_create_step(session, training_id, step, batch=batch)
        record_step_rollout_start(session, step_id)
        session.commit()
        return step_id
    except Exception as e:
        logger.warning(f"Failed to record step before rollout: {e}")
        session.rollback()
        return None


def record_step_after_rollout(step: int, model_path: Optional[str] = None) -> Optional[int]:
    """
    Record step after rollout completes.
    Called after trajectories are collected but before training.
    
    Returns:
        Database step ID, or None if database not available
    """
    session = get_database_session()
    training_id = get_training_id()
    
    if not session or not training_id:
        return None
    
    try:
        from tinker_cookbook.recipes.cua_rl.database_dao import get_step_by_training_and_step
        step_obj = get_step_by_training_and_step(session, training_id, step)
        if not step_obj:
            return None
        
        # Count rollouts for this step
        rollouts = list_rollouts_by_step(session, step_obj.id)
        num_trajectories = len(rollouts)
        
        record_step_rollout_complete(
            session,
            step_obj.id,
            num_trajectories=num_trajectories,
        )
        session.commit()
        return step_obj.id
    except Exception as e:
        logger.warning(f"Failed to record step after rollout: {e}")
        session.rollback()
        return None


def record_step_before_training(step: int) -> Optional[int]:
    """
    Record step before training starts.
    Called before the training step begins.
    
    Returns:
        Database step ID, or None if database not available
    """
    session = get_database_session()
    training_id = get_training_id()
    
    if not session or not training_id:
        return None
    
    try:
        from tinker_cookbook.recipes.cua_rl.database_dao import get_step_by_training_and_step
        step_obj = get_step_by_training_and_step(session, training_id, step)
        if not step_obj:
            return None
        
        record_step_training_start(session, step_obj.id)
        session.commit()
        return step_obj.id
    except Exception as e:
        logger.warning(f"Failed to record step before training: {e}")
        session.rollback()
        return None


def record_step_after_training(
    step: int,
    model_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """
    Record step after training completes.
    Called after the training step finishes.
    
    Returns:
        Database step ID, or None if database not available
    """
    session = get_database_session()
    training_id = get_training_id()
    
    if not session or not training_id:
        return None
    
    try:
        from tinker_cookbook.recipes.cua_rl.database_dao import get_step_by_training_and_step
        step_obj = get_step_by_training_and_step(session, training_id, step)
        if not step_obj:
            return None
        
        # Extract metrics
        loss = metrics.get("train/loss") if metrics else None
        kl_divergence = metrics.get("train/kl") if metrics else None
        reward_mean = metrics.get("trajectory/reward_mean") if metrics else None
        reward_std = metrics.get("trajectory/reward_std") if metrics else None
        
        # Count tokens from rollouts
        rollouts = list_rollouts_by_step(session, step_obj.id)
        num_tokens = 0
        for rollout in rollouts:
            # Estimate tokens from turns (rough estimate)
            if rollout.num_turns:
                num_tokens += rollout.num_turns * 100  # Rough estimate
        
        record_step_completion(
            session,
            step_obj.id,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            loss=loss,
            kl_divergence=kl_divergence,
            reward_mean=reward_mean,
            reward_std=reward_std,
            num_trajectories=len(rollouts),
            num_tokens=num_tokens,
            metrics=metrics,
        )
        
        # Also update training-level current_step
        update_training_progress(step)
        
        session.commit()
        return step_obj.id
    except Exception as e:
        logger.warning(f"Failed to record step after training: {e}")
        session.rollback()
        return None


def update_training_progress(step: int, total_steps: Optional[int] = None) -> None:
    """Update training progress."""
    session = get_database_session()
    training_id = get_training_id()
    
    if not session or not training_id:
        return
    
    try:
        from tinker_cookbook.recipes.cua_rl.database_dao import update_training
        progress_percent = (step / total_steps * 100.0) if total_steps else None
        update_training(
            session,
            training_id,
            current_step=step,
            total_steps=total_steps,
            progress_percent=progress_percent,
        )
        session.commit()
    except Exception as e:
        logger.warning(f"Failed to update training progress: {e}")
        session.rollback()

