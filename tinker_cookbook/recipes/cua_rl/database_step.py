"""
Database integration for step recording.

This module provides functions to record training step data to the database.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from tinker_cookbook.recipes.cua_rl.database_dao import (
    create_step,
    update_step,
    get_step_by_training_and_step,
)

logger = logging.getLogger(__name__)


def record_step_start(
    session: Session,
    training_id: int,
    step: int,
    batch: Optional[int] = None,
    **kwargs
) -> int:
    """
    Record the start of a training step.
    
    Returns:
        Database step ID
    """
    step_obj = create_step(
        session,
        training_id=training_id,
        step=step,
        batch=batch,
        status="pending",
        start_time=datetime.utcnow(),
        **kwargs
    )
    return step_obj.id


def record_step_rollout_start(
    session: Session,
    step_id: int,
    **kwargs
) -> None:
    """Record that rollout has started for this step."""
    update_step(
        session,
        step_id,
        status="rollout_running",
        current_phase="rollout_execution",
        rollout_start_time=datetime.utcnow(),
        **kwargs
    )


def record_step_rollout_progress(
    session: Session,
    step_id: int,
    completed_groups: int,
    total_groups: int,
    completed_envs: int,
    total_envs: int,
) -> None:
    """Update rollout progress for a step."""
    progress = {
        "completed_groups": completed_groups,
        "total_groups": total_groups,
        "completed_envs": completed_envs,
        "total_envs": total_envs,
    }
    progress_percent = (completed_envs / total_envs * 100) if total_envs > 0 else 0.0
    
    update_step(
        session,
        step_id,
        rollout_progress=progress,
        progress_percent=progress_percent,
    )


def record_step_rollout_complete(
    session: Session,
    step_id: int,
    **kwargs
) -> None:
    """Record that rollout has completed for this step."""
    update_step(
        session,
        step_id,
        rollout_end_time=datetime.utcnow(),
        **kwargs
    )


def record_step_training_start(
    session: Session,
    step_id: int,
    **kwargs
) -> None:
    """Record that training has started for this step."""
    update_step(
        session,
        step_id,
        status="training",
        current_phase="training",
        training_start_time=datetime.utcnow(),
        **kwargs
    )


def record_step_training_progress(
    session: Session,
    step_id: int,
    completed_substeps: int,
    total_substeps: int,
) -> None:
    """Update training progress for a step."""
    progress = {
        "completed_substeps": completed_substeps,
        "total_substeps": total_substeps,
    }
    # Calculate overall progress (assume rollout is 50%, training is 50%)
    rollout_progress = 50.0  # Rollout is complete
    training_progress = (completed_substeps / total_substeps * 50.0) if total_substeps > 0 else 0.0
    progress_percent = rollout_progress + training_progress
    
    update_step(
        session,
        step_id,
        training_progress=progress,
        progress_percent=progress_percent,
    )


def record_step_completion(
    session: Session,
    step_id: int,
    model_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    loss: Optional[float] = None,
    kl_divergence: Optional[float] = None,
    policy_gradient_norm: Optional[float] = None,
    reward_mean: Optional[float] = None,
    reward_std: Optional[float] = None,
    num_trajectories: Optional[int] = None,
    num_tokens: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """Record step completion with all metrics."""
    update_kwargs = {
        "status": "completed",
        "end_time": datetime.utcnow(),
        "training_end_time": datetime.utcnow(),
        "progress_percent": 100.0,
        **kwargs
    }
    
    if model_path:
        update_kwargs["model_path"] = model_path
    if checkpoint_path:
        update_kwargs["checkpoint_path"] = checkpoint_path
    if loss is not None:
        update_kwargs["loss"] = loss
    if kl_divergence is not None:
        update_kwargs["kl_divergence"] = kl_divergence
    if policy_gradient_norm is not None:
        update_kwargs["policy_gradient_norm"] = policy_gradient_norm
    if reward_mean is not None:
        update_kwargs["reward_mean"] = reward_mean
    if reward_std is not None:
        update_kwargs["reward_std"] = reward_std
    if num_trajectories is not None:
        update_kwargs["num_trajectories"] = num_trajectories
    if num_tokens is not None:
        update_kwargs["num_tokens"] = num_tokens
    if metrics:
        update_kwargs["metrics_json"] = metrics
    
    update_step(session, step_id, **update_kwargs)


def get_or_create_step(
    session: Session,
    training_id: int,
    step: int,
    batch: Optional[int] = None,
) -> int:
    """
    Get existing step or create a new one.
    
    Returns:
        Database step ID
    """
    existing_step = get_step_by_training_and_step(session, training_id, step)
    if existing_step:
        return existing_step.id
    
    return record_step_start(session, training_id, step, batch=batch)

