"""
Database integration for evaluation recording.

This module provides functions to record evaluation and baseline data to the database.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from tinker_cookbook.recipes.cua_rl.database.database_dao import (
    create_eval,
    create_baseline,
    update_eval,
    update_baseline,
    get_eval_by_training_and_step,
)

logger = logging.getLogger(__name__)


def record_eval_start(
    session: Session,
    training_id: int,
    step: int,
    model_path: str,
    total_tasks: int,
    **kwargs
) -> int:
    """
    Record the start of an evaluation.
    
    Returns:
        Database eval ID
    """
    eval_obj = create_eval(
        session,
        training_id=training_id,
        step=step,
        model_path=model_path,
        status="pending",
        total_tasks=total_tasks,
        completed_tasks=0,
        start_time=datetime.utcnow(),
        **kwargs
    )
    return eval_obj.id


def record_eval_progress(
    session: Session,
    eval_id: int,
    completed_tasks: int,
    current_task_index: Optional[int] = None,
    current_phase: Optional[str] = None,
) -> None:
    """Update evaluation progress."""
    eval_obj = update_eval(
        session,
        eval_id,
        completed_tasks=completed_tasks,
        current_task_index=current_task_index,
        current_phase=current_phase,
    )
    
    if eval_obj and eval_obj.total_tasks:
        progress_percent = (completed_tasks / eval_obj.total_tasks) * 100.0
        update_eval(
            session,
            eval_id,
            progress_percent=progress_percent,
        )


def record_eval_completion(
    session: Session,
    eval_id: int,
    success_rate: float,
    avg_reward: float,
    avg_turns: float,
    successful_tasks: int,
    metrics: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """Record evaluation completion."""
    update_kwargs = {
        "status": "completed",
        "end_time": datetime.utcnow(),
        "eval_time": datetime.utcnow(),
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_turns": avg_turns,
        "successful_tasks": successful_tasks,
        "progress_percent": 100.0,
        **kwargs
    }
    
    if metrics:
        update_kwargs["metrics_json"] = metrics
    
    update_eval(session, eval_id, **update_kwargs)


def get_or_create_eval(
    session: Session,
    training_id: int,
    step: int,
    model_path: str,
    total_tasks: int = 0,
) -> int:
    """
    Get existing eval or create a new one.
    
    Returns:
        Database eval ID
    """
    existing_eval = get_eval_by_training_and_step(session, training_id, step)
    if existing_eval:
        return existing_eval.id
    
    return record_eval_start(session, training_id, step, model_path, total_tasks)


def record_baseline_start(
    session: Session,
    training_id: int,
    model_path: str,
    total_tasks: int,
    **kwargs
) -> int:
    """
    Record the start of a baseline evaluation.
    
    Creates baseline with status="pending", then immediately updates to "running"
    since evaluation starts right after this function is called.
    
    Returns:
        Database baseline ID
    """
    from tinker_cookbook.recipes.cua_rl.database.database_dao import update_baseline
    
    baseline = create_baseline(
        session,
        training_id=training_id,
        model_path=model_path,
        status="pending",  # Initial status, will be updated immediately
        total_tasks=total_tasks,
        completed_tasks=0,
        start_time=datetime.utcnow(),
        **kwargs
    )
    
    # Immediately update status to "running" since evaluation starts right away
    update_baseline(
        session,
        baseline.id,
        status="running",
        current_phase="evaluation",
        progress_percent=0.0,
    )
    
    return baseline.id


def record_baseline_progress(
    session: Session,
    baseline_id: int,
    completed_tasks: int,
    current_task_index: Optional[int] = None,
    current_phase: Optional[str] = None,
) -> None:
    """Update baseline evaluation progress."""
    baseline = update_baseline(
        session,
        baseline_id,
        completed_tasks=completed_tasks,
        current_task_index=current_task_index,
        current_phase=current_phase,
    )
    
    if baseline and baseline.total_tasks:
        progress_percent = (completed_tasks / baseline.total_tasks) * 100.0
        update_baseline(
            session,
            baseline_id,
            progress_percent=progress_percent,
        )


def record_baseline_completion(
    session: Session,
    baseline_id: int,
    success_rate: float,
    avg_reward: float,
    avg_turns: float,
    successful_tasks: int,
    metrics: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """Record baseline evaluation completion."""
    update_kwargs = {
        "status": "completed",
        "end_time": datetime.utcnow(),
        "eval_time": datetime.utcnow(),
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_turns": avg_turns,
        "successful_tasks": successful_tasks,
        "progress_percent": 100.0,
        **kwargs
    }
    
    if metrics:
        update_kwargs["metrics_json"] = metrics
    
    update_baseline(session, baseline_id, **update_kwargs)

