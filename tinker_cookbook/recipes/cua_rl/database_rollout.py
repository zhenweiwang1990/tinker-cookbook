"""
Database integration for rollout recording.

This module provides functions to record rollout data to the database.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from tinker_cookbook.recipes.cua_rl.database_dao import (
    create_rollout,
    create_turn,
    create_action,
    create_observation,
    create_validation,
    create_environment,
    update_rollout,
    update_turn,
    get_rollout_by_rollout_id,
    get_task_by_task_id,
)

logger = logging.getLogger(__name__)


def record_rollout_start(
    session: Session,
    source_type: str,
    rollout_id: str,
    task_id: int,  # Database task ID
    model_path: str,
    step_id: Optional[int] = None,
    eval_id: Optional[int] = None,
    baseline_id: Optional[int] = None,
    batch: Optional[int] = None,
    group: Optional[int] = None,
    env_index: Optional[int] = None,
    is_eval: bool = False,
    **kwargs
) -> int:
    """
    Record the start of a rollout.
    
    Returns:
        Database rollout ID
    """
    rollout = create_rollout(
        session,
        source_type=source_type,
        rollout_id=rollout_id,
        task_id=task_id,
        model_path=model_path,
        step_id=step_id,
        eval_id=eval_id,
        baseline_id=baseline_id,
        batch=batch,
        group=group,
        env_index=env_index,
        is_eval=is_eval,
        status="pending",
        start_time=datetime.utcnow(),
        **kwargs
    )
    return rollout.id


def record_rollout_status(
    session: Session,
    rollout_id: str,  # Rollout ID string
    status: str,
    progress_percent: Optional[float] = None,
    current_phase: Optional[str] = None,
    status_message: Optional[str] = None,
    current_turn: Optional[int] = None,
    **kwargs
) -> None:
    """Update rollout status."""
    db_rollout = get_rollout_by_rollout_id(session, rollout_id)
    if not db_rollout:
        logger.warning(f"Rollout {rollout_id} not found in database")
        return
    
    update_kwargs = {
        "status": status,
        **kwargs
    }
    if progress_percent is not None:
        update_kwargs["progress_percent"] = progress_percent
    if current_phase is not None:
        update_kwargs["current_phase"] = current_phase
    if status_message is not None:
        update_kwargs["status_message"] = status_message
    if current_turn is not None:
        update_kwargs["current_turn"] = current_turn
    
    update_rollout(session, db_rollout.id, **update_kwargs)


def record_rollout_completion(
    session: Session,
    rollout_id: str,
    task_completed: bool,
    task_success: bool,
    agent_reported_success: bool,
    validation_passed: bool,
    num_turns: int,
    reward: float,
    rollout_time: float,
    errors: Optional[list] = None,
    summary: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """Record rollout completion."""
    db_rollout = get_rollout_by_rollout_id(session, rollout_id)
    if not db_rollout:
        logger.warning(f"Rollout {rollout_id} not found in database")
        return
    
    update_kwargs = {
        "status": "completed",
        "task_completed": task_completed,
        "task_success": task_success,
        "agent_reported_success": agent_reported_success,
        "validation_passed": validation_passed,
        "num_turns": num_turns,
        "reward": reward,
        "rollout_time": rollout_time,
        "end_time": datetime.utcnow(),
        "progress_percent": 100.0,
        **kwargs
    }
    
    if errors:
        update_kwargs["errors"] = errors
    if summary:
        update_kwargs["summary_json"] = summary
    
    update_rollout(session, db_rollout.id, **update_kwargs)


def record_turn(
    session: Session,
    rollout_id: str,
    turn: int,
    reward: float,
    episode_done: bool,
    metrics: Optional[Dict[str, Any]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    turn_time: Optional[float] = None,
) -> int:
    """
    Record a turn.
    
    Returns:
        Database turn ID
    """
    db_rollout = get_rollout_by_rollout_id(session, rollout_id)
    if not db_rollout:
        logger.warning(f"Rollout {rollout_id} not found in database")
        return None
    
    turn_obj = create_turn(
        session,
        rollout_id=db_rollout.id,
        turn=turn,
        reward=reward,
        episode_done=episode_done,
        metrics_json=metrics,
        start_time=start_time or datetime.utcnow(),
        end_time=end_time,
        turn_time=turn_time,
    )
    return turn_obj.id


def record_action(
    session: Session,
    turn_id: int,
    action_type: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
    tokens: Optional[list] = None,
    logprobs: Optional[list] = None,
) -> int:
    """
    Record an action.
    
    Returns:
        Database action ID
    """
    action = create_action(
        session,
        turn_id=turn_id,
        action_type=action_type,
        tool_name=tool_name,
        tool_args=tool_args,
        tokens=tokens,
        logprobs=logprobs,
        num_tokens=len(tokens) if tokens else 0,
    )
    return action.id


def record_observation(
    session: Session,
    turn_id: int,
    obs_type: Optional[str] = None,
    screenshot_uri: Optional[str] = None,
    text_content: Optional[str] = None,
    model_input: Optional[Any] = None,
) -> int:
    """
    Record an observation.
    
    Returns:
        Database observation ID
    """
    model_input_json = None
    if model_input is not None:
        # Try to serialize model_input
        try:
            if hasattr(model_input, 'to_dict'):
                model_input_json = json.dumps(model_input.to_dict(), default=str)
            elif isinstance(model_input, dict):
                model_input_json = json.dumps(model_input, default=str)
            else:
                model_input_json = str(model_input)
        except Exception as e:
            logger.warning(f"Failed to serialize model_input: {e}")
            model_input_json = str(model_input)
    
    observation = create_observation(
        session,
        turn_id=turn_id,
        obs_type=obs_type,
        screenshot_uri=screenshot_uri,
        text_content=text_content,
        model_input_json=model_input_json,
    )
    return observation.id


def record_validation(
    session: Session,
    rollout_id: str,
    success: bool,
    validation_query: Optional[str] = None,
    expected_result: Optional[str] = None,
    actual_result: Optional[str] = None,
    execution_time: Optional[float] = None,
    error_message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    validator_id: Optional[int] = None,
) -> int:
    """
    Record validation result.
    
    Returns:
        Database validation ID
    """
    db_rollout = get_rollout_by_rollout_id(session, rollout_id)
    if not db_rollout:
        logger.warning(f"Rollout {rollout_id} not found in database")
        return None
    
    validation = create_validation(
        session,
        rollout_id=db_rollout.id,
        success=success,
        validation_query=validation_query,
        expected_result=expected_result,
        actual_result=actual_result,
        execution_time=execution_time,
        error_message=error_message,
        details_json=details,
        validator_id=validator_id,
    )
    return validation.id


def record_environment(
    session: Session,
    rollout_id: str,
    env_type: str,
    gbox_id: Optional[str] = None,
    box_type: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    status: str = "pending",
) -> int:
    """
    Record environment information.
    
    Returns:
        Database environment ID
    """
    db_rollout = get_rollout_by_rollout_id(session, rollout_id)
    if not db_rollout:
        logger.warning(f"Rollout {rollout_id} not found in database")
        return None
    
    env = create_environment(
        session,
        rollout_id=db_rollout.id,
        env_type=env_type,
        gbox_id=gbox_id,
        box_type=box_type,
        config_json=config,
        status=status,
        creation_time=datetime.utcnow() if status != "pending" else None,
    )
    return env.id


def get_task_db_id(session: Session, task_id: str) -> Optional[int]:
    """
    Get database task ID from task_id string.
    
    Args:
        session: Database session
        task_id: Task ID string (e.g., from CUATask.id)
        
    Returns:
        Database task ID, or None if not found
    """
    task = get_task_by_task_id(session, task_id)
    return task.id if task else None

