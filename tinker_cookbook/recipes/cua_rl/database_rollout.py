"""
Database integration for rollout recording.

This module provides functions to record rollout data to the database.
"""

import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path
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
from tinker_cookbook.recipes.cua_rl.database_models import Task

logger = logging.getLogger(__name__)

# Screenshot storage directory (relative to project root or absolute path)
# Default: training-monitor/public/screenshots (Next.js will serve files from public/ as static assets)
# Calculate path: from tinker_cookbook/recipes/cua_rl/database_rollout.py
# go up 5 levels to project root, then into training-monitor/public/screenshots
_project_root = Path(__file__).parent.parent.parent.parent.parent
SCREENSHOT_STORAGE_DIR = os.getenv("SCREENSHOT_STORAGE_DIR", str(_project_root / "training-monitor" / "public" / "screenshots"))


def save_screenshot_to_file(screenshot_uri: str, rollout_id: str, turn_id: int, obs_id: Optional[int] = None) -> str:
    """
    Save screenshot from data URI to file system.
    
    Args:
        screenshot_uri: Data URI (data:image/...) or file path
        rollout_id: Rollout ID for organizing files
        turn_id: Turn ID for organizing files
        obs_id: Optional observation ID for unique filename
        
    Returns:
        Relative file path (e.g., "rollout_123/turn_1/obs_456.png")
    """
    # If it's already a file path (not data URI), return as-is
    if not screenshot_uri.startswith("data:"):
        return screenshot_uri
    
    # Parse data URI
    try:
        header, encoded = screenshot_uri.split(",", 1)
        # Extract image format from header (e.g., "data:image/png;base64" -> "png")
        image_format = "png"  # default
        if "image/" in header:
            format_part = header.split("image/")[1].split(";")[0]
            if format_part in ["png", "jpeg", "jpg", "webp"]:
                image_format = format_part
        
        # Decode base64
        image_bytes = base64.b64decode(encoded)
        
        # Create directory structure: screenshots/rollout_{rollout_id}/turn_{turn_id}/
        storage_path = Path(SCREENSHOT_STORAGE_DIR)
        rollout_dir = storage_path / f"rollout_{rollout_id}" / f"turn_{turn_id}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if obs_id is not None:
            filename = f"obs_{obs_id}.{image_format}"
        else:
            # Use timestamp if no obs_id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"screenshot_{timestamp}.{image_format}"
        
        file_path = rollout_dir / filename
        
        # Write file
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        # Return relative path from screenshots directory
        relative_path = f"rollout_{rollout_id}/turn_{turn_id}/{filename}"
        logger.debug(f"Saved screenshot to {file_path}, relative path: {relative_path}")
        
        return relative_path
        
    except Exception as e:
        logger.error(f"Failed to save screenshot to file: {e}")
        # Fallback: return original URI
        return screenshot_uri


def record_rollout_start(
    session: Session,
    source_type: str,
    rollout_id: str,  # UUID string (rollout_id is now unique UUID)
    task_id: int,  # Database task ID
    model_path: str,
    env_type: str,  # Environment type (android/linux)
    step_id: Optional[int] = None,
    eval_id: Optional[int] = None,
    baseline_id: Optional[int] = None,
    batch: Optional[int] = None,
    group: Optional[int] = None,
    env_index: Optional[int] = None,
    is_eval: bool = False,
    group_id: Optional[int] = None,  # Database group ID (if already created)
    box_type: Optional[str] = None,  # Box type for environment
    **kwargs
) -> str:
    """
    Record the start of a rollout.
    
    Creates Environment first, then Rollout. Environment is created with status="pending"
    and can be updated later with box_id when the box is created.
    
    If group_id is not provided, will create or get the group record.
    
    Returns:
        Rollout ID (UUID string)
    """
    # First, create Environment record (without box_id, it will be updated later)
    from tinker_cookbook.recipes.cua_rl.database_dao import create_environment
    env = create_environment(
        session,
        env_type=env_type,
        box_type=box_type,
        status="pending",  # Will be updated to "creating" when box creation starts
    )
    env_id = env.id
    
    # Create or get group if not provided
    if group_id is None and group is not None:
        from tinker_cookbook.recipes.cua_rl.database_dao import get_or_create_group
        group_obj = get_or_create_group(
            session,
            source_type=source_type,
            group_num=group,
            step_id=step_id,
            eval_id=eval_id,
            baseline_id=baseline_id,
            batch=batch,
        )
        group_id = group_obj.id
    
    # Update group: increment num_rollouts (whether group_id was provided or just created)
    # Note: We update group AFTER creating rollout to avoid any session issues
    # The group update will happen after rollout is created and flushed
    
    rollout = create_rollout(
        session,
        source_type=source_type,
        rollout_id=rollout_id,  # Keep rollout_id string for backward compatibility/logging
        task_id=task_id,
        model_path=model_path,
        env_id=env_id,  # Use the environment ID we just created
        step_id=step_id,
        eval_id=eval_id,
        baseline_id=baseline_id,
        group_id=group_id,
        batch=batch,
        group_num=group,  # Use group_num instead of group to avoid conflict with relationship
        env_index=env_index,
        is_eval=is_eval,
        status="pending",
        start_time=datetime.utcnow(),
        **kwargs
    )
    
    # Update group: increment num_rollouts AFTER rollout is created and flushed
    # This avoids any session state issues
    if group_id is not None:
        try:
            from tinker_cookbook.recipes.cua_rl.database_dao import get_group, update_group
            group_obj = get_group(session, group_id)
            if group_obj:
                update_group(
                    session,
                    group_id,
                    num_rollouts=(group_obj.num_rollouts or 0) + 1,
                )
        except Exception as e:
            logger.warning(f"Failed to update group {group_id} num_rollouts: {e}", exc_info=True)
            # Continue anyway - group update is not critical for rollout creation
    
    # Return rollout_id (which is now a UUID) for unique identification
    return rollout.rollout_id


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
    rollout_id: str,  # UUID string (rollout_id is now unique UUID)
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
    # Use rollout_id (which is now a unique UUID) for lookup
    db_rollout = get_rollout_by_rollout_id(session, rollout_id)
    if not db_rollout:
        logger.warning(f"Rollout with ID {rollout_id} not found in database")
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


def record_turn_start(
    session: Session,
    rollout_id: str,  # UUID string (rollout_id is now unique UUID)
    turn: int,
    start_time: Optional[datetime] = None,
    expected_task_id_str: Optional[str] = None,  # Optional: task ID string for verification
) -> int:
    """
    Record the start of a turn (create turn record at the beginning).
    
    **CRITICAL**: This function must be called with a session that is not shared
    across concurrent tasks to avoid data corruption. Each environment should
    use its own database session.
    
    Args:
        session: Database session
        rollout_id: Unique UUID string of the rollout
        turn: Turn number
        start_time: Start time (defaults to now)
    
    Returns:
        Database turn ID
    """
    # Expire all cached objects to ensure we get fresh data from database
    session.expire_all()
    
    # Use rollout_id (which is now a unique UUID) for lookup
    db_rollout = get_rollout_by_rollout_id(session, rollout_id)
    
    if not db_rollout:
        logger.warning(f"Rollout with ID {rollout_id} not found in database")
        return None
    
    # Verify rollout_id matches (extra safety check)
    if db_rollout.rollout_id != rollout_id:
        logger.error(
            f"CRITICAL: Rollout ID mismatch! Expected {rollout_id}, "
            f"but found {db_rollout.rollout_id} in database. "
            f"This indicates a data corruption issue."
        )
        return None
    
    # CRITICAL: Verify task_id matches if expected_task_id_str is provided
    if expected_task_id_str is not None:
        # Get the task from rollout to verify it matches
        rollout_task = session.query(Task).filter(Task.id == db_rollout.task_id).first()
        if rollout_task:
            if rollout_task.task_id != expected_task_id_str:
                logger.error(
                    f"CRITICAL: Task ID mismatch! Rollout {rollout_id} (DB ID: {db_rollout.id}) "
                    f"is associated with task_id '{rollout_task.task_id}' (DB ID: {db_rollout.task_id}), "
                    f"but agent is executing task_id '{expected_task_id_str}'. "
                    f"This indicates the rollout was created with the wrong task. Aborting turn start."
                )
                return None
        else:
            logger.warning(
                f"Could not find task for rollout {rollout_id} (task_id={db_rollout.task_id}) "
                f"to verify against expected_task_id_str={expected_task_id_str}"
            )
    
    # Check if turn already exists
    from tinker_cookbook.recipes.cua_rl.database_dao import get_turn_by_rollout_and_turn
    existing_turn = get_turn_by_rollout_and_turn(session, db_rollout.id, turn)
    if existing_turn:
        # CRITICAL: Verify that existing turn belongs to the correct rollout
        # Reload to avoid stale session cache
        session.expire(existing_turn)
        session.refresh(existing_turn)
        
        if existing_turn.rollout_id != db_rollout.id:
            logger.error(
                f"CRITICAL: Turn {turn} ID {existing_turn.id} belongs to rollout_id {existing_turn.rollout_id}, "
                f"but we're trying to use it for rollout_id {db_rollout.id} (rollout_id UUID: {rollout_id}). "
                f"This indicates a data corruption issue."
            )
            return None
        
        logger.debug(
            f"Reusing existing Turn {turn} (ID: {existing_turn.id}) for Rollout {rollout_id} (DB ID: {db_rollout.id})"
        )
        return existing_turn.id
    
    # Create turn record at start
    logger.debug(
        f"Creating new Turn {turn} for Rollout {rollout_id} (DB ID: {db_rollout.id}, Task ID: {db_rollout.task_id})"
    )
    turn_obj = create_turn(
        session,
        rollout_id=db_rollout.id,
        turn=turn,
        start_time=start_time or datetime.utcnow(),
    )
    
    # CRITICAL: Verify the created turn belongs to the correct rollout
    session.flush()  # Ensure turn is persisted
    session.refresh(turn_obj)  # Reload from database
    
    if turn_obj.rollout_id != db_rollout.id:
        logger.error(
            f"CRITICAL: Created Turn {turn_obj.id} has rollout_id={turn_obj.rollout_id}, "
            f"but expected {db_rollout.id} for Rollout {rollout_id}. "
            f"This indicates a serious data corruption issue. Deleting incorrect turn."
        )
        session.delete(turn_obj)
        session.flush()
        return None
    
    # Update rollout current_turn
    update_rollout(
        session,
        db_rollout.id,
        current_turn=turn,
        status="running",
    )
    
    logger.debug(
        f"Successfully created Turn {turn_obj.id} for Rollout {rollout_id} (DB ID: {db_rollout.id})"
    )
    return turn_obj.id


def record_turn(
    session: Session,
    rollout_id: str,  # UUID string (rollout_id is now unique UUID)
    turn: int,
    reward: float,
    episode_done: bool,
    metrics: Optional[Dict[str, Any]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    turn_time: Optional[float] = None,
    model_response: Optional[str] = None,  # Full LLM output text for this turn
    expected_task_id_str: Optional[str] = None,  # Optional: task ID string for verification
) -> int:
    """
    Record a turn (update existing turn record or create if not exists).
    
    This function can be called at turn end to update the turn record.
    If the turn record doesn't exist, it will be created.
    
    **CRITICAL**: This function must be called with a session that is not shared
    across concurrent tasks to avoid data corruption. Each environment should
    use its own database session.
    
    Args:
        session: Database session
        rollout_id: Unique UUID string of the rollout
        turn: Turn number
        reward: Reward for this turn
        episode_done: Whether episode is done
        metrics: Optional metrics dict
        start_time: Start time
        end_time: End time
        turn_time: Duration of turn
        model_response: Full LLM output text
    
    Returns:
        Database turn ID
    """
    # Expire all cached objects to ensure we get fresh data from database
    # This is important in concurrent scenarios to avoid stale data
    session.expire_all()
    
    # Use rollout_id (which is now a unique UUID) for lookup
    db_rollout = get_rollout_by_rollout_id(session, rollout_id)
    
    if not db_rollout:
        logger.warning(f"Rollout with ID {rollout_id} not found in database")
        return None
    
    # Verify rollout_id matches (extra safety check)
    if db_rollout.rollout_id != rollout_id:
        logger.error(
            f"CRITICAL: Rollout ID mismatch! Expected {rollout_id}, "
            f"but found {db_rollout.rollout_id} in database. "
            f"This indicates a data corruption issue."
        )
        return None
    
    # CRITICAL: Verify task_id matches if expected_task_id_str is provided
    if expected_task_id_str is not None:
        # Get the task from rollout to verify it matches
        from tinker_cookbook.recipes.cua_rl.database_models import Task
        rollout_task = session.query(Task).filter(Task.id == db_rollout.task_id).first()
        if rollout_task:
            if rollout_task.task_id != expected_task_id_str:
                logger.error(
                    f"CRITICAL: Task ID mismatch in record_turn! Rollout {rollout_id} (DB ID: {db_rollout.id}) "
                    f"is associated with task_id '{rollout_task.task_id}' (DB ID: {db_rollout.task_id}), "
                    f"but agent is executing task_id '{expected_task_id_str}'. "
                    f"This indicates the rollout was created with the wrong task. Aborting turn record."
                )
                return None
        else:
            logger.warning(
                f"Could not find task for rollout {rollout_id} (task_id={db_rollout.task_id}) "
                f"to verify against expected_task_id_str={expected_task_id_str}"
            )
    
    # Check if turn already exists
    # Use fresh query to avoid session cache issues
    # CRITICAL: Always query by database rollout.id (integer) to ensure we get the correct Turn
    from tinker_cookbook.recipes.cua_rl.database_dao import get_turn_by_rollout_and_turn, update_turn
    existing_turn = get_turn_by_rollout_and_turn(session, db_rollout.id, turn)
    
    if existing_turn:
        # CRITICAL: Verify that existing turn belongs to the correct rollout
        # Double-check: reload the turn from database to avoid stale session cache
        session.expire(existing_turn)
        session.refresh(existing_turn)
        
        if existing_turn.rollout_id != db_rollout.id:
            logger.error(
                f"CRITICAL: Turn {turn} ID {existing_turn.id} belongs to rollout_id {existing_turn.rollout_id}, "
                f"but we're trying to update it for rollout_id {db_rollout.id} (rollout_id UUID: {rollout_id}). "
                f"This indicates a data corruption issue. Creating new turn instead."
            )
            # Don't update the wrong turn - create a new one instead
            existing_turn = None
        else:
            # Additional verification: check that the Turn's rollout actually matches our UUID
            # Reload the rollout to ensure we have fresh data
            session.expire(db_rollout)
            session.refresh(db_rollout)
            if db_rollout.rollout_id != rollout_id:
                logger.error(
                    f"CRITICAL: After refresh, rollout ID mismatch! Expected {rollout_id}, "
                    f"but found {db_rollout.rollout_id} in database. Turn update aborted."
                )
                return None
            
            # Final verification: ensure Turn belongs to the correct Rollout
            # existing_turn.rollout_id is the database integer ID, which should match db_rollout.id
            # We already verified existing_turn.rollout_id == db_rollout.id above, so we're good
            
            # Update existing turn
            update_turn(
                session,
                existing_turn.id,
                reward=reward,
                episode_done=episode_done,
                metrics_json=metrics,
                end_time=end_time or datetime.utcnow(),
                turn_time=turn_time,
                model_response=model_response,
            )
            logger.debug(
                f"Updated existing Turn {turn} (ID: {existing_turn.id}) for Rollout {rollout_id} "
                f"(DB ID: {db_rollout.id}, Task ID: {db_rollout.task_id})"
            )
            return existing_turn.id
    else:
        # Create new turn (fallback if record_turn_start wasn't called)
        logger.debug(
            f"Creating new Turn {turn} (fallback) for Rollout {rollout_id} (DB ID: {db_rollout.id}, Task ID: {db_rollout.task_id})"
        )
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
            model_response=model_response,
        )
        
        # CRITICAL: Verify the created turn belongs to the correct rollout
        session.flush()  # Ensure turn is persisted
        session.refresh(turn_obj)  # Reload from database
        
        if turn_obj.rollout_id != db_rollout.id:
            logger.error(
                f"CRITICAL: Created Turn {turn_obj.id} (fallback) has rollout_id={turn_obj.rollout_id}, "
                f"but expected {db_rollout.id} for Rollout {rollout_id}. "
                f"This indicates a serious data corruption issue. Deleting incorrect turn."
            )
            session.delete(turn_obj)
            session.flush()
            return None
        
        logger.debug(
            f"Successfully created Turn {turn_obj.id} (fallback) for Rollout {rollout_id} (DB ID: {db_rollout.id})"
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
    
    **CRITICAL**: turn_id must be a valid database turn ID from the same session.
    Always use the turn_id returned by record_turn() in the same transaction.
    
    This function validates that turn_id exists and belongs to the expected rollout
    to prevent data corruption in concurrent scenarios.
    
    Returns:
        Database action ID
    """
    # Verify turn_id exists (safety check)
    from tinker_cookbook.recipes.cua_rl.database_dao import get_turn
    turn_obj = get_turn(session, turn_id)
    if turn_obj is None:
        logger.error(f"CRITICAL: Turn ID {turn_id} not found in database. Cannot record action.")
        raise ValueError(f"Invalid turn_id: {turn_id}")
    
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
    rollout_id: Optional[str] = None,
) -> int:
    """
    Record an observation.
    
    **CRITICAL**: turn_id must be a valid database turn ID from the same session.
    Always use the turn_id returned by record_turn() in the same transaction.
    
    Args:
        session: Database session
        turn_id: Turn ID
        obs_type: Observation type
        screenshot_uri: Screenshot URI (data URI or file path). If data URI, will be saved to file.
        text_content: Text content
        model_input: Model input (will be serialized to JSON)
        rollout_id: Rollout ID (required if screenshot_uri is a data URI, for file organization)
    
    Returns:
        Database observation ID
    """
    # Verify turn_id exists (safety check)
    from tinker_cookbook.recipes.cua_rl.database_dao import get_turn
    turn_obj = get_turn(session, turn_id)
    if turn_obj is None:
        logger.error(f"CRITICAL: Turn ID {turn_id} not found in database. Cannot record observation.")
        raise ValueError(f"Invalid turn_id: {turn_id}")
    
    # Save screenshot to file if it's a data URI
    screenshot_file_path = screenshot_uri
    if screenshot_uri and screenshot_uri.startswith("data:"):
        if not rollout_id:
            # Try to get rollout_id from turn
            rollout_obj = turn_obj.rollout
            if rollout_obj:
                rollout_id = rollout_obj.rollout_id
            else:
                logger.warning(f"Could not determine rollout_id for saving screenshot, using original URI")
                screenshot_file_path = screenshot_uri
        else:
            # Save to file (obs_id not available yet, will be None)
            screenshot_file_path = save_screenshot_to_file(screenshot_uri, rollout_id, turn_id, obs_id=None)
            # If we got an obs_id, we'll update it after creation
            # For now, we'll use a temporary filename and update later if needed
    
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
        screenshot_uri=screenshot_file_path,
        text_content=text_content,
        model_input_json=model_input_json,
    )
    
    # If screenshot was saved with temporary filename, update it with obs_id
    if screenshot_uri and screenshot_uri.startswith("data:") and rollout_id:
        try:
            # Re-save with proper obs_id filename
            new_path = save_screenshot_to_file(screenshot_uri, rollout_id, turn_id, obs_id=observation.id)
            if new_path != screenshot_file_path:
                # Update database with new path
                observation.screenshot_uri = new_path
                session.commit()
        except Exception as e:
            logger.warning(f"Failed to update screenshot path with obs_id: {e}")
    
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


def record_environment_start(
    session: Session,
    env_type: str,
    gbox_id: Optional[str] = None,
    box_type: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    status: str = "pending",
) -> int:
    """
    Record environment information. Environment is created BEFORE Rollout.
    
    Returns:
        Database environment ID
    """
    env = create_environment(
        session,
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

