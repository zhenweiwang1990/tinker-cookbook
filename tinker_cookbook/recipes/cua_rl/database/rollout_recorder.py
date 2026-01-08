"""
Refactored database recording logic for CUA rollouts.

This module provides a clean interface for recording rollout data with proper
isolation and verification to prevent data corruption from concurrent access.

Key principles:
1. Each rollout uses its own database session
2. All queries include explicit verification
3. Task lookups are session-isolated
4. Clear logging for debugging
5. Integrates with ProgressTracker for consistent progress updates
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from tinker_cookbook.recipes.cua_rl.database.database_models import Task, Rollout, Turn
from tinker_cookbook.recipes.cua_rl.database.database_dao import (
    get_task_by_task_id,
    create_environment,
    get_or_create_group,
    update_group,
    get_group,
    create_rollout as dao_create_rollout,
    get_rollout_by_rollout_id,
    update_rollout as dao_update_rollout,
    create_turn as dao_create_turn,
    update_turn as dao_update_turn,
    get_turn_by_rollout_and_turn,
    create_action as dao_create_action,
    create_observation as dao_create_observation,
    create_validation as dao_create_validation,
    get_validation_by_rollout as dao_get_validation_by_rollout,
)
from tinker_cookbook.recipes.cua_rl.database.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class RolloutRecorder:
    """
    Records rollout data to database with proper isolation and verification.
    
    Each RolloutRecorder instance should use its own database session to avoid
    concurrent access issues.
    """
    
    def __init__(self, session: Session, rollout_uuid: str):
        """
        Initialize rollout recorder.
        
        Args:
            session: Database session (should be unique to this rollout)
            rollout_uuid: Unique UUID for this rollout
        """
        self.session = session
        self.rollout_uuid = rollout_uuid
        self.rollout_db_id: Optional[int] = None
        self.task_db_id: Optional[int] = None
        self.task_id_str: Optional[str] = None
        self.max_turns: Optional[int] = None
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(session)
        
    def start_rollout(
        self,
        task_id_str: str,
        task_description: str,
        model_path: str,
        env_type: str,
        source_type: str,
        step_id: Optional[int] = None,
        eval_id: Optional[int] = None,
        baseline_id: Optional[int] = None,
        batch: Optional[int] = None,
        group_num: Optional[int] = None,
        env_index: Optional[int] = None,
        is_eval: bool = False,
        group_id: Optional[int] = None,
        box_type: Optional[str] = None,
        max_turns: Optional[int] = None,
    ) -> bool:
        """
        Record the start of a rollout.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Look up task in database (with session isolation)
            # Force a fresh query by expiring all cached objects
            self.session.expire_all()
            
            task = self.session.query(Task).filter(Task.task_id == task_id_str).first()
            
            if not task:
                logger.error(
                    f"[RolloutRecorder] Task '{task_id_str}' not found in database. "
                    f"Tasks should be pre-loaded during dataset initialization."
                )
                return False
            
            self.task_db_id = task.id
            self.task_id_str = task_id_str
            
            # Verify task description matches (sanity check)
            if task.description[:50] != task_description[:50]:
                logger.warning(
                    f"[RolloutRecorder] Task description mismatch! "
                    f"DB: '{task.description[:50]}...' vs Env: '{task_description[:50]}...'"
                )
            
            logger.info(
                f"[RolloutRecorder] Found task in DB: ID={task.id}, task_id='{task.task_id}', "
                f"desc='{task.description[:80]}...'"
            )
            
            # Step 2: Create environment record
            env = create_environment(
                self.session,
                env_type=env_type,
                box_type=box_type,
                status="pending",
            )
            env_id = env.id
            
            # Step 3: Create or get group
            if group_id is None and group_num is not None:
                # Validate source_type before creating group
                if source_type not in ["step", "eval", "baseline"]:
                    logger.warning(
                        f"[RolloutRecorder] Invalid source_type '{source_type}'. "
                        f"Defaulting to 'step' for training rollout."
                    )
                    source_type = "step"
                
                group_obj = get_or_create_group(
                    self.session,
                    source_type=source_type,
                    group_num=group_num,
                    step_id=step_id,
                    eval_id=eval_id,
                    baseline_id=baseline_id,
                    batch=batch,
                )
                group_id = group_obj.id
            
            # Step 4: Create rollout record
            rollout = dao_create_rollout(
                self.session,
                source_type=source_type,
                rollout_id=self.rollout_uuid,
                task_id=self.task_db_id,  # Use the verified task_db_id
                model_path=model_path,
                env_id=env_id,
                step_id=step_id,
                eval_id=eval_id,
                baseline_id=baseline_id,
                group_id=group_id,
                batch=batch,
                group_num=group_num,
                env_index=env_index,
                is_eval=is_eval,
                status="pending",
                start_time=datetime.utcnow(),
                max_turns=max_turns or 20,  # Default 20 turns
            )
            
            self.rollout_db_id = rollout.id
            self.max_turns = max_turns or 20
            
            # Step 5: Update group num_rollouts
            if group_id is not None:
                group_obj = get_group(self.session, group_id)
                if group_obj:
                    update_group(
                        self.session,
                        group_id,
                        num_rollouts=(group_obj.num_rollouts or 0) + 1,
                    )
            
            # Commit
            self.session.commit()
            
            logger.info(
                f"[RolloutRecorder] Created rollout: UUID={self.rollout_uuid}, "
                f"DB_ID={self.rollout_db_id}, task_id='{self.task_id_str}' (DB ID={self.task_db_id})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[RolloutRecorder] Failed to start rollout: {e}", exc_info=True)
            self.session.rollback()
            return False
    
    def update_status(
        self,
        status: str,
        progress_percent: Optional[float] = None,
        current_phase: Optional[str] = None,
        status_message: Optional[str] = None,
        current_turn: Optional[int] = None,
    ) -> bool:
        """Update rollout status."""
        if not self.rollout_db_id:
            logger.warning("[RolloutRecorder] Cannot update status - rollout not initialized")
            return False
        
        try:
            update_kwargs = {"status": status}
            if progress_percent is not None:
                update_kwargs["progress_percent"] = progress_percent
            if current_phase is not None:
                update_kwargs["current_phase"] = current_phase
            if status_message is not None:
                update_kwargs["status_message"] = status_message
            if current_turn is not None:
                update_kwargs["current_turn"] = current_turn
            
            dao_update_rollout(self.session, self.rollout_db_id, **update_kwargs)
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"[RolloutRecorder] Failed to update status: {e}")
            self.session.rollback()
            return False
    
    def start_turn(self, turn_num: int) -> Optional[int]:
        """
        Record the start of a turn and update progress.
        
        Returns:
            Turn database ID, or None if failed
        """
        if not self.rollout_db_id:
            logger.error("[RolloutRecorder] Cannot start turn - rollout not initialized")
            return None
        
        try:
            # Expire cached objects to ensure fresh query
            self.session.expire_all()
            
            # Verify rollout still exists and matches
            rollout = self.session.query(Rollout).filter(Rollout.id == self.rollout_db_id).first()
            if not rollout or rollout.rollout_id != self.rollout_uuid:
                logger.error(
                    f"[RolloutRecorder] Rollout verification failed! "
                    f"Expected UUID={self.rollout_uuid}, DB_ID={self.rollout_db_id}"
                )
                return None
            
            # Check if turn already exists
            existing_turn = get_turn_by_rollout_and_turn(self.session, self.rollout_db_id, turn_num)
            if existing_turn:
                logger.debug(f"[RolloutRecorder] Turn {turn_num} already exists (ID={existing_turn.id})")
                return existing_turn.id
            
            # Create turn
            turn_obj = dao_create_turn(
                self.session,
                rollout_id=self.rollout_db_id,
                turn=turn_num,
                start_time=datetime.utcnow(),
            )
            
            # Update rollout progress using ProgressTracker
            max_turns = self.max_turns or rollout.max_turns or 20
            progress_stats = self.progress_tracker.update_rollout_progress(
                rollout_id=self.rollout_db_id,
                current_turn=turn_num,
                max_turns=max_turns,
                status="running",
            )
            
            self.session.commit()
            
            logger.debug(
                f"[RolloutRecorder] Started turn {turn_num} (ID={turn_obj.id}), "
                f"progress={progress_stats.progress_percent:.1f}%, "
                f"ETA={self.progress_tracker.format_time_estimate(progress_stats.estimated_remaining_time)}"
            )
            return turn_obj.id
            
        except Exception as e:
            logger.error(f"[RolloutRecorder] Failed to start turn {turn_num}: {e}", exc_info=True)
            self.session.rollback()
            return None
    
    def end_turn(
        self,
        turn_num: int,
        model_response: str,
        reward: float = 0.0,
        episode_done: bool = False,
        turn_time: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """
        Record the end of a turn and update progress.
        
        Returns:
            Turn database ID, or None if failed
        """
        if not self.rollout_db_id:
            logger.error("[RolloutRecorder] Cannot end turn - rollout not initialized")
            return None
        
        try:
            # Expire cached objects
            self.session.expire_all()
            
            # Get turn (should already exist from start_turn)
            turn_obj = get_turn_by_rollout_and_turn(self.session, self.rollout_db_id, turn_num)
            
            if not turn_obj:
                # Create turn if it doesn't exist (fallback)
                logger.warning(f"[RolloutRecorder] Turn {turn_num} not found, creating it now")
                turn_obj = dao_create_turn(
                    self.session,
                    rollout_id=self.rollout_db_id,
                    turn=turn_num,
                    start_time=datetime.utcnow(),
                )
            
            # Update turn
            dao_update_turn(
                self.session,
                turn_obj.id,
                reward=reward,
                episode_done=episode_done,
                metrics_json=metrics,
                end_time=datetime.utcnow(),
                turn_time=turn_time,
                model_response=model_response,
            )
            
            self.session.commit()
            
            # Log with comprehensive info
            task_desc_preview = ""
            if self.task_db_id:
                task = self.session.query(Task).filter(Task.id == self.task_db_id).first()
                if task:
                    task_desc_preview = task.description[:50] + "..." if len(task.description) > 50 else task.description
            
            model_resp_preview = model_response[:100] + "..." if len(model_response) > 100 else model_response
            
            logger.info(
                f"[RolloutRecorder] Ended turn {turn_num} (Turn DB ID={turn_obj.id}) | "
                f"Rollout UUID={self.rollout_uuid} | Task='{self.task_id_str}' | "
                f"Task desc: {task_desc_preview} | Model response: {model_resp_preview}"
            )
            
            return turn_obj.id
            
        except Exception as e:
            logger.error(f"[RolloutRecorder] Failed to end turn {turn_num}: {e}", exc_info=True)
            self.session.rollback()
            return None
    
    def complete_rollout(
        self,
        task_completed: bool,
        task_success: bool,
        agent_reported_success: bool,
        validation_passed: bool,
        num_turns: int,
        reward: float,
        rollout_time: float,
        **kwargs
    ) -> bool:
        """Record rollout completion and update progress."""
        if not self.rollout_db_id:
            logger.warning("[RolloutRecorder] Cannot complete rollout - not initialized")
            return False
        
        try:
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
            
            dao_update_rollout(self.session, self.rollout_db_id, **update_kwargs)
            
            # Update progress using ProgressTracker (this will also cascade to group/step/eval/baseline/training)
            max_turns = kwargs.get('max_turns', self.max_turns or 20)
            progress_stats = self.progress_tracker.update_rollout_progress(
                rollout_id=self.rollout_db_id,
                current_turn=num_turns,
                max_turns=max_turns,
                status="completed",
            )
            
            self.session.commit()
            
            logger.info(
                f"[RolloutRecorder] Completed rollout: UUID={self.rollout_uuid}, "
                f"success={task_success}, reward={reward:.4f}, "
                f"progress=100%, turns={num_turns}/{max_turns}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[RolloutRecorder] Failed to complete rollout: {e}")
            self.session.rollback()
            return False

    def record_validation(
        self,
        *,
        success: bool,
        validation_query: Optional[str] = None,
        expected_result: Optional[str] = None,
        actual_result: Optional[str] = None,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None,
        screenshot_uri: Optional[str] = None,
        details_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Create/update validation row for this rollout (drives Validation tab in training-monitor)."""
        if not self.rollout_db_id:
            logger.warning("[RolloutRecorder] Cannot record validation - not initialized")
            return None

        try:
            merged_details: Dict[str, Any] = {}
            if details_json:
                merged_details.update(details_json)
            if screenshot_uri:
                merged_details.setdefault("screenshot_uri", screenshot_uri)

            existing = dao_get_validation_by_rollout(self.session, self.rollout_db_id)
            if existing is None:
                v = dao_create_validation(
                    self.session,
                    rollout_id=self.rollout_db_id,
                    success=success,
                    validation_query=validation_query,
                    expected_result=expected_result,
                    actual_result=actual_result,
                    execution_time=execution_time,
                    error_message=error_message,
                    details_json=merged_details or None,
                )
                self.session.commit()
                logger.info(f"[RolloutRecorder] Recorded validation (Validation DB ID={v.id})")
                return v.id

            # Update existing row (training-monitor expects one validation per rollout)
            existing.success = success
            if validation_query is not None:
                existing.validation_query = validation_query
            if expected_result is not None:
                existing.expected_result = expected_result
            if actual_result is not None:
                existing.actual_result = actual_result
            if execution_time is not None:
                existing.execution_time = execution_time
            if error_message is not None:
                existing.error_message = error_message
            if merged_details:
                import json
                existing.details_json = json.dumps(merged_details)

            self.session.commit()
            logger.info(f"[RolloutRecorder] Updated validation (Validation DB ID={existing.id})")
            return existing.id
        except Exception as e:
            logger.error(f"[RolloutRecorder] Failed to record validation: {e}", exc_info=True)
            self.session.rollback()
            return None
    
    # Backward-compatible helper methods for gradual migration
    def record_turn_wrapper(
        self,
        turn_num: int,
        model_response: str,
        reward: float = 0.0,
        episode_done: bool = False,
        metrics: Optional[Dict[str, Any]] = None,
        turn_time: Optional[float] = None,
    ) -> Optional[int]:
        """
        Backward-compatible wrapper for record_turn().
        
        Combines start_turn() and end_turn() into one call.
        """
        # Ensure turn exists
        turn_id = self.start_turn(turn_num)
        if not turn_id:
            logger.warning(f"[RolloutRecorder] Failed to start turn {turn_num}")
            # Try to end it anyway
            pass
        
        # End turn with details
        return self.end_turn(
            turn_num=turn_num,
            model_response=model_response,
            reward=reward,
            episode_done=episode_done,
            turn_time=turn_time,
            metrics=metrics,
        )
    
    def record_action(
        self,
        turn_num: int,
        action_type: str,
        **kwargs
    ) -> Optional[int]:
        """
        Record an action for a turn.
        
        Args:
            turn_num: Turn number
            action_type: Type of action (e.g., 'click', 'input', 'wait')
            **kwargs: Additional action fields (target_description, screenshot_before_path, etc.)
        
        Returns:
            Action DB ID if successful, None otherwise
        """
        try:
            # Import here to avoid circular dependency
            from tinker_cookbook.recipes.cua_rl.database.database_dao import create_action
            from tinker_cookbook.recipes.cua_rl.database.database_models import Turn
            
            # Get turn DB ID - if turn doesn't exist, create it
            turn_obj = self.session.query(Turn).filter(
                Turn.rollout_id == self.rollout_db_id,
                Turn.turn == turn_num
            ).first()
            
            if not turn_obj:
                logger.info(f"[RolloutRecorder] Turn {turn_num} not found, creating it now for action recording")
                turn_id = self.start_turn(turn_num)
                if not turn_id:
                    logger.error(f"[RolloutRecorder] Failed to create turn {turn_num} for action")
                    return None
                # Re-fetch the turn object
                turn_obj = self.session.query(Turn).filter(
                    Turn.rollout_id == self.rollout_db_id,
                    Turn.turn == turn_num
                ).first()
                if not turn_obj:
                    logger.error(f"[RolloutRecorder] Turn {turn_num} created but not found in DB")
                    return None
            
            # Create action
            action = create_action(
                self.session,
                turn_id=turn_obj.id,
                action_type=action_type,
                **kwargs
            )
            
            self.session.commit()
            logger.info(f"[RolloutRecorder] Recorded action {action_type} for turn {turn_num} (Action DB ID={action.id})")
            return action.id
            
        except Exception as e:
            logger.error(f"[RolloutRecorder] Failed to record action for turn {turn_num}: {e}", exc_info=True)
            self.session.rollback()
            return None
    
    def record_observation(
        self,
        turn_num: int,
        **kwargs
    ) -> Optional[int]:
        """
        Record an observation for a turn.
        
        Args:
            turn_num: Turn number
            **kwargs: Additional observation fields (obs_type, screenshot_uri, text_content, model_input, etc.)
                     rollout_id: Optional, used for screenshot saving (not passed to DB model)
        
        Returns:
            Observation DB ID if successful, None otherwise
        """
        try:
            # Import here to avoid circular dependency
            from tinker_cookbook.recipes.cua_rl.database.database_dao import create_observation
            from tinker_cookbook.recipes.cua_rl.database.database_models import Turn
            
            # Get turn DB ID - if turn doesn't exist, create it
            turn_obj = self.session.query(Turn).filter(
                Turn.rollout_id == self.rollout_db_id,
                Turn.turn == turn_num
            ).first()
            
            if not turn_obj:
                logger.info(f"[RolloutRecorder] Turn {turn_num} not found, creating it now for observation recording")
                turn_id = self.start_turn(turn_num)
                if not turn_id:
                    logger.error(f"[RolloutRecorder] Failed to create turn {turn_num} for observation")
                    return None
                # Re-fetch the turn object
                turn_obj = self.session.query(Turn).filter(
                    Turn.rollout_id == self.rollout_db_id,
                    Turn.turn == turn_num
                ).first()
                if not turn_obj:
                    logger.error(f"[RolloutRecorder] Turn {turn_num} created but not found in DB")
                    return None
            
            # Extract rollout_id for screenshot saving (don't pass to DB model)
            rollout_id_for_screenshot = kwargs.pop('rollout_id', None)
            
            # Handle screenshot saving if it's a data URI
            screenshot_uri = kwargs.get('screenshot_uri')
            if screenshot_uri and screenshot_uri.startswith("data:"):
                from tinker_cookbook.recipes.cua_rl.database.database_rollout import save_screenshot_to_file
                
                # Use provided rollout_id or get from turn's rollout
                actual_rollout_id = rollout_id_for_screenshot
                if not actual_rollout_id:
                    rollout_obj = turn_obj.rollout
                    if rollout_obj:
                        actual_rollout_id = rollout_obj.rollout_id
                
                if actual_rollout_id:
                    screenshot_file_path = save_screenshot_to_file(
                        screenshot_uri, 
                        actual_rollout_id, 
                        turn_obj.id, 
                        obs_id=None
                    )
                    kwargs['screenshot_uri'] = screenshot_file_path
                else:
                    logger.warning(f"[RolloutRecorder] Could not determine rollout_id for screenshot saving")
            
            # Handle model_input serialization if needed
            if "model_input" in kwargs and isinstance(kwargs["model_input"], (dict, list)):
                from tinker_cookbook.recipes.cua_rl.database.database_dao import json_serialize
                kwargs["model_input_json"] = json_serialize(kwargs.pop("model_input"))
            
            # Create observation (rollout_id has been removed from kwargs)
            observation = create_observation(
                self.session,
                turn_id=turn_obj.id,
                **kwargs
            )
            
            self.session.commit()
            logger.info(f"[RolloutRecorder] Recorded observation for turn {turn_num} (Observation DB ID={observation.id})")
            return observation.id
            
        except Exception as e:
            logger.error(f"[RolloutRecorder] Failed to record observation for turn {turn_num}: {e}", exc_info=True)
            self.session.rollback()
            return None

