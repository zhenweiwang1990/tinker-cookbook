"""
Database Data Access Object (DAO) layer for CUA RL training.

This module provides high-level CRUD operations for all database models.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from tinker_cookbook.recipes.cua_rl.database import get_session, json_serialize, json_deserialize
from tinker_cookbook.recipes.cua_rl.database_models import (
    Action,
    Baseline,
    Environment,
    Eval,
    Group,
    Observation,
    Rollout,
    StatusHistory,
    Step,
    Task,
    Training,
    Turn,
    Validation,
    Validator,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Training DAO
# ============================================================================

def create_training(session: Session, **kwargs) -> Training:
    """Create a new training session."""
    # Serialize JSON fields
    if "config_json" in kwargs and isinstance(kwargs["config_json"], (dict, list)):
        kwargs["config_json"] = json_serialize(kwargs["config_json"])
    
    training = Training(**kwargs)
    session.add(training)
    session.flush()
    
    # Record status history
    record_status_change(
        session,
        entity_type="training",
        entity_id=training.id,
        new_status=training.status or "pending",
    )
    
    return training


def get_training(session: Session, training_id: int) -> Optional[Training]:
    """Get a training session by ID."""
    return session.query(Training).filter(Training.id == training_id).first()


def get_training_by_run_name(session: Session, run_name: str) -> Optional[Training]:
    """Get a training session by run name."""
    return session.query(Training).filter(Training.run_name == run_name).first()


def update_training(session: Session, training_id: int, **kwargs) -> Optional[Training]:
    """Update a training session."""
    training = get_training(session, training_id)
    if not training:
        return None
    
    old_status = training.status
    
    # Serialize JSON fields
    if "config_json" in kwargs and isinstance(kwargs["config_json"], (dict, list)):
        kwargs["config_json"] = json_serialize(kwargs["config_json"])
    
    for key, value in kwargs.items():
        if hasattr(training, key):
            setattr(training, key, value)
    
    training.updated_at = datetime.utcnow()
    session.flush()
    
    # Record status change if status changed
    if "status" in kwargs and kwargs["status"] != old_status:
        record_status_change(
            session,
            entity_type="training",
            entity_id=training_id,
            old_status=old_status,
            new_status=kwargs["status"],
            progress_percent=kwargs.get("progress_percent"),
            status_message=kwargs.get("status_message"),
        )
    
    return training


def list_trainings(session: Session, status: Optional[str] = None) -> List[Training]:
    """List all training sessions, optionally filtered by status."""
    query = session.query(Training)
    if status:
        query = query.filter(Training.status == status)
    return query.order_by(Training.created_at.desc()).all()


# ============================================================================
# Task DAO
# ============================================================================

def create_or_get_task(session: Session, task_id: str, **kwargs) -> Task:
    """Create a task or get existing one by task_id."""
    task = session.query(Task).filter(Task.task_id == task_id).first()
    if task:
        # Update existing task
        for key, value in kwargs.items():
            if hasattr(task, key) and value is not None:
                setattr(task, key, value)
        task.updated_at = datetime.utcnow()
        return task
    
    # Create new task
    # Serialize JSON fields
    for field in ["tags", "prerequisites"]:
        if field in kwargs and isinstance(kwargs[field], (list, dict)):
            kwargs[field] = json_serialize(kwargs[field])
    
    task = Task(task_id=task_id, **kwargs)
    session.add(task)
    session.flush()
    return task


def get_task(session: Session, task_id: int) -> Optional[Task]:
    """Get a task by ID."""
    return session.query(Task).filter(Task.id == task_id).first()


def get_task_by_task_id(session: Session, task_id: str) -> Optional[Task]:
    """Get a task by task_id string."""
    return session.query(Task).filter(Task.task_id == task_id).first()


def list_tasks(session: Session) -> List[Task]:
    """List all tasks."""
    return session.query(Task).all()


# ============================================================================
# Validator DAO
# ============================================================================

def create_or_get_validator(
    session: Session,
    task_id: int,
    validator_type: str,
    **kwargs
) -> Validator:
    """Create a validator or get existing one."""
    validator = (
        session.query(Validator)
        .filter(
            and_(
                Validator.task_id == task_id,
                Validator.validator_type == validator_type,
            )
        )
        .first()
    )
    
    if validator:
        # Update existing validator
        for key, value in kwargs.items():
            if hasattr(validator, key) and value is not None:
                setattr(validator, key, value)
        return validator
    
    # Create new validator
    if "config_json" in kwargs and isinstance(kwargs["config_json"], (dict, list)):
        kwargs["config_json"] = json_serialize(kwargs["config_json"])
    
    validator = Validator(task_id=task_id, validator_type=validator_type, **kwargs)
    session.add(validator)
    session.flush()
    return validator


def get_validator(session: Session, validator_id: int) -> Optional[Validator]:
    """Get a validator by ID."""
    return session.query(Validator).filter(Validator.id == validator_id).first()


def list_validators_by_task(session: Session, task_id: int) -> List[Validator]:
    """List all validators for a task."""
    return session.query(Validator).filter(Validator.task_id == task_id).all()


# ============================================================================
# Step DAO
# ============================================================================

def create_step(session: Session, training_id: int, step: int, **kwargs) -> Step:
    """Create a new training step."""
    # Serialize JSON fields
    for field in ["rollout_progress", "training_progress", "metrics_json"]:
        if field in kwargs and isinstance(kwargs[field], (dict, list)):
            kwargs[field] = json_serialize(kwargs[field])
    
    step_obj = Step(training_id=training_id, step=step, **kwargs)
    session.add(step_obj)
    session.flush()
    
    # Record status history
    record_status_change(
        session,
        entity_type="step",
        entity_id=step_obj.id,
        new_status=step_obj.status or "pending",
    )
    
    return step_obj


def get_step(session: Session, step_id: int) -> Optional[Step]:
    """Get a step by ID."""
    return session.query(Step).filter(Step.id == step_id).first()


def get_step_by_training_and_step(
    session: Session,
    training_id: int,
    step: int
) -> Optional[Step]:
    """Get a step by training_id and step number."""
    return (
        session.query(Step)
        .filter(and_(Step.training_id == training_id, Step.step == step))
        .first()
    )


def update_step(session: Session, step_id: int, **kwargs) -> Optional[Step]:
    """Update a training step."""
    step_obj = get_step(session, step_id)
    if not step_obj:
        return None
    
    old_status = step_obj.status
    
    # Serialize JSON fields
    for field in ["rollout_progress", "training_progress", "metrics_json"]:
        if field in kwargs and isinstance(kwargs[field], (dict, list)):
            kwargs[field] = json_serialize(kwargs[field])
    
    for key, value in kwargs.items():
        if hasattr(step_obj, key):
            setattr(step_obj, key, value)
    
    step_obj.updated_at = datetime.utcnow()
    session.flush()
    
    # Record status change
    if "status" in kwargs and kwargs["status"] != old_status:
        record_status_change(
            session,
            entity_type="step",
            entity_id=step_id,
            old_status=old_status,
            new_status=kwargs["status"],
            progress_percent=kwargs.get("progress_percent"),
            status_message=kwargs.get("status_message"),
        )
    
    return step_obj


def list_steps_by_training(session: Session, training_id: int) -> List[Step]:
    """List all steps for a training session."""
    return (
        session.query(Step)
        .filter(Step.training_id == training_id)
        .order_by(Step.step.asc())
        .all()
    )


def get_latest_completed_step(session: Session, training_id: int) -> Optional[Step]:
    """Get the latest completed step for a training session."""
    return (
        session.query(Step)
        .filter(
            and_(
                Step.training_id == training_id,
                Step.status == "completed",
                Step.checkpoint_path.isnot(None),  # Must have checkpoint_path to resume
            )
        )
        .order_by(Step.step.desc())
        .first()
    )


# ============================================================================
# Rollout DAO
# ============================================================================

def create_rollout(
    session: Session,
    source_type: str,
    rollout_id: str,
    task_id: int,
    model_path: str,
    env_id: int,  # Required: Environment must be created first
    **kwargs
) -> Rollout:
    """Create a new rollout. Environment must be created first and passed via env_id."""
    # Validate source_type and set appropriate ID
    if source_type == "step":
        if "step_id" not in kwargs:
            raise ValueError("step_id required when source_type='step'")
        kwargs["eval_id"] = None
        kwargs["baseline_id"] = None
    elif source_type == "eval":
        if "eval_id" not in kwargs:
            raise ValueError("eval_id required when source_type='eval'")
        kwargs["step_id"] = None
        kwargs["baseline_id"] = None
    elif source_type == "baseline":
        if "baseline_id" not in kwargs:
            raise ValueError("baseline_id required when source_type='baseline'")
        kwargs["step_id"] = None
        kwargs["eval_id"] = None
    else:
        raise ValueError(f"Invalid source_type: {source_type}. Must be 'step', 'eval', or 'baseline'")
    
    # Serialize JSON fields
    for field in ["errors", "summary_json"]:
        if field in kwargs and isinstance(kwargs[field], (dict, list)):
            kwargs[field] = json_serialize(kwargs[field])
    
    rollout = Rollout(
        source_type=source_type,
        rollout_id=rollout_id,
        task_id=task_id,
        model_path=model_path,
        env_id=env_id,  # Required field
        **kwargs
    )
    session.add(rollout)
    session.flush()
    
    # Record status history
    record_status_change(
        session,
        entity_type="rollout",
        entity_id=rollout.id,
        new_status=rollout.status or "pending",
    )
    
    return rollout


def get_rollout(session: Session, rollout_id: int) -> Optional[Rollout]:
    """Get a rollout by ID."""
    return session.query(Rollout).filter(Rollout.id == rollout_id).first()


def get_rollout_by_rollout_id(session: Session, rollout_id: str) -> Optional[Rollout]:
    """
    Get a rollout by rollout_id string.
    
    NOTE: rollout_id is now a unique UUID string for reliable tracking.
    """
    return session.query(Rollout).filter(Rollout.rollout_id == rollout_id).first()


def update_rollout(session: Session, rollout_id: int, **kwargs) -> Optional[Rollout]:
    """Update a rollout."""
    rollout = get_rollout(session, rollout_id)
    if not rollout:
        return None
    
    old_status = rollout.status
    
    # Serialize JSON fields
    for field in ["errors", "summary_json", "trajectory_data_json"]:
        if field in kwargs:
            if isinstance(kwargs[field], (dict, list)):
                kwargs[field] = json_serialize(kwargs[field])
            elif kwargs[field] is not None and not isinstance(kwargs[field], str):
                # If it's not a string and not None, try to serialize
                kwargs[field] = json_serialize(kwargs[field])
    
    for key, value in kwargs.items():
        if hasattr(rollout, key):
            setattr(rollout, key, value)
    
    rollout.updated_at = datetime.utcnow()
    session.flush()
    
    # Record status change
    if "status" in kwargs and kwargs["status"] != old_status:
        record_status_change(
            session,
            entity_type="rollout",
            entity_id=rollout_id,
            old_status=old_status,
            new_status=kwargs["status"],
            progress_percent=kwargs.get("progress_percent"),
            status_message=kwargs.get("status_message"),
        )
    
    return rollout


def list_rollouts_by_step(session: Session, step_id: int) -> List[Rollout]:
    """List all rollouts for a step."""
    return (
        session.query(Rollout)
        .filter(and_(Rollout.source_type == "step", Rollout.step_id == step_id))
        .all()
    )


def list_rollouts_by_eval(session: Session, eval_id: int) -> List[Rollout]:
    """List all rollouts for an eval."""
    return (
        session.query(Rollout)
        .filter(and_(Rollout.source_type == "eval", Rollout.eval_id == eval_id))
        .all()
    )


def list_rollouts_by_baseline(session: Session, baseline_id: int) -> List[Rollout]:
    """List all rollouts for a baseline."""
    return (
        session.query(Rollout)
        .filter(and_(Rollout.source_type == "baseline", Rollout.baseline_id == baseline_id))
        .all()
    )


def list_rollouts_by_task(session: Session, task_id: int) -> List[Rollout]:
    """List all rollouts for a task."""
    return session.query(Rollout).filter(Rollout.task_id == task_id).all()


def list_rollouts_by_group(session: Session, group_id: int) -> List[Rollout]:
    """List all rollouts for a group."""
    return session.query(Rollout).filter(Rollout.group_id == group_id).all()


# ============================================================================
# Group DAO
# ============================================================================

def create_group(
    session: Session,
    source_type: str,
    group_num: int,
    step_id: Optional[int] = None,
    eval_id: Optional[int] = None,
    baseline_id: Optional[int] = None,
    **kwargs
) -> Group:
    """Create a new group."""
    # Validate source_type and set appropriate ID
    if source_type == "step":
        if step_id is None:
            raise ValueError("step_id required when source_type='step'")
        eval_id = None
        baseline_id = None
    elif source_type == "eval":
        if eval_id is None:
            raise ValueError("eval_id required when source_type='eval'")
        step_id = None
        baseline_id = None
    elif source_type == "baseline":
        if baseline_id is None:
            raise ValueError("baseline_id required when source_type='baseline'")
        step_id = None
        eval_id = None
    else:
        raise ValueError(f"Invalid source_type: {source_type}. Must be 'step', 'eval', or 'baseline'")
    
    # Serialize JSON fields
    if "metrics_json" in kwargs and isinstance(kwargs["metrics_json"], (dict, list)):
        kwargs["metrics_json"] = json_serialize(kwargs["metrics_json"])
    
    group = Group(
        source_type=source_type,
        group_num=group_num,
        step_id=step_id,
        eval_id=eval_id,
        baseline_id=baseline_id,
        status="pending",
        start_time=datetime.utcnow(),
        **kwargs
    )
    session.add(group)
    session.flush()
    
    # Record status history
    record_status_change(
        session,
        entity_type="group",
        entity_id=group.id,
        new_status=group.status or "pending",
    )
    
    return group


def get_group(session: Session, group_id: int) -> Optional[Group]:
    """Get a group by ID."""
    return session.query(Group).filter(Group.id == group_id).first()


def get_or_create_group(
    session: Session,
    source_type: str,
    group_num: int,
    step_id: Optional[int] = None,
    eval_id: Optional[int] = None,
    baseline_id: Optional[int] = None,
    **kwargs
) -> Group:
    """Get or create a group."""
    # Try to find existing group
    query = session.query(Group).filter(
        Group.source_type == source_type,
        Group.group_num == group_num
    )
    
    if source_type == "step" and step_id is not None:
        query = query.filter(Group.step_id == step_id)
    elif source_type == "eval" and eval_id is not None:
        query = query.filter(Group.eval_id == eval_id)
    elif source_type == "baseline" and baseline_id is not None:
        query = query.filter(Group.baseline_id == baseline_id)
    
    existing_group = query.first()
    if existing_group:
        return existing_group
    
    # Create new group
    return create_group(
        session,
        source_type=source_type,
        group_num=group_num,
        step_id=step_id,
        eval_id=eval_id,
        baseline_id=baseline_id,
        **kwargs
    )


def update_group(session: Session, group_id: int, **kwargs) -> Optional[Group]:
    """Update a group."""
    group = get_group(session, group_id)
    if not group:
        return None
    
    old_status = group.status
    
    # Serialize JSON fields
    if "metrics_json" in kwargs and isinstance(kwargs["metrics_json"], (dict, list)):
        kwargs["metrics_json"] = json_serialize(kwargs["metrics_json"])
    
    for key, value in kwargs.items():
        if hasattr(group, key):
            setattr(group, key, value)
    
    group.updated_at = datetime.utcnow()
    session.flush()
    
    # Record status change
    if "status" in kwargs and kwargs["status"] != old_status:
        record_status_change(
            session,
            entity_type="group",
            entity_id=group_id,
            old_status=old_status,
            new_status=kwargs["status"],
            progress_percent=kwargs.get("progress_percent"),
            status_message=kwargs.get("status_message"),
        )
    
    return group


def list_groups_by_step(session: Session, step_id: int) -> List[Group]:
    """List all groups for a step."""
    return (
        session.query(Group)
        .filter(and_(Group.source_type == "step", Group.step_id == step_id))
        .order_by(Group.group_num.asc())
        .all()
    )


def list_groups_by_eval(session: Session, eval_id: int) -> List[Group]:
    """List all groups for an eval."""
    return (
        session.query(Group)
        .filter(and_(Group.source_type == "eval", Group.eval_id == eval_id))
        .order_by(Group.group_num.asc())
        .all()
    )


def list_groups_by_baseline(session: Session, baseline_id: int) -> List[Group]:
    """List all groups for a baseline."""
    return (
        session.query(Group)
        .filter(and_(Group.source_type == "baseline", Group.baseline_id == baseline_id))
        .order_by(Group.group_num.asc())
        .all()
    )


# ============================================================================
# Turn DAO
# ============================================================================

def create_turn(
    session: Session,
    rollout_id: int,
    turn: int,
    **kwargs
) -> Turn:
    """Create a new turn."""
    if "metrics_json" in kwargs and isinstance(kwargs["metrics_json"], (dict, list)):
        kwargs["metrics_json"] = json_serialize(kwargs["metrics_json"])
    
    turn_obj = Turn(rollout_id=rollout_id, turn=turn, **kwargs)
    session.add(turn_obj)
    session.flush()
    return turn_obj


def get_turn(session: Session, turn_id: int) -> Optional[Turn]:
    """Get a turn by ID."""
    return session.query(Turn).filter(Turn.id == turn_id).first()


def get_turn_by_rollout_and_turn(
    session: Session,
    rollout_id: int,
    turn: int
) -> Optional[Turn]:
    """Get a turn by rollout_id and turn number."""
    return (
        session.query(Turn)
        .filter(and_(Turn.rollout_id == rollout_id, Turn.turn == turn))
        .first()
    )


def update_turn(session: Session, turn_id: int, **kwargs) -> Optional[Turn]:
    """Update a turn."""
    turn_obj = get_turn(session, turn_id)
    if not turn_obj:
        return None
    
    if "metrics_json" in kwargs and isinstance(kwargs["metrics_json"], (dict, list)):
        kwargs["metrics_json"] = json_serialize(kwargs["metrics_json"])
    
    for key, value in kwargs.items():
        if hasattr(turn_obj, key):
            setattr(turn_obj, key, value)
    
    return turn_obj


def list_turns_by_rollout(session: Session, rollout_id: int) -> List[Turn]:
    """List all turns for a rollout."""
    return (
        session.query(Turn)
        .filter(Turn.rollout_id == rollout_id)
        .order_by(Turn.turn.asc())
        .all()
    )


# ============================================================================
# Action DAO
# ============================================================================

def create_action(session: Session, turn_id: int, **kwargs) -> Action:
    """Create a new action."""
    # Serialize JSON fields
    for field in ["tool_args", "tokens", "logprobs"]:
        if field in kwargs and isinstance(kwargs[field], (list, dict)):
            kwargs[field] = json_serialize(kwargs[field])
    
    action = Action(turn_id=turn_id, **kwargs)
    session.add(action)
    session.flush()
    return action


def list_actions_by_turn(session: Session, turn_id: int) -> List[Action]:
    """List all actions for a turn."""
    return session.query(Action).filter(Action.turn_id == turn_id).all()


# ============================================================================
# Observation DAO
# ============================================================================

def create_observation(session: Session, turn_id: int, **kwargs) -> Observation:
    """Create a new observation."""
    if "model_input_json" in kwargs and isinstance(kwargs["model_input_json"], (dict, list)):
        kwargs["model_input_json"] = json_serialize(kwargs["model_input_json"])
    
    observation = Observation(turn_id=turn_id, **kwargs)
    session.add(observation)
    session.flush()
    return observation


def list_observations_by_turn(session: Session, turn_id: int) -> List[Observation]:
    """List all observations for a turn."""
    return session.query(Observation).filter(Observation.turn_id == turn_id).all()


# ============================================================================
# Validation DAO
# ============================================================================

def create_validation(
    session: Session,
    rollout_id: int,
    success: bool,
    **kwargs
) -> Validation:
    """Create a new validation result."""
    if "details_json" in kwargs and isinstance(kwargs["details_json"], (dict, list)):
        kwargs["details_json"] = json_serialize(kwargs["details_json"])
    
    validation = Validation(rollout_id=rollout_id, success=success, **kwargs)
    session.add(validation)
    session.flush()
    return validation


def get_validation_by_rollout(session: Session, rollout_id: int) -> Optional[Validation]:
    """Get validation result for a rollout."""
    return session.query(Validation).filter(Validation.rollout_id == rollout_id).first()


# ============================================================================
# Environment DAO
# ============================================================================

def create_environment(session: Session, env_type: str, **kwargs) -> Environment:
    """Create a new environment. Environment is created before Rollout, and Rollout references it via env_id."""
    if "config_json" in kwargs and isinstance(kwargs["config_json"], (dict, list)):
        kwargs["config_json"] = json_serialize(kwargs["config_json"])
    
    env = Environment(env_type=env_type, **kwargs)
    session.add(env)
    session.flush()
    
    # Record status history
    record_status_change(
        session,
        entity_type="environment",
        entity_id=env.id,
        new_status=env.status or "pending",
    )
    
    return env


def get_environment_by_rollout(session: Session, rollout_id: int) -> Optional[Environment]:
    """Get environment for a rollout."""
    from tinker_cookbook.recipes.cua_rl.database_models import Rollout
    rollout = session.query(Rollout).filter(Rollout.id == rollout_id).first()
    if rollout and rollout.env_id:
        return session.query(Environment).filter(Environment.id == rollout.env_id).first()
    return None


def update_environment(session: Session, env_id: int, **kwargs) -> Optional[Environment]:
    """Update an environment."""
    env = session.query(Environment).filter(Environment.id == env_id).first()
    if not env:
        return None
    
    old_status = env.status
    
    if "config_json" in kwargs and isinstance(kwargs["config_json"], (dict, list)):
        kwargs["config_json"] = json_serialize(kwargs["config_json"])
    
    for key, value in kwargs.items():
        if hasattr(env, key):
            setattr(env, key, value)
    
    env.updated_at = datetime.utcnow()
    session.flush()
    
    # Record status change
    if "status" in kwargs and kwargs["status"] != old_status:
        record_status_change(
            session,
            entity_type="environment",
            entity_id=env_id,
            old_status=old_status,
            new_status=kwargs["status"],
            status_message=kwargs.get("status_message"),
        )
    
    return env


# ============================================================================
# Baseline DAO
# ============================================================================

def create_baseline(session: Session, training_id: int, model_path: str, **kwargs) -> Baseline:
    """Create a new baseline evaluation."""
    if "metrics_json" in kwargs and isinstance(kwargs["metrics_json"], (dict, list)):
        kwargs["metrics_json"] = json_serialize(kwargs["metrics_json"])
    
    baseline = Baseline(training_id=training_id, model_path=model_path, **kwargs)
    session.add(baseline)
    session.flush()
    
    # Record status history
    record_status_change(
        session,
        entity_type="baseline",
        entity_id=baseline.id,
        new_status=baseline.status or "pending",
    )
    
    return baseline


def get_baseline(session: Session, baseline_id: int) -> Optional[Baseline]:
    """Get a baseline by ID."""
    return session.query(Baseline).filter(Baseline.id == baseline_id).first()


def update_baseline(session: Session, baseline_id: int, **kwargs) -> Optional[Baseline]:
    """Update a baseline evaluation."""
    baseline = get_baseline(session, baseline_id)
    if not baseline:
        return None
    
    old_status = baseline.status
    
    if "metrics_json" in kwargs and isinstance(kwargs["metrics_json"], (dict, list)):
        kwargs["metrics_json"] = json_serialize(kwargs["metrics_json"])
    
    for key, value in kwargs.items():
        if hasattr(baseline, key):
            setattr(baseline, key, value)
    
    baseline.updated_at = datetime.utcnow()
    session.flush()
    
    # Record status change
    if "status" in kwargs and kwargs["status"] != old_status:
        record_status_change(
            session,
            entity_type="baseline",
            entity_id=baseline_id,
            old_status=old_status,
            new_status=kwargs["status"],
            progress_percent=kwargs.get("progress_percent"),
            status_message=kwargs.get("status_message"),
        )
    
    return baseline


def list_baselines_by_training(session: Session, training_id: int) -> List[Baseline]:
    """List all baselines for a training session."""
    return session.query(Baseline).filter(Baseline.training_id == training_id).all()


# ============================================================================
# Eval DAO
# ============================================================================

def create_eval(
    session: Session,
    training_id: int,
    step: int,
    model_path: str,
    **kwargs
) -> Eval:
    """Create a new evaluation."""
    if "metrics_json" in kwargs and isinstance(kwargs["metrics_json"], (dict, list)):
        kwargs["metrics_json"] = json_serialize(kwargs["metrics_json"])
    
    eval_obj = Eval(training_id=training_id, step=step, model_path=model_path, **kwargs)
    session.add(eval_obj)
    session.flush()
    
    # Record status history
    record_status_change(
        session,
        entity_type="eval",
        entity_id=eval_obj.id,
        new_status=eval_obj.status or "pending",
    )
    
    return eval_obj


def get_eval(session: Session, eval_id: int) -> Optional[Eval]:
    """Get an eval by ID."""
    return session.query(Eval).filter(Eval.id == eval_id).first()


def get_eval_by_training_and_step(
    session: Session,
    training_id: int,
    step: int
) -> Optional[Eval]:
    """Get an eval by training_id and step."""
    return (
        session.query(Eval)
        .filter(and_(Eval.training_id == training_id, Eval.step == step))
        .first()
    )


def update_eval(session: Session, eval_id: int, **kwargs) -> Optional[Eval]:
    """Update an evaluation."""
    eval_obj = get_eval(session, eval_id)
    if not eval_obj:
        return None
    
    old_status = eval_obj.status
    
    if "metrics_json" in kwargs and isinstance(kwargs["metrics_json"], (dict, list)):
        kwargs["metrics_json"] = json_serialize(kwargs["metrics_json"])
    
    for key, value in kwargs.items():
        if hasattr(eval_obj, key):
            setattr(eval_obj, key, value)
    
    eval_obj.updated_at = datetime.utcnow()
    session.flush()
    
    # Record status change
    if "status" in kwargs and kwargs["status"] != old_status:
        record_status_change(
            session,
            entity_type="eval",
            entity_id=eval_id,
            old_status=old_status,
            new_status=kwargs["status"],
            progress_percent=kwargs.get("progress_percent"),
            status_message=kwargs.get("status_message"),
        )
    
    return eval_obj


def list_evals_by_training(session: Session, training_id: int) -> List[Eval]:
    """List all evals for a training session."""
    return (
        session.query(Eval)
        .filter(Eval.training_id == training_id)
        .order_by(Eval.step.asc())
        .all()
    )


# ============================================================================
# Status History DAO
# ============================================================================

def record_status_change(
    session: Session,
    entity_type: str,
    entity_id: int,
    new_status: str,
    old_status: Optional[str] = None,
    progress_percent: Optional[float] = None,
    status_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> StatusHistory:
    """Record a status change in the status history."""
    metadata_json = json_serialize(metadata) if metadata else None
    
    history = StatusHistory(
        entity_type=entity_type,
        entity_id=entity_id,
        old_status=old_status,
        new_status=new_status,
        progress_percent=progress_percent,
        status_message=status_message,
        metadata_json=metadata_json,
    )
    session.add(history)
    session.flush()
    return history


def get_status_history(
    session: Session,
    entity_type: str,
    entity_id: int,
    limit: Optional[int] = None
) -> List[StatusHistory]:
    """Get status history for an entity."""
    query = (
        session.query(StatusHistory)
        .filter(
            and_(
                StatusHistory.entity_type == entity_type,
                StatusHistory.entity_id == entity_id,
            )
        )
        .order_by(StatusHistory.changed_at.desc())
    )
    
    if limit:
        query = query.limit(limit)
    
    return query.all()

