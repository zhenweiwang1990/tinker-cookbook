"""
SQLAlchemy models for CUA RL training database.

This module defines all database models using SQLAlchemy ORM.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    CheckConstraint,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Training(Base):
    """Training session model."""
    __tablename__ = "training"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String, unique=True, nullable=False, index=True)
    log_path = Column(Text, nullable=False)
    model_name = Column(String, nullable=False)
    lora_rank = Column(Integer)
    learning_rate = Column(Float)
    batch_size = Column(Integer)
    group_size = Column(Integer)
    groups_per_batch = Column(Integer)
    max_tokens = Column(Integer)
    temperature = Column(Float)
    kl_penalty_coef = Column(Float)
    num_substeps = Column(Integer)
    max_turns = Column(Integer)
    seed = Column(Integer)
    box_type = Column(String)
    renderer_name = Column(String)
    wandb_project = Column(String)
    wandb_name = Column(String)
    status = Column(String, default="pending", index=True)
    progress_percent = Column(Float, default=0.0)
    current_step = Column(Integer)
    total_steps = Column(Integer)
    current_phase = Column(String)
    status_message = Column(Text)
    error_message = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    last_heartbeat = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    config_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    steps = relationship("Step", back_populates="training", cascade="all, delete-orphan")
    baselines = relationship("Baseline", back_populates="training", cascade="all, delete-orphan")
    evals = relationship("Eval", back_populates="training", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_training_status_heartbeat", "status", "last_heartbeat"),
    )


class Baseline(Base):
    """Baseline evaluation model."""
    __tablename__ = "baseline"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_id = Column(Integer, ForeignKey("training.id"), nullable=False, index=True)
    model_path = Column(Text, nullable=False)
    status = Column(String, default="pending", index=True)
    progress_percent = Column(Float, default=0.0)
    current_task_index = Column(Integer)
    total_tasks = Column(Integer)
    completed_tasks = Column(Integer)
    current_phase = Column(String)
    status_message = Column(Text)
    error_message = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    eval_time = Column(DateTime)
    success_rate = Column(Float)
    avg_reward = Column(Float)
    avg_turns = Column(Float)
    successful_tasks = Column(Integer)
    metrics_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training = relationship("Training", back_populates="baselines")
    groups = relationship("Group", back_populates="baseline", cascade="all, delete-orphan")
    rollouts = relationship("Rollout", back_populates="baseline", cascade="all, delete-orphan")


class Eval(Base):
    """Evaluation model."""
    __tablename__ = "eval"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_id = Column(Integer, ForeignKey("training.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    model_path = Column(Text, nullable=False)
    status = Column(String, default="pending", index=True)
    progress_percent = Column(Float, default=0.0)
    current_task_index = Column(Integer)
    total_tasks = Column(Integer)
    completed_tasks = Column(Integer)
    current_phase = Column(String)
    status_message = Column(Text)
    error_message = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    eval_time = Column(DateTime)
    success_rate = Column(Float)
    avg_reward = Column(Float)
    avg_turns = Column(Float)
    successful_tasks = Column(Integer)
    metrics_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training = relationship("Training", back_populates="evals")
    groups = relationship("Group", back_populates="eval", cascade="all, delete-orphan")
    rollouts = relationship("Rollout", back_populates="eval", cascade="all, delete-orphan")

    # Unique constraint
    __table_args__ = (
        UniqueConstraint("training_id", "step", name="uq_eval_training_step"),
        Index("idx_eval_training_step", "training_id", "step"),
    )


class Task(Base):
    """Task model."""
    __tablename__ = "task"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    difficulty = Column(String)
    category = Column(String)
    max_steps = Column(Integer)
    validation_type = Column(String)
    validation_query = Column(Text)
    expected_result = Column(Text)
    tags = Column(Text)  # JSON array
    prerequisites = Column(Text)  # JSON array
    app_name = Column(String)
    source_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    validators = relationship("Validator", back_populates="task", cascade="all, delete-orphan")
    rollouts = relationship("Rollout", back_populates="task")


class Validator(Base):
    """Validator model."""
    __tablename__ = "validator"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("task.id"), nullable=False, index=True)
    validator_type = Column(String, nullable=False)
    validation_query = Column(Text)
    validation_method = Column(String)
    config_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    task = relationship("Task", back_populates="validators")
    validations = relationship("Validation", back_populates="validator")


class Step(Base):
    """Training step model."""
    __tablename__ = "step"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_id = Column(Integer, ForeignKey("training.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    batch = Column(Integer)
    status = Column(String, default="pending", index=True)
    progress_percent = Column(Float, default=0.0)
    current_phase = Column(String)
    rollout_progress = Column(Text)  # JSON
    training_progress = Column(Text)  # JSON
    status_message = Column(Text)
    error_message = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    rollout_start_time = Column(DateTime)
    rollout_end_time = Column(DateTime)
    training_start_time = Column(DateTime)
    training_end_time = Column(DateTime)
    learning_rate = Column(Float)
    model_path = Column(Text)
    checkpoint_path = Column(Text)
    loss = Column(Float)
    kl_divergence = Column(Float)
    policy_gradient_norm = Column(Float)
    reward_mean = Column(Float)
    reward_std = Column(Float)
    num_trajectories = Column(Integer)
    num_tokens = Column(Integer)
    metrics_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training = relationship("Training", back_populates="steps")
    groups = relationship("Group", back_populates="step", cascade="all, delete-orphan")
    rollouts = relationship("Rollout", back_populates="step", cascade="all, delete-orphan")

    # Unique constraint
    __table_args__ = (
        UniqueConstraint("training_id", "step", name="uq_step_training_step"),
        Index("idx_step_training_step", "training_id", "step"),
    )


class Group(Base):
    """Rollout group model."""
    __tablename__ = "group"

    id = Column(Integer, primary_key=True, autoincrement=True)
    step_id = Column(Integer, ForeignKey("step.id"), index=True)
    eval_id = Column(Integer, ForeignKey("eval.id"), index=True)
    baseline_id = Column(Integer, ForeignKey("baseline.id"), index=True)
    group_num = Column(Integer, nullable=False)  # Group number within step/eval/baseline
    batch = Column(Integer)
    source_type = Column(String, nullable=False)  # 'step', 'eval', or 'baseline'
    status = Column(String, default="pending", index=True)
    progress_percent = Column(Float, default=0.0)
    current_phase = Column(String)
    status_message = Column(Text)
    error_message = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    num_rollouts = Column(Integer, default=0)
    completed_rollouts = Column(Integer, default=0)
    reward_mean = Column(Float)
    reward_std = Column(Float)
    success_count = Column(Integer, default=0)
    metrics_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    step = relationship("Step", back_populates="groups")
    eval = relationship("Eval", back_populates="groups")
    baseline = relationship("Baseline", back_populates="groups")
    rollouts = relationship("Rollout", back_populates="group", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_group_source_step", "source_type", "step_id", "group_num"),
        Index("idx_group_source_eval", "source_type", "eval_id", "group_num"),
        Index("idx_group_source_baseline", "source_type", "baseline_id", "group_num"),
        CheckConstraint(
            "(source_type = 'step' AND step_id IS NOT NULL AND eval_id IS NULL AND baseline_id IS NULL) OR "
            "(source_type = 'eval' AND eval_id IS NOT NULL AND step_id IS NULL AND baseline_id IS NULL) OR "
            "(source_type = 'baseline' AND baseline_id IS NOT NULL AND step_id IS NULL AND eval_id IS NULL)",
            name="check_group_source"
        ),
    )


class Rollout(Base):
    """Rollout model."""
    __tablename__ = "rollout"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String, nullable=False)  # 'step', 'eval', or 'baseline'
    step_id = Column(Integer, ForeignKey("step.id"), index=True)
    eval_id = Column(Integer, ForeignKey("eval.id"), index=True)
    baseline_id = Column(Integer, ForeignKey("baseline.id"), index=True)
    group_id = Column(Integer, ForeignKey("group.id"), index=True)  # Reference to group table
    rollout_id = Column(String, unique=True, nullable=False, index=True)
    batch = Column(Integer)
    group_num = Column(Integer)  # Group number (kept for backward compatibility, renamed from 'group' to avoid conflict)
    env_index = Column(Integer)
    task_id = Column(Integer, ForeignKey("task.id"), nullable=False, index=True)
    model_path = Column(Text, nullable=False)
    is_eval = Column(Boolean, default=False)
    status = Column(String, default="pending", index=True)
    progress_percent = Column(Float, default=0.0)
    current_phase = Column(String)
    current_turn = Column(Integer)
    status_message = Column(Text)
    error_message = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    env_creation_time = Column(DateTime)
    agent_init_time = Column(DateTime)
    task_start_time = Column(DateTime)
    task_end_time = Column(DateTime)
    validation_time = Column(DateTime)
    rollout_time = Column(Float)
    task_completed = Column(Boolean)
    task_success = Column(Boolean)
    agent_reported_success = Column(Boolean)
    validation_passed = Column(Boolean)
    num_turns = Column(Integer)
    max_turns = Column(Integer)
    reward = Column(Float)
    temperature = Column(Float)
    num_total_actions = Column(Integer)
    consecutive_repeated_actions = Column(Integer)
    parse_errors = Column(Integer)
    tool_name_errors = Column(Integer)
    tool_arg_errors = Column(Integer)
    runtime_errors = Column(Integer)
    ran_out_of_turns = Column(Boolean)
    attempted_completion = Column(Boolean)
    turn_first_success = Column(Integer)
    turn_task_completed = Column(Integer)
    errors = Column(Text)  # JSON array
    summary_json = Column(Text)
    trajectory_path = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    step = relationship("Step", back_populates="rollouts")
    eval = relationship("Eval", back_populates="rollouts")
    baseline = relationship("Baseline", back_populates="rollouts")
    group = relationship("Group", back_populates="rollouts")
    task = relationship("Task", back_populates="rollouts")
    turns = relationship("Turn", back_populates="rollout", cascade="all, delete-orphan")
    validation = relationship("Validation", back_populates="rollout", uselist=False, cascade="all, delete-orphan")
    environment = relationship("Environment", back_populates="rollout", uselist=False, cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_rollout_source_step", "source_type", "step_id"),
        Index("idx_rollout_source_eval", "source_type", "eval_id"),
        Index("idx_rollout_source_baseline", "source_type", "baseline_id"),
        CheckConstraint(
            "(source_type = 'step' AND step_id IS NOT NULL AND eval_id IS NULL AND baseline_id IS NULL) OR "
            "(source_type = 'eval' AND eval_id IS NOT NULL AND step_id IS NULL AND baseline_id IS NULL) OR "
            "(source_type = 'baseline' AND baseline_id IS NOT NULL AND step_id IS NULL AND eval_id IS NULL)",
            name="check_rollout_source"
        ),
    )


class Turn(Base):
    """Turn model."""
    __tablename__ = "turn"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rollout_id = Column(Integer, ForeignKey("rollout.id"), nullable=False, index=True)
    turn = Column(Integer, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    turn_time = Column(Float)
    reward = Column(Float)
    episode_done = Column(Boolean)
    metrics_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    rollout = relationship("Rollout", back_populates="turns")
    actions = relationship("Action", back_populates="turn", cascade="all, delete-orphan")
    observations = relationship("Observation", back_populates="turn", cascade="all, delete-orphan")

    # Unique constraint
    __table_args__ = (
        UniqueConstraint("rollout_id", "turn", name="uq_turn_rollout_turn"),
        Index("idx_turn_rollout_turn", "rollout_id", "turn"),
    )


class Action(Base):
    """Action model."""
    __tablename__ = "action"

    id = Column(Integer, primary_key=True, autoincrement=True)
    turn_id = Column(Integer, ForeignKey("turn.id"), nullable=False, index=True)
    action_type = Column(String)
    tool_name = Column(String)
    tool_args = Column(Text)  # JSON
    tokens = Column(Text)  # JSON array
    logprobs = Column(Text)  # JSON array
    num_tokens = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    turn = relationship("Turn", back_populates="actions")


class Observation(Base):
    """Observation model."""
    __tablename__ = "obs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    turn_id = Column(Integer, ForeignKey("turn.id"), nullable=False, index=True)
    obs_type = Column(String)
    screenshot_uri = Column(Text)
    text_content = Column(Text)
    model_input_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    turn = relationship("Turn", back_populates="observations")


class Validation(Base):
    """Validation model."""
    __tablename__ = "validation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rollout_id = Column(Integer, ForeignKey("rollout.id"), nullable=False, index=True)
    validator_id = Column(Integer, ForeignKey("validator.id"), index=True)
    validation_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    validation_query = Column(Text)
    expected_result = Column(Text)
    actual_result = Column(Text)
    success = Column(Boolean, nullable=False)
    execution_time = Column(Float)
    error_message = Column(Text)
    details_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    rollout = relationship("Rollout", back_populates="validation")
    validator = relationship("Validator", back_populates="validations")


class Environment(Base):
    """Environment model."""
    __tablename__ = "environment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rollout_id = Column(Integer, ForeignKey("rollout.id"), nullable=False, index=True)
    env_type = Column(String, nullable=False)
    status = Column(String, default="pending", index=True)
    gbox_id = Column(String)
    box_type = Column(String)
    creation_time = Column(DateTime)
    termination_time = Column(DateTime)
    status_message = Column(Text)
    error_message = Column(Text)
    config_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    rollout = relationship("Rollout", back_populates="environment")


class StatusHistory(Base):
    """Status history model."""
    __tablename__ = "status_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String, nullable=False)  # 'training', 'baseline', 'eval', 'step', 'rollout', 'environment'
    entity_id = Column(Integer, nullable=False)
    old_status = Column(String)
    new_status = Column(String, nullable=False)
    progress_percent = Column(Float)
    status_message = Column(Text)
    metadata_json = Column(Text)
    changed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("idx_status_history_entity", "entity_type", "entity_id"),
        Index("idx_status_history_entity_time", "entity_type", "entity_id", "changed_at"),
    )

