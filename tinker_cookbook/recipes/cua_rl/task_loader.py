"""
Task loader for CUA RL training.

This module provides flexible task loading from various sources:
- demo_tasks.py (DEMO_TRAINING_TASKS, DEMO_EVAL_TASKS)
- Custom task lists
- Task files
- Task IDs, categories, difficulties

Supports easy extension for new dataset sources.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from tinker_cookbook.recipes.cua_rl.demo_tasks import (
    CUATask,
    TaskDifficulty,
    TaskCategory,
    DEMO_TRAINING_TASKS,
    DEMO_EVAL_TASKS,
    get_task_by_id,
    get_tasks_by_category,
    get_tasks_by_difficulty,
    get_training_tasks,
    get_eval_tasks,
    get_all_tasks,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskSourceConfig:
    """Configuration for loading tasks from a source."""
    
    # Source type: "demo_training", "demo_eval", "demo_all", "ids", "file", "custom"
    source_type: str
    
    # For "ids": list of task IDs
    task_ids: Optional[List[str]] = None
    
    # For "demo_*" sources: filter by category
    category: Optional[Union[str, TaskCategory]] = None
    
    # For "demo_*" sources: filter by difficulty
    difficulty: Optional[Union[str, TaskDifficulty]] = None
    
    # For "file": path to task file
    file_path: Optional[str] = None
    
    # For "custom": list of task descriptions (strings)
    custom_tasks: Optional[List[str]] = None
    
    # Limit number of tasks (for sampling)
    limit: Optional[int] = None
    
    # Random seed for sampling
    seed: Optional[int] = None


def load_tasks_from_config(config: TaskSourceConfig) -> List[CUATask]:
    """
    Load tasks based on configuration.
    
    Returns a list of CUATask objects that can be used by CUADataset.
    
    Args:
        config: TaskSourceConfig specifying how to load tasks
        
    Returns:
        List of CUATask objects
    """
    tasks: List[CUATask] = []
    
    if config.source_type == "demo_training":
        tasks = get_training_tasks()
    elif config.source_type == "demo_eval":
        tasks = get_eval_tasks()
    elif config.source_type == "demo_all":
        tasks = get_all_tasks()
    elif config.source_type == "ids":
        if not config.task_ids:
            raise ValueError("task_ids must be provided for source_type='ids'")
        tasks = []
        for task_id in config.task_ids:
            task = get_task_by_id(task_id)
            if task is None:
                logger.warning(f"Task ID '{task_id}' not found, skipping")
                continue
            tasks.append(task)
    elif config.source_type == "file":
        if not config.file_path:
            raise ValueError("file_path must be provided for source_type='file'")
        # Load from file - each line is a task description
        # For file source, we create minimal CUATask objects with just descriptions
        # (no validation_query, so validation won't be performed)
        file_path = Path(config.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Task file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            task_descriptions = [line.strip() for line in f if line.strip()]
        # Create minimal CUATask objects (no validation for file-based tasks)
        tasks = [
            CUATask(
                id=f"file_task_{i}",
                name=f"File Task {i+1}",
                description=desc,
                difficulty=TaskDifficulty.MEDIUM,
                category=TaskCategory.SYSTEM,
                validation_query=None,  # No validation for file-based tasks
                expected_result=None,
            )
            for i, desc in enumerate(task_descriptions)
        ]
    elif config.source_type == "custom":
        if not config.custom_tasks:
            raise ValueError("custom_tasks must be provided for source_type='custom'")
        # Create minimal CUATask objects (no validation for custom tasks)
        tasks = [
            CUATask(
                id=f"custom_task_{i}",
                name=f"Custom Task {i+1}",
                description=desc,
                difficulty=TaskDifficulty.MEDIUM,
                category=TaskCategory.SYSTEM,
                validation_query=None,  # No validation for custom tasks
                expected_result=None,
            )
            for i, desc in enumerate(config.custom_tasks)
        ]
    else:
        raise ValueError(
            f"Unknown source_type: {config.source_type}. "
            "Supported: 'demo_training', 'demo_eval', 'demo_all', 'ids', 'file', 'custom'"
        )
    
    # Apply filters for demo sources
    if config.category is not None:
        category = (
            TaskCategory(config.category) if isinstance(config.category, str) else config.category
        )
        tasks = [t for t in tasks if t.category == category]
    
    if config.difficulty is not None:
        difficulty = (
            TaskDifficulty(config.difficulty) if isinstance(config.difficulty, str) else config.difficulty
        )
        tasks = [t for t in tasks if t.difficulty == difficulty]
    
    # Apply limit with optional sampling
    if config.limit is not None and config.limit < len(tasks):
        if config.seed is not None:
            import random
            rng = random.Random(config.seed)
            tasks = rng.sample(tasks, config.limit)
        else:
            tasks = tasks[:config.limit]
    
    logger.info(
        f"Loaded {len(tasks)} tasks from source_type='{config.source_type}'"
        + (f", category={config.category}" if config.category else "")
        + (f", difficulty={config.difficulty}" if config.difficulty else "")
        + (f", limit={config.limit}" if config.limit else "")
    )
    
    return tasks


def load_tasks_from_multiple_sources(configs: List[TaskSourceConfig]) -> List[CUATask]:
    """
    Load tasks from multiple sources and combine them.
    
    Args:
        configs: List of TaskSourceConfig objects
        
    Returns:
        Combined list of CUATask objects
    """
    all_tasks: List[CUATask] = []
    for config in configs:
        tasks = load_tasks_from_config(config)
        all_tasks.extend(tasks)
    
    logger.info(f"Loaded {len(all_tasks)} total tasks from {len(configs)} sources")
    return all_tasks


# Convenience functions for common use cases

def load_demo_training_tasks(
    category: Optional[Union[str, TaskCategory]] = None,
    difficulty: Optional[Union[str, TaskDifficulty]] = None,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[CUATask]:
    """Load training tasks from demo_tasks.py."""
    config = TaskSourceConfig(
        source_type="demo_training",
        category=category,
        difficulty=difficulty,
        limit=limit,
        seed=seed,
    )
    return load_tasks_from_config(config)


def load_demo_eval_tasks(
    category: Optional[Union[str, TaskCategory]] = None,
    difficulty: Optional[Union[str, TaskDifficulty]] = None,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[CUATask]:
    """Load evaluation tasks from demo_tasks.py."""
    config = TaskSourceConfig(
        source_type="demo_eval",
        category=category,
        difficulty=difficulty,
        limit=limit,
        seed=seed,
    )
    return load_tasks_from_config(config)


def load_tasks_by_ids(task_ids: List[str]) -> List[CUATask]:
    """Load tasks by their IDs."""
    config = TaskSourceConfig(source_type="ids", task_ids=task_ids)
    return load_tasks_from_config(config)


def load_tasks_from_file(file_path: str) -> List[CUATask]:
    """Load tasks from a file (one task description per line)."""
    config = TaskSourceConfig(source_type="file", file_path=file_path)
    return load_tasks_from_config(config)


__all__ = [
    "TaskSourceConfig",
    "load_tasks_from_config",
    "load_tasks_from_multiple_sources",
    "load_demo_training_tasks",
    "load_demo_eval_tasks",
    "load_tasks_by_ids",
    "load_tasks_from_file",
]

