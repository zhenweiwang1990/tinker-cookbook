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
from tinker_cookbook.recipes.cua_rl.tasks.task_adapter import (
    TaskAdapter,
    get_tasks_train_eval,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskSourceConfig:
    """Configuration for loading tasks from a source."""
    
    # Source type: "demo_training", "demo_eval", "demo_all", "ids", "file", "custom", "task_adapter"
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
    
    # For "task_adapter": configuration
    tasks_dir: Optional[str] = None  # Path to tasks directory (default: auto-detect)
    train_ratio: float = 0.8  # Ratio for train/eval split (only used if split_type is specified)
    split_type: Optional[str] = None  # "train" or "eval" - which split to use. If None, uses all tasks.
    category: Optional[str] = None  # Filter by category (e.g., "demo", "airbnb", "instagram")
    task_names: Optional[str] = None  # Filter by task names (comma-separated, e.g., "12_enable_battery_saver,06_min_brightness")
    
    # Limit number of tasks (for sampling)
    limit: Optional[int] = None
    
    # Random seed for sampling and splitting
    seed: Optional[int] = 42


def load_tasks_from_config(
    config: TaskSourceConfig,
    save_to_db: bool = True,
    db_session = None,
) -> List[CUATask]:
    """
    Load tasks based on configuration.
    
    Returns a list of CUATask objects that can be used by CUADataset.
    
    Args:
        config: TaskSourceConfig specifying how to load tasks
        save_to_db: If True, save tasks to database (requires db_session)
        db_session: Database session (required if save_to_db=True)
        
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
    elif config.source_type == "task_adapter":
        # Load tasks from task adapter (discovers tasks from tasks directory)
        adapter = TaskAdapter(
            train_ratio=config.train_ratio,
            tasks_dir=config.tasks_dir,
            seed=config.seed or 42,
        )
        
        if config.split_type == "train":
            task_infos = adapter.get_train_tasks()
        elif config.split_type == "eval":
            task_infos = adapter.get_eval_tasks()
        else:
            # Use all tasks (both train and eval)
            train_infos = adapter.get_train_tasks()
            eval_infos = adapter.get_eval_tasks()
            task_infos = train_infos + eval_infos
        
        # Convert task infos to CUATask objects
        # Extract app information from task path/module
        tasks = []
        for i, task_info in enumerate(task_infos):
            # Extract description from task_info
            desc = None
            task_instance = None
            if isinstance(task_info, dict):
                task_instance = task_info.get("task_instance")
                if task_instance and hasattr(task_instance, "description"):
                    desc = task_instance.description
                else:
                    desc = task_info.get("name", f"Task {i+1}")
            else:
                desc = str(task_info)
            
            # Identify app from module path or task path
            app_name = None
            module_path = task_info.get("module_path", "") if isinstance(task_info, dict) else ""
            task_path = task_info.get("path", "") if isinstance(task_info, dict) else ""
            
            # Check module path or file path for app name
            if 'airbnb' in module_path.lower() or 'airbnb' in task_path.lower():
                app_name = "airbnb"
            elif 'instagram' in module_path.lower() or 'instagram' in task_path.lower():
                app_name = "instagram"
            elif 'demo' in module_path.lower() or 'demo' in task_path.lower():
                app_name = "demo"
            else:
                # Fallback: identify from description
                desc_lower = desc.lower()
                if any(kw in desc_lower for kw in ['airbnb', 'listing', 'host', 'booking', 'reservation', 'save listings']):
                    app_name = "airbnb"
                elif any(kw in desc_lower for kw in ['instagram', 'post', 'reel', 'follow', 'like', 'comment']):
                    app_name = "instagram"
            
            # Get task name if available
            task_name = None
            if isinstance(task_info, dict):
                if task_instance and hasattr(task_instance, "name"):
                    task_name = task_instance.name
                else:
                    task_name = task_info.get("name", f"Task {i+1}")
            else:
                task_name = f"Task {i+1}"
            
            # Generate unique task ID based on task name (not loop index!)
            # This ensures train and eval tasks have different IDs even if they come from the same pool
            import hashlib
            # Use task name + description to create a unique, stable ID
            unique_str = f"{task_name}_{desc[:100] if desc else ''}"
            task_hash = hashlib.md5(unique_str.encode()).hexdigest()[:8]
            task_id = f"task_adapter_{task_hash}"
            
            # Create task with app info
            # Note: We don't set validation_query here - validation will be done via _original_task's validator
            task = CUATask(
                id=task_id,
                name=task_name,
                description=desc,
                difficulty=TaskDifficulty.MEDIUM,
                category=TaskCategory.SYSTEM,
                validation_query=None,  # Will use _original_task's validator instead
                expected_result=None,
                tags=[app_name] if app_name else [],
            )
            # Store app name as attribute for easy access
            task.app_name = app_name
            # Store original task instance so we can use its validator
            if task_instance:
                task._original_task = task_instance
            tasks.append(task)
    else:
        raise ValueError(
            f"Unknown source_type: {config.source_type}. "
            "Supported: 'demo_training', 'demo_eval', 'demo_all', 'ids', 'file', 'custom', 'task_adapter'"
        )
    
    # Apply filters for demo sources (only for demo_* source types, not task_adapter)
    if config.category is not None and config.source_type.startswith("demo_"):
        category = (
            TaskCategory(config.category) if isinstance(config.category, str) else config.category
        )
        tasks = [t for t in tasks if t.category == category]
    
    if config.difficulty is not None and config.source_type.startswith("demo_"):
        difficulty = (
            TaskDifficulty(config.difficulty) if isinstance(config.difficulty, str) else config.difficulty
        )
        tasks = [t for t in tasks if t.difficulty == difficulty]
    
    # Apply category filter for task_adapter (filter by app_name)
    if config.source_type == "task_adapter" and config.category is not None:
        category_lower = config.category.lower()
        filtered_tasks = []
        for t in tasks:
            app_name = getattr(t, 'app_name', None)
            if app_name and app_name.lower() == category_lower:
                filtered_tasks.append(t)
        tasks = filtered_tasks
        logger.info(f"Filtered to {len(tasks)} tasks with category='{config.category}'")
    
    # Apply task_names filter for task_adapter
    if config.source_type == "task_adapter" and config.task_names is not None:
        # Parse comma-separated task names
        requested_names = [name.strip() for name in config.task_names.split(',') if name.strip()]
        if requested_names:
            filtered_tasks = []
            for t in tasks:
                task_name = t.name if hasattr(t, 'name') else None
                if task_name and task_name in requested_names:
                    filtered_tasks.append(t)
            tasks = filtered_tasks
            logger.info(f"Filtered to {len(tasks)} tasks with task_names={requested_names}")
    
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
        + (f", task_names={config.task_names}" if config.task_names else "")
        + (f", limit={config.limit}" if config.limit else "")
    )
    
    # Save to database if requested
    if save_to_db and db_session is not None and tasks:
        try:
            from tinker_cookbook.recipes.cua_rl.database.database_task_loader import (
                save_cua_tasks_to_database,
            )
            save_cua_tasks_to_database(
                session=db_session,
                tasks=tasks,
                source_type=config.source_type,
            )
        except Exception as e:
            logger.warning(f"Failed to save tasks to database: {e}")
    
    return tasks


def load_tasks_from_multiple_sources(
    configs: List[TaskSourceConfig],
    save_to_db: bool = True,
    db_session = None,
) -> List[CUATask]:
    """
    Load tasks from multiple sources and combine them.
    
    Args:
        configs: List of TaskSourceConfig objects
        save_to_db: If True, save tasks to database (requires db_session)
        db_session: Database session (required if save_to_db=True)
        
    Returns:
        Combined list of CUATask objects
    """
    all_tasks: List[CUATask] = []
    for config in configs:
        tasks = load_tasks_from_config(config, save_to_db=save_to_db, db_session=db_session)
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

