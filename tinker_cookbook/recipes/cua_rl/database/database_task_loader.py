"""
Database integration for task loading.

This module provides functions to save tasks and validators to the database
when they are loaded.
"""

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from tinker_cookbook.recipes.cua_rl.demo_tasks import CUATask
from tinker_cookbook.recipes.cua_rl.database.database_dao import (
    create_or_get_task,
    create_or_get_validator,
    get_task_by_task_id,
)
from tinker_cookbook.recipes.cua_rl.tasks.task_adapter import TaskAdapter

logger = logging.getLogger(__name__)


def save_task_to_database(session: Session, task: CUATask, source_type: str) -> int:
    """
    Save a CUATask to the database and return the database task ID.
    
    Args:
        session: Database session
        task: CUATask object
        source_type: Source type (e.g., 'demo_training', 'demo_eval', 'task_adapter')
        
    Returns:
        Database task ID
    """
    # Check if task already exists
    db_task = get_task_by_task_id(session, task.id)
    if db_task:
        logger.debug(f"Task {task.id} already exists in database, using existing record")
        return db_task.id
    
    # Extract app_name if available
    app_name = getattr(task, 'app_name', None)
    
    # Create task in database
    db_task = create_or_get_task(
        session,
        task_id=task.id,
        name=task.name,
        description=task.description,
        difficulty=task.difficulty.value if hasattr(task.difficulty, 'value') else str(task.difficulty),
        category=task.category.value if hasattr(task.category, 'value') else str(task.category),
        max_steps=task.max_steps,
        validation_type=task.validation_type,
        validation_query=task.validation_query,
        expected_result=str(task.expected_result) if task.expected_result is not None else None,
        tags=task.tags if task.tags else [],
        prerequisites=task.prerequisites if task.prerequisites else [],
        app_name=app_name,
        source_type=source_type,
    )
    
    logger.debug(f"Saved task {task.id} to database with ID {db_task.id}")
    return db_task.id


def save_validator_to_database(
    session: Session,
    task_id: int,
    validator_instance: Optional[object],
    validation_query: Optional[str] = None,
) -> Optional[int]:
    """
    Save a validator to the database.
    
    Args:
        session: Database session
        task_id: Database task ID
        validator_instance: Validator instance (has validate method)
        validation_query: Validation query string (optional)
        
    Returns:
        Database validator ID, or None if no validator
    """
    if validator_instance is None and validation_query is None:
        return None
    
    # Determine validator type
    validator_type = type(validator_instance).__name__ if validator_instance else "query_based"
    
    # Try to extract validation query from validator if not provided
    if validation_query is None and validator_instance:
        # Check if validator has a validation_query attribute or method
        if hasattr(validator_instance, 'validation_query'):
            validation_query = validator_instance.validation_query
        elif hasattr(validator_instance, 'get_validation_query'):
            validation_query = validator_instance.get_validation_query()
    
    # Create validator in database
    validator = create_or_get_validator(
        session,
        task_id=task_id,
        validator_type=validator_type,
        validation_query=validation_query,
        validation_method="validate" if validator_instance else "query",
    )
    
    logger.debug(f"Saved validator for task_id {task_id} to database with ID {validator.id}")
    return validator.id


def save_cua_tasks_to_database(
    session: Session,
    tasks: List[CUATask],
    source_type: str,
) -> List[int]:
    """
    Save a list of CUATasks to the database.
    
    Args:
        session: Database session
        tasks: List of CUATask objects
        source_type: Source type (e.g., 'demo_training', 'demo_eval', 'task_adapter')
        
    Returns:
        List of database task IDs
    """
    task_ids = []
    
    for task in tasks:
        # Save task
        db_task_id = save_task_to_database(session, task, source_type)
        task_ids.append(db_task_id)
        
        # Save validator if available
        validator_instance = None
        validation_query = task.validation_query
        
        # Check for _original_task with validator
        if hasattr(task, '_original_task') and task._original_task:
            original_task = task._original_task
            if hasattr(original_task, 'get_validator'):
                validator_instance = original_task.get_validator()
        
        save_validator_to_database(
            session,
            task_id=db_task_id,
            validator_instance=validator_instance,
            validation_query=validation_query,
        )
    
    logger.info(f"Saved {len(tasks)} tasks to database from source_type='{source_type}'")
    return task_ids


def load_tasks_from_database(session: Session, task_ids: Optional[List[str]] = None) -> List[CUATask]:
    """
    Load tasks from database and convert to CUATask objects.
    
    Args:
        session: Database session
        task_ids: Optional list of task_id strings to filter by
        
    Returns:
        List of CUATask objects
    """
    from tinker_cookbook.recipes.cua_rl.demo_tasks import TaskDifficulty, TaskCategory
    from tinker_cookbook.recipes.cua_rl.database.database_dao import list_tasks, get_task_by_task_id
    from tinker_cookbook.recipes.cua_rl.database import json_deserialize
    
    if task_ids:
        # Load specific tasks
        tasks = []
        for task_id in task_ids:
            db_task = get_task_by_task_id(session, task_id)
            if db_task:
                tasks.append(db_task)
    else:
        # Load all tasks
        tasks = list_tasks(session)
    
    # Convert to CUATask objects
    cua_tasks = []
    for db_task in tasks:
        # Parse difficulty and category
        difficulty = TaskDifficulty(db_task.difficulty) if db_task.difficulty else TaskDifficulty.MEDIUM
        category = TaskCategory(db_task.category) if db_task.category else TaskCategory.SYSTEM
        
        # Parse tags and prerequisites
        tags = json_deserialize(db_task.tags) if db_task.tags else []
        prerequisites = json_deserialize(db_task.prerequisites) if db_task.prerequisites else []
        
        cua_task = CUATask(
            id=db_task.task_id,
            name=db_task.name,
            description=db_task.description,
            difficulty=difficulty,
            category=category,
            max_steps=db_task.max_steps or 10,
            validation_type=db_task.validation_type or "state",
            validation_query=db_task.validation_query,
            expected_result=db_task.expected_result,
            tags=tags if isinstance(tags, list) else [],
            prerequisites=prerequisites if isinstance(prerequisites, list) else [],
        )
        
        # Set app_name if available
        if db_task.app_name:
            cua_task.app_name = db_task.app_name
        
        cua_tasks.append(cua_task)
    
    logger.info(f"Loaded {len(cua_tasks)} tasks from database")
    return cua_tasks

