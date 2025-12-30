"""Task adapter for splitting tasks into training and evaluation sets."""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def discover_all_tasks(tasks_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover all tasks from the tasks directory.
    
    Args:
        tasks_dir: Path to tasks directory. If None, uses default location.
        
    Returns:
        List of task metadata dictionaries with keys: name, path, module_path, create_task_func
    """
    if tasks_dir is None:
        # Default to tasks directory relative to this file
        current_file = Path(__file__).parent
        tasks_dir = str(current_file)
    
    tasks = []
    tasks_path = Path(tasks_dir)
    
    # Look for task.py files in subdirectories
    for task_file in tasks_path.rglob("task.py"):
        # Skip if in __pycache__ or other hidden directories
        if "__pycache__" in str(task_file):
            continue
        
        try:
            # Get relative path from tasks_dir
            # task_file is like: /path/to/tasks/airbnb/01_i_plan_to_go_to_united/task.py
            # tasks_path is like: /path/to/tasks
            rel_path = task_file.relative_to(tasks_path)
            # rel_path is like: airbnb/01_i_plan_to_go_to_united/task.py
            
            # Convert to module path
            # Need to find the base package path
            # Assuming we're in tinker_cookbook/recipes/cua_rl/tasks/
            # The module path should be: tinker_cookbook.recipes.cua_rl.tasks.airbnb.01_i_plan_to_go_to_united.task
            parts = list(rel_path.parts)
            # Remove .py extension from last part
            parts[-1] = parts[-1].replace(".py", "")
            # Build module path
            module_path = "tinker_cookbook.recipes.cua_rl.tasks." + ".".join(parts)
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find create_task function
            if hasattr(module, "create_task"):
                create_task_func = getattr(module, "create_task")
                # Try to get task name from the function
                task_instance = create_task_func()
                task_name = getattr(task_instance, "name", None) or task_file.parent.name
                
                tasks.append({
                    "name": task_name,
                    "path": str(task_file),
                    "module_path": module_path,
                    "create_task_func": create_task_func,
                    "task_instance": task_instance,
                })
                logger.debug(f"Discovered task: {task_name} from {module_path}")
        except Exception as e:
            logger.warning(f"Failed to load task from {task_file}: {e}")
            continue
    
    return tasks


def split_tasks_train_eval(
    tasks: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split tasks into training and evaluation sets.
    
    Args:
        tasks: List of task metadata dictionaries
        train_ratio: Ratio of tasks for training (default: 0.8, meaning 80% train, 20% eval)
        seed: Random seed for reproducible splitting
        
    Returns:
        Tuple of (training_tasks, eval_tasks)
    """
    if not tasks:
        return [], []
    
    # Sort tasks by name for deterministic ordering
    sorted_tasks = sorted(tasks, key=lambda t: t["name"])
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle with fixed seed
    shuffled = sorted_tasks.copy()
    random.shuffle(shuffled)
    
    # Split
    n_train = int(len(shuffled) * train_ratio)
    train_tasks = shuffled[:n_train]
    eval_tasks = shuffled[n_train:]
    
    logger.info(
        f"Split {len(tasks)} tasks into {len(train_tasks)} training ({train_ratio*100:.0f}%) "
        f"and {len(eval_tasks)} evaluation ({(1-train_ratio)*100:.0f}%) tasks (seed={seed})"
    )
    
    return train_tasks, eval_tasks


def get_task_descriptions(tasks: List[Dict[str, Any]]) -> List[str]:
    """Extract task descriptions from task instances.
    
    Args:
        tasks: List of task metadata dictionaries
        
    Returns:
        List of task description strings
    """
    descriptions = []
    for task_info in tasks:
        task_instance = task_info.get("task_instance")
        if task_instance and hasattr(task_instance, "description"):
            descriptions.append(task_instance.description)
        else:
            # Fallback to name if no description
            descriptions.append(task_info.get("name", "Unknown task"))
    return descriptions


class TaskAdapter:
    """Adapter for loading and splitting tasks from the tasks directory."""
    
    def __init__(
        self,
        tasks_dir: Optional[str] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        """Initialize task adapter.
        
        Args:
            tasks_dir: Path to tasks directory. If None, uses default location.
            train_ratio: Ratio of tasks for training (default: 0.8)
            seed: Random seed for splitting (default: 42)
        """
        self.tasks_dir = tasks_dir
        self.train_ratio = train_ratio
        self.seed = seed
        self._all_tasks: Optional[List[Dict[str, Any]]] = None
        self._train_tasks: Optional[List[Dict[str, Any]]] = None
        self._eval_tasks: Optional[List[Dict[str, Any]]] = None
    
    def discover_tasks(self) -> List[Dict[str, Any]]:
        """Discover all tasks from the tasks directory."""
        if self._all_tasks is None:
            self._all_tasks = discover_all_tasks(self.tasks_dir)
        return self._all_tasks
    
    def get_train_tasks(self) -> List[Dict[str, Any]]:
        """Get training tasks."""
        if self._train_tasks is None:
            all_tasks = self.discover_tasks()
            train, eval_tasks = split_tasks_train_eval(
                all_tasks, 
                train_ratio=self.train_ratio,
                seed=self.seed
            )
            self._train_tasks = train
            self._eval_tasks = eval_tasks
        return self._train_tasks
    
    def get_eval_tasks(self) -> List[Dict[str, Any]]:
        """Get evaluation tasks."""
        if self._eval_tasks is None:
            all_tasks = self.discover_tasks()
            train, eval_tasks = split_tasks_train_eval(
                all_tasks,
                train_ratio=self.train_ratio,
                seed=self.seed
            )
            self._train_tasks = train
            self._eval_tasks = eval_tasks
        return self._eval_tasks
    
    def get_train_descriptions(self) -> List[str]:
        """Get training task descriptions."""
        return get_task_descriptions(self.get_train_tasks())
    
    def get_eval_descriptions(self) -> List[str]:
        """Get evaluation task descriptions."""
        return get_task_descriptions(self.get_eval_tasks())


# Convenience function for easy usage
def get_tasks_train_eval(
    tasks_dir: Optional[str] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Get training and evaluation task descriptions.
    
    Args:
        tasks_dir: Path to tasks directory. If None, uses default location.
        train_ratio: Ratio of tasks for training (default: 0.8)
        seed: Random seed for splitting (default: 42)
        
    Returns:
        Tuple of (train_descriptions, eval_descriptions)
    """
    adapter = TaskAdapter(tasks_dir=tasks_dir, train_ratio=train_ratio, seed=seed)
    return adapter.get_train_descriptions(), adapter.get_eval_descriptions()

