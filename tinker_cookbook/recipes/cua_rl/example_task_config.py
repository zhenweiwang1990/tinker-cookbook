"""
Example script showing how to use the flexible task configuration system.

This demonstrates various ways to configure tasks for CUA RL training.
"""

from tinker_cookbook.recipes.cua_rl.task_loader import (
    TaskSourceConfig,
    load_tasks_from_config,
    load_tasks_from_multiple_sources,
    load_demo_training_tasks,
    load_demo_eval_tasks,
    load_tasks_by_ids,
)


def example_1_all_training_tasks():
    """Example 1: Load all demo training tasks."""
    print("=" * 60)
    print("Example 1: All Demo Training Tasks")
    print("=" * 60)
    
    config = TaskSourceConfig(source_type="demo_training")
    tasks = load_tasks_from_config(config)
    print(f"Loaded {len(tasks)} tasks")
    print(f"First 3 tasks: {tasks[:3]}")
    print()


def example_2_filter_by_category():
    """Example 2: Filter by category."""
    print("=" * 60)
    print("Example 2: Settings Tasks Only")
    print("=" * 60)
    
    config = TaskSourceConfig(
        source_type="demo_training",
        category="settings",
    )
    tasks = load_tasks_from_config(config)
    print(f"Loaded {len(tasks)} settings tasks")
    print(f"First 3 tasks: {tasks[:3]}")
    print()


def example_3_filter_by_difficulty():
    """Example 3: Filter by difficulty."""
    print("=" * 60)
    print("Example 3: Easy Tasks Only")
    print("=" * 60)
    
    config = TaskSourceConfig(
        source_type="demo_training",
        difficulty="easy",
    )
    tasks = load_tasks_from_config(config)
    print(f"Loaded {len(tasks)} easy tasks")
    print(f"First 3 tasks: {tasks[:3]}")
    print()


def example_4_combine_filters():
    """Example 4: Combine filters with limit."""
    print("=" * 60)
    print("Example 4: Easy Settings Tasks (Limited to 5)")
    print("=" * 60)
    
    config = TaskSourceConfig(
        source_type="demo_training",
        category="settings",
        difficulty="easy",
        limit=5,
        seed=42,
    )
    tasks = load_tasks_from_config(config)
    print(f"Loaded {len(tasks)} tasks")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")
    print()


def example_5_specific_task_ids():
    """Example 5: Load specific tasks by ID."""
    print("=" * 60)
    print("Example 5: Specific Task IDs")
    print("=" * 60)
    
    task_ids = [
        "train_01_open_settings",
        "train_02_enable_wifi",
        "train_05_airplane_mode",
    ]
    tasks = load_tasks_by_ids(task_ids)
    print(f"Loaded {len(tasks)} tasks")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")
    print()


def example_6_multiple_sources():
    """Example 6: Load from multiple sources."""
    print("=" * 60)
    print("Example 6: Multiple Sources")
    print("=" * 60)
    
    configs = [
        TaskSourceConfig(
            source_type="demo_training",
            category="settings",
            limit=5,
        ),
        TaskSourceConfig(
            source_type="demo_training",
            category="navigation",
            limit=3,
        ),
    ]
    tasks = load_tasks_from_multiple_sources(configs)
    print(f"Loaded {len(tasks)} tasks from {len(configs)} sources")
    print(f"First 5 tasks: {tasks[:5]}")
    print()


def example_7_convenience_functions():
    """Example 7: Using convenience functions."""
    print("=" * 60)
    print("Example 7: Convenience Functions")
    print("=" * 60)
    
    # Using convenience functions
    training_tasks = load_demo_training_tasks(category="settings", limit=3)
    eval_tasks = load_demo_eval_tasks(difficulty="medium", limit=2)
    
    print(f"Training tasks ({len(training_tasks)}):")
    for task in training_tasks:
        print(f"  - {task}")
    print(f"\nEval tasks ({len(eval_tasks)}):")
    for task in eval_tasks:
        print(f"  - {task}")
    print()


def example_8_dict_config():
    """Example 8: Using dict config (for CLI/JSON)."""
    print("=" * 60)
    print("Example 8: Dict Config (for CLI/JSON)")
    print("=" * 60)
    
    # This is how you'd pass it from CLI or JSON
    config_dict = {
        "source_type": "demo_training",
        "category": "app",
        "difficulty": "easy",
        "limit": 5,
        "seed": 123,
    }
    
    config = TaskSourceConfig(**config_dict)
    tasks = load_tasks_from_config(config)
    print(f"Loaded {len(tasks)} tasks from dict config")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CUA RL Task Configuration Examples")
    print("=" * 60 + "\n")
    
    example_1_all_training_tasks()
    example_2_filter_by_category()
    example_3_filter_by_difficulty()
    example_4_combine_filters()
    example_5_specific_task_ids()
    example_6_multiple_sources()
    example_7_convenience_functions()
    example_8_dict_config()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)

