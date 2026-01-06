"""Example usage of AdbClient with GBox and TaskAdapter."""
from __future__ import annotations

import os
import asyncio
from tinker_cookbook.recipes.cua_rl.tasks.adb import AdbClient
from tinker_cookbook.recipes.cua_rl.tasks.task_adapter import TaskAdapter, get_tasks_train_eval
from tinker_cookbook.recipes.cua_rl.gbox.client import CuaGBoxClient


async def example_gbox_adb():
    """Example: Using AdbClient with GBox."""
    api_key = os.environ.get("GBOX_API_KEY")
    if not api_key:
        print("GBOX_API_KEY not set, skipping GBox example")
        return
    
    # Create GBox client
    gbox_client = CuaGBoxClient(api_key=api_key)
    await gbox_client.create_box(box_type="android")
    
    try:
        # Create AdbClient with GBox
        adb = AdbClient(gbox_client=gbox_client)
        
        # Use adb commands (executed via GBox)
        print("Checking installed packages...")
        result = adb._run("shell", "pm", "list", "packages", "airbnb", capture_output=True)
        print(f"Result: {result}")
        
        # Run SQLite query
        print("Running SQLite query...")
        query_result = adb.run_sqlite_query(
            package_name="com.airbnb.clone",
            db_relative_path="/data/data/"+config.get_package_name()+"/databases/airbnbSQLiteSQLite.db",
            sql="SELECT COUNT(*) FROM favorites;"
        )
        print(f"Query result: {query_result}")
        
    finally:
        await gbox_client.close()


def example_local_adb():
    """Example: Using AdbClient with local ADB."""
    # Create AdbClient (uses local ADB by default)
    adb = AdbClient()
    
    # Use adb commands
    print("Checking installed packages...")
    result = adb._run("shell", "pm", "list", "packages", "airbnb", capture_output=True)
    print(f"Result: {result}")


def example_task_adapter():
    """Example: Using TaskAdapter to split tasks."""
    # Create adapter with default settings (80% train, 20% eval, seed=42)
    adapter = TaskAdapter(train_ratio=0.8)
    
    # Get task descriptions
    train_descriptions = adapter.get_train_descriptions()
    eval_descriptions = adapter.get_eval_descriptions()
    
    print(f"Training tasks: {len(train_descriptions)}")
    print(f"Evaluation tasks: {len(eval_descriptions)}")
    
    # Print first few training tasks
    print("\nFirst 5 training tasks:")
    for i, desc in enumerate(train_descriptions[:5], 1):
        print(f"{i}. {desc[:80]}...")
    
    # Get full task instances
    train_tasks = adapter.get_train_tasks()
    print(f"\nFirst training task details:")
    if train_tasks:
        task_info = train_tasks[0]
        task = task_info["task_instance"]
        print(f"  Name: {task.name}")
        print(f"  Description: {task.description[:100]}...")
        print(f"  Module: {task_info['module_path']}")


def example_convenience_function():
    """Example: Using convenience function."""
    train_descriptions, eval_descriptions = get_tasks_train_eval(
        train_ratio=0.95,
        seed=42
    )
    
    print(f"Training: {len(train_descriptions)} tasks")
    print(f"Evaluation: {len(eval_descriptions)} tasks")


if __name__ == "__main__":
    print("=== Task Adapter Example ===")
    example_task_adapter()
    
    print("\n=== Convenience Function Example ===")
    example_convenience_function()
    
    print("\n=== Local ADB Example ===")
    try:
        example_local_adb()
    except Exception as e:
        print(f"Local ADB not available: {e}")
    
    print("\n=== GBox ADB Example ===")
    asyncio.run(example_gbox_adb())

