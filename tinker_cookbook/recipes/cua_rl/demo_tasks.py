"""Sample tasks for CUA Agent GRPO training.

This module defines 10 sample Android tasks for training and evaluation.
Each task can be validated via gbox system state APIs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class TaskDifficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskCategory(str, Enum):
    """Task categories."""
    SYSTEM = "system"
    NAVIGATION = "navigation"
    SETTINGS = "settings"
    APP = "app"
    INPUT = "input"


@dataclass
class CUATask:
    """A task for CUA agent to complete on Android."""
    
    id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    category: TaskCategory
    max_steps: int = 10
    
    # Validation config
    validation_type: str = "state"  # "state", "screenshot", "api"
    validation_query: Optional[str] = None  # Query to check against gbox state
    expected_result: Optional[Any] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "category": self.category.value,
            "max_steps": self.max_steps,
            "validation_type": self.validation_type,
            "validation_query": self.validation_query,
            "expected_result": self.expected_result,
            "tags": self.tags,
            "prerequisites": self.prerequisites,
        }

# =============================================================================
# TRAINING TASKS
# =============================================================================

DEMO_TRAINING_TASKS = [
    # Task 1: Open Settings app
    CUATask(
        id="train_01_open_settings",
        name="Open Settings",
        description="Open the Settings app from the home screen.",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.APP,
        max_steps=5,
        validation_type="state",
        validation_query="current_app",
        expected_result="com.android.settings",
        tags=["app", "settings", "launch"],
    ),

    # Task 2: Set screen brightness to maximum
    CUATask(
        id="train_02_max_brightness",
        name="Maximum Brightness",
        description="Open Settings and set the screen brightness to maximum.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="brightness_level",
        expected_result=255,
        tags=["settings", "display", "brightness"],
    ),

    # Task 3: Enable Airplane Mode
    CUATask(
        id="train_03_airplane_mode",
        name="Enable Airplane Mode",
        description="Go to Settings and turn on Airplane Mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="airplane_mode",
        expected_result=True,
        tags=["settings", "network", "airplane"],
    ),

    # Task 4: Check brightness level
    CUATask(
        id="train_04_check_brightness",
        name="Check Brightness Level",
        description="Open Settings and navigate to the Brightness section to check the current brightness level.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="result_message_contains",
        expected_result="83",
        tags=["settings", "battery", "info"],
    ),

    # Task 5: Disable WiFi
    CUATask(
        id="train_05_disable_wifi",
        name="Disable WiFi",
        description="Go to Settings and turn off WiFi if it's on.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=8,
        validation_type="state",
        validation_query="wifi_enabled",
        expected_result=False,
        tags=["settings", "wifi", "toggle", "off"],
    ),

    # Task 6: Set screen brightness to minimum
    CUATask(
        id="train_06_min_brightness",
        name="Minimum Brightness",
        description="Open Settings and set the screen brightness to the minimum level.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="brightness_level",
        expected_result=1,
        tags=["settings", "display", "brightness", "low"],
    ),

    # Task 9: Clear all notifications
    CUATask(
        id="train_09_clear_notifications",
        name="Clear All Notifications",
        description="Open the notifications shade and clear all existing notifications.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=8,
        validation_type="state",
        validation_query="notification_count",
        expected_result=0,
        tags=["system", "notifications", "clear"],
    ),

    # Task 10: Change screen timeout to 30 seconds
    CUATask(
        id="train_10_timeout_30s",
        name="Change Screen Timeout to 30s",
        description="Open Settings, navigate to Display, and set the screen timeout to 30 seconds.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=30000,
        tags=["settings", "display", "timeout"],
    ),

    # Task 11: Change screen timeout to 1 minute
    CUATask(
        id="train_11_timeout_1min",
        name="Change Screen Timeout to 1 Minute",
        description="Open Settings, navigate to Display, and set the screen timeout to 1 minute.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=60000,
        tags=["settings", "display", "timeout"],
    ),

    # Task 12: Change screen timeout to 2 minutes
    CUATask(
        id="train_12_timeout_2min",
        name="Change Screen Timeout to 2 Minutes",
        description="Open Settings, navigate to Display, and set the screen timeout to 2 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=120000,
        tags=["settings", "display", "timeout"],
    ),


    # Task 13: Disable automatic date & time
    CUATask(
        id="train_13_disable_auto_time",
        name="Disable Automatic Date & Time",
        description="In Date & Time settings, disable automatic date & time.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="auto_time_enabled",
        expected_result=False,
        tags=["settings", "system", "time", "manual"],
    ),

    # Task 14: Enable battery saver
    CUATask(
        id="train_14_enable_battery_saver",
        name="Enable Battery Saver",
        description="Open Settings, go to Battery, and enable Battery Saver mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="battery_saver_enabled",
        expected_result=True,
        tags=["settings", "battery", "saver"],
    ),

    # Task 19: Uninstall Instagram
    CUATask(
        id="train_19_uninstall_instagram",
        name="Uninstall Instagram",
        description="Uninstall the Instagram app from the device.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=12,
        validation_type="state",
        validation_query="app_installed",
        expected_result={"package": "com.instagram.android", "installed": False},
        tags=["system", "uninstall", "instagram"],
    ),


    # Task 21: Create folder in Downloads
    CUATask(
        id="train_21_create_downloads_folder",
        name="Create Folder in Downloads",
        description="Create a new folder named 'abc' in the Downloads directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Download/abc", "exists": True},
        tags=["system", "file", "folder", "downloads"],
    ),

    # Task 22: Download logo from gbox.ai
    CUATask(
        id="train_22_download_gbox_logo",
        name="Download GBOX Logo",
        description="Navigate to gbox.ai website and download the logo file. Verify the downloaded file is named logo.svg.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.INPUT,
        max_steps=25,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Download/logo.svg", "exists": True},
        tags=["browser", "download", "file", "gbox"],
    ),

    # Task 23: Check GBOX Keyboard storage size
    CUATask(
        id="train_23_check_gbox_keyboard_storage",
        name="Check GBOX Keyboard Storage Size",
        description="Check how much storage space the GBOX Keyboard app is using.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="api",
        validation_query="finish_message_contains",
        expected_result="73.73KB",
        tags=["settings", "apps", "storage", "gbox"],
    ),

    # Task 26: Check Chrome storage size
    CUATask(
        id="train_26_check_chrome_storage",
        name="Check Chrome Storage Size",
        description="Check how much storage space the Chrome app is using. You must report the exact storage size (include units like KB, MB, or GB) in your finish message.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="api",
        validation_query="finish_message_contains_size",
        expected_result="storage_size_reported",
        tags=["settings", "apps", "storage", "chrome"],
    ),


    # Task 30: Uninstall Facebook
    CUATask(
        id="train_30_uninstall_facebook",
        name="Uninstall Facebook",
        description="Uninstall the Facebook app from the device.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="app_installed",
        expected_result={"package": "com.facebook.katana", "installed": False},
        tags=["system", "uninstall", "facebook"],
    ),

    # Task 32: Check Instagram storage size
    CUATask(
        id="train_32_check_instagram_storage",
        name="Check Instagram Storage Size",
        description="Check how much storage space the Instagram app is using. You must report the exact storage size (include units like KB, MB, or GB) in your finish message.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="api",
        validation_query="finish_message_contains_size",
        expected_result="storage_size_reported",
        tags=["settings", "apps", "storage", "instagram"],
    ),

    # Task 35: Change screen timeout to 5 minutes
    CUATask(
        id="train_35_timeout_5min",
        name="Change Screen Timeout to 5 Minutes",
        description="Open Settings, navigate to Display, and set the screen timeout to 5 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=16,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=300000,
        tags=["settings", "display", "timeout"],
    ),

    # Task 36: Change screen timeout to 10 minutes
    CUATask(
        id="train_36_timeout_10min",
        name="Change Screen Timeout to 10 Minutes",
        description="Open Settings, navigate to Display, and set the screen timeout to 10 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=16,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=600000,
        tags=["settings", "display", "timeout"],
    ),

    # Task 43: Clear all app notifications
    CUATask(
        id="train_43_clear_all_notifications",
        name="Clear All Notifications",
        description="Open the notifications shade and clear all existing notifications.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="notification_count",
        expected_result=0,
        tags=["system", "notifications", "clear"],
    ),


    # Task 48: Change screen timeout to 30 minutes
    CUATask(
        id="train_48_timeout_30min",
        name="Change Screen Timeout to 30 Minutes",
        description="Open Settings, navigate to Display, and set the screen timeout to 30 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=16,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=1800000,
        tags=["settings", "display", "timeout"],
    ),

]

# =============================================================================
# EVALUATION TASKS
# =============================================================================

DEMO_EVAL_TASKS = [
    # Eval Task 1: Change display timeout to 5 minutes
    CUATask(
        id="eval_01_display_timeout_5min",
        name="Change Display Timeout to 5 Minutes",
        description="Open Settings, go to Display, and change the screen timeout to 5 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=16,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=300000,
        tags=["settings", "display", "timeout"],
    ),

    # Eval Task 2: Enable Do Not Disturb
    CUATask(
        id="eval_02_dnd_mode_enable",
        name="Enable Do Not Disturb",
        description="Enable Do Not Disturb mode from quick settings or the Settings app.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=16,
        validation_type="state",
        validation_query="dnd_enabled",
        expected_result=True,
        tags=["settings", "notifications", "dnd"],
    ),

    # Eval Task 5: Enable battery saver
    CUATask(
        id="eval_05_enable_battery_saver",
        name="Enable Battery Saver",
        description="Open Battery settings and enable Battery Saver mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=16,
        validation_type="state",
        validation_query="battery_saver_enabled",
        expected_result=True,
        tags=["settings", "battery", "saver"],
    ),

    # Eval Task 6: Uninstall Instagram
    CUATask(
        id="eval_06_uninstall_instagram",
        name="Uninstall Instagram",
        description="Uninstall the Instagram app from the device.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="app_installed",
        expected_result={"package": "com.instagram.android", "installed": False},
        tags=["system", "uninstall", "instagram"],
    ),

    # Eval Task 9: Create folder in Downloads
    CUATask(
        id="eval_09_create_downloads_folder",
        name="Create Folder in Downloads",
        description="Create a new folder named 'abc' in the Downloads directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Download/abc", "exists": True},
        tags=["system", "file", "folder", "downloads"],
    ),

    # Eval Task 10: Download logo from gbox.ai
    CUATask(
        id="eval_10_download_gbox_logo",
        name="Download GBOX Logo",
        description="Navigate to gbox.ai website and download the logo file. Verify the downloaded file is named logo.svg.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.INPUT,
        max_steps=25,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Download/logo.svg", "exists": True},
        tags=["browser", "download", "file", "gbox"],
    ),

    # Eval Task 11: Check GBOX Keyboard storage size
    CUATask(
        id="eval_11_check_gbox_keyboard_storage",
        name="Check GBOX Keyboard Storage Size",
        description="Check how much storage space the GBOX Keyboard app is using. You must report the exact storage size (e.g., '20MB') in your finish message.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="api",
        validation_query="finish_message_contains",
        expected_result="73.73KB",
        tags=["settings", "apps", "storage", "gbox"],
    ),

    # Eval Task 12: Check Chrome storage size
    CUATask(
        id="eval_12_check_chrome_storage",
        name="Check Chrome Storage Size",
        description="Check how much storage space the Chrome app is using. You must report the exact storage size (e.g., '20MB') in your finish message.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="api",
        validation_query="finish_message_contains",
        expected_result="51.54MB",
        tags=["settings", "apps", "storage", "chrome"],
    ),

    # Eval Task 16: Create folder in Downloads with different name
    CUATask(
        id="eval_16_create_downloads_folder_xyz",
        name="Create Folder in Downloads",
        description="Create a new folder named 'xyz' in the Downloads directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Download/xyz", "exists": True},
        tags=["system", "file", "folder", "downloads"],
    ),

]

# =============================================================================
# TASK UTILITIES
# =============================================================================

def get_all_tasks() -> List[CUATask]:
    """Get all tasks (training + evaluation)."""
    return DEMO_TRAINING_TASKS + DEMO_EVAL_TASKS


def get_training_tasks() -> List[CUATask]:
    """Get training tasks only."""
    return DEMO_TRAINING_TASKS


def get_eval_tasks() -> List[CUATask]:
    """Get evaluation tasks only."""
    return DEMO_EVAL_TASKS


def get_task_by_id(task_id: str) -> Optional[CUATask]:
    """Get a task by its ID."""
    for task in get_all_tasks():
        if task.id == task_id:
            return task
    return None


def get_tasks_by_category(category: TaskCategory) -> List[CUATask]:
    """Get tasks by category."""
    return [t for t in get_all_tasks() if t.category == category]


def get_tasks_by_difficulty(difficulty: TaskDifficulty) -> List[CUATask]:
    """Get tasks by difficulty."""
    return [t for t in get_all_tasks() if t.difficulty == difficulty]


__all__ = [
    "CUATask",
    "TaskDifficulty", 
    "TaskCategory",
    "DEMO_TRAINING_TASKS",
    "DEMO_EVAL_TASKS",
    "get_all_tasks",
    "get_training_tasks",
    "get_eval_tasks",
    "get_task_by_id",
    "get_tasks_by_category",
    "get_tasks_by_difficulty",
]

