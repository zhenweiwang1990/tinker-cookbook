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
# TRAINING TASKS (48 tasks)
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
        max_steps=8,
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

    # Task 7: Increase media volume to maximum
    CUATask(
        id="train_07_max_media_volume",
        name="Max Media Volume",
        description="Open Settings or use volume controls to set the media volume to maximum.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="media_volume",
        expected_result=100,
        tags=["settings", "sound", "volume", "media"],
    ),

    # Task 8: Mute media volume
    CUATask(
        id="train_08_mute_media_volume",
        name="Mute Media Volume",
        description="Open Settings or use volume controls to mute the media volume (set to 0).",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="media_volume",
        expected_result=0,
        tags=["settings", "sound", "volume", "media", "mute"],
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

    # Task 15: Clear Chrome app cache only
    CUATask(
        id="train_15_clear_chrome_cache",
        name="Clear Chrome App Cache",
        description="Navigate to Chrome's App Info > Storage and clear only the app cache (not data).",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=18,
        validation_type="state",
        validation_query="chrome_cache_cleared",
        expected_result=True,
        tags=["settings", "apps", "chrome", "cache"],
    ),


    # Task 16: Clear Facebook app cache only
    CUATask(
        id="train_16_clear_facebook_cache",
        name="Clear Facebook App Cache",
        description="Navigate to Facebook's App Info > Storage and clear only the app cache (not data).",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=18,
        validation_type="state",
        validation_query="facebook_cache_cleared",
        expected_result=True,
        tags=["settings", "apps", "facebook", "cache"],
    ),


    # Task 17: Change system language
    CUATask(
        id="train_17_change_system_language",
        name="Change System Language to Chinese",
        description="Open Language & input settings and change the system language to Chinese.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=22,
        validation_type="state",
        validation_query="system_language",
        expected_result="zh",
        tags=["settings", "language", "system"],
    ),

    # Task 18: Enable dark theme
    CUATask(
        id="train_18_enable_dark_theme",
        name="Enable Dark Theme",
        description="Open Display settings or quick settings and enable the system dark theme.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="dark_theme_enabled",
        expected_result=True,
        tags=["settings", "display", "theme", "dark"],
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

    # Task 20: Disable Facebook contacts permission
    CUATask(
        id="train_20_disable_facebook_contacts",
        name="Disable Facebook Contacts Permission",
        description="Go to Facebook app settings and disable the Contacts permission.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="app_permission",
        expected_result={"package": "com.facebook.katana", "permission": "android.permission.READ_CONTACTS", "granted": False},
        tags=["settings", "apps", "facebook", "permissions", "contacts"],
    ),

    # Task 21: Create folder in Downloads
    CUATask(
        id="train_21_create_downloads_folder",
        name="Create Folder in Downloads",
        description="Create a new folder named 'abc' in the Downloads directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
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
        description="Check how much storage space the GBOX Keyboard app is using. You must report the exact storage size (e.g., '73.73KB') in your finish message. The correct answer is 73.73KB.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=12,
        validation_type="api",
        validation_query="finish_message_contains",
        expected_result="73.73KB",
        tags=["settings", "apps", "storage", "gbox"],
    ),

    # Task 24: Disable Instagram location permission
    CUATask(
        id="train_24_disable_instagram_location",
        name="Disable Instagram Location Permission",
        description="Go to Instagram app settings and disable the Location permission.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="app_permission",
        expected_result={"package": "com.instagram.android", "permission": "android.permission.ACCESS_FINE_LOCATION", "granted": False},
        tags=["settings", "apps", "instagram", "permissions", "location"],
    ),

    # Task 25: Create folder in Documents
    CUATask(
        id="train_25_create_documents_folder",
        name="Create Folder in Documents",
        description="Create a new folder named 'test' in the Documents directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Documents/test", "exists": True},
        tags=["system", "file", "folder", "documents"],
    ),

    # Task 26: Check Chrome storage size
    CUATask(
        id="train_26_check_chrome_storage",
        name="Check Chrome Storage Size",
        description="Check how much storage space the Chrome app is using. You must report the exact storage size (include units like KB, MB, or GB) in your finish message.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=12,
        validation_type="api",
        validation_query="finish_message_contains_size",
        expected_result="storage_size_reported",
        tags=["settings", "apps", "storage", "chrome"],
    ),

    # Task 27: Enable Instagram camera permission
    CUATask(
        id="train_27_enable_instagram_camera",
        name="Enable Instagram Camera Permission",
        description="Go to Instagram app settings and enable the Camera permission if it's disabled.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=14,
        validation_type="state",
        validation_query="app_permission",
        expected_result={"package": "com.instagram.android", "permission": "android.permission.CAMERA", "granted": True},
        tags=["settings", "apps", "instagram", "permissions", "camera"],
    ),

    # Task 28: Clear Instagram cache
    CUATask(
        id="train_28_clear_instagram_cache",
        name="Clear Instagram App Cache",
        description="Navigate to Instagram's App Info > Storage and clear only the app cache.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=18,
        validation_type="state",
        validation_query="instagram_cache_cleared",
        expected_result=True,
        tags=["settings", "apps", "instagram", "cache"],
    ),

    # Task 29: Create folder in Pictures
    CUATask(
        id="train_29_create_pictures_folder",
        name="Create Folder in Pictures",
        description="Create a new folder named 'screenshots' in the Pictures directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Pictures/screenshots", "exists": True},
        tags=["system", "file", "folder", "pictures"],
    ),

    # Task 30: Uninstall Facebook
    CUATask(
        id="train_30_uninstall_facebook",
        name="Uninstall Facebook",
        description="Uninstall the Facebook app from the device.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=12,
        validation_type="state",
        validation_query="app_installed",
        expected_result={"package": "com.facebook.katana", "installed": False},
        tags=["system", "uninstall", "facebook"],
    ),

    # Task 31: Disable Chrome location permission
    CUATask(
        id="train_31_disable_chrome_location",
        name="Disable Chrome Location Permission",
        description="Go to Chrome app settings and disable the Location permission.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="app_permission",
        expected_result={"package": "com.android.chrome", "permission": "android.permission.ACCESS_FINE_LOCATION", "granted": False},
        tags=["settings", "apps", "chrome", "permissions", "location"],
    ),

    # Task 32: Check Instagram storage size
    CUATask(
        id="train_32_check_instagram_storage",
        name="Check Instagram Storage Size",
        description="Check how much storage space the Instagram app is using. You must report the exact storage size (include units like KB, MB, or GB) in your finish message.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=12,
        validation_type="api",
        validation_query="finish_message_contains_size",
        expected_result="storage_size_reported",
        tags=["settings", "apps", "storage", "instagram"],
    ),

    # Task 33: Create folder in Documents
    CUATask(
        id="train_33_create_documents_folder",
        name="Create Folder in Documents",
        description="Create a new folder named 'new_folder' in the Documents directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Documents/new_folder", "exists": True},
        tags=["system", "file", "folder", "documents"],
    ),

    # Task 34: Change media volume to 50%
    CUATask(
        id="train_34_media_volume_50",
        name="Set Media Volume to 50%",
        description="Open Sound settings or use volume controls to set media volume to 50%.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="media_volume",
        expected_result=50,
        tags=["settings", "sound", "volume", "media"],
    ),

    # Task 35: Change screen timeout to 5 minutes
    CUATask(
        id="train_35_timeout_5min",
        name="Change Screen Timeout to 5 Minutes",
        description="Open Settings, navigate to Display, and set the screen timeout to 5 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
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
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=600000,
        tags=["settings", "display", "timeout"],
    ),

    # Task 37: Create folder in Documents
    CUATask(
        id="train_37_create_documents_folder",
        name="Create Folder in Documents",
        description="Create a new folder named 'project_folder' in the Documents directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Documents/project_folder", "exists": True},
        tags=["system", "file", "folder", "documents"],
    ),

    # Task 38: Create folder in Documents
    CUATask(
        id="train_38_create_documents_folder",
        name="Create Folder in Documents",
        description="Create a new folder named 'work_folder' in the Documents directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Documents/work_folder", "exists": True},
        tags=["system", "file", "folder", "documents"],
    ),

    # Task 39: Create folder in Documents
    CUATask(
        id="train_39_create_documents_folder",
        name="Create Folder in Documents",
        description="Create a new folder named 'custom' in the Documents directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Documents/custom", "exists": True},
        tags=["system", "file", "folder", "documents"],
    ),

    # Task 40: Enable Facebook location permission
    CUATask(
        id="train_40_enable_facebook_location",
        name="Enable Facebook Location Permission",
        description="Go to Facebook app settings and enable the Location permission if it's disabled.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=14,
        validation_type="state",
        validation_query="app_permission",
        expected_result={"package": "com.facebook.katana", "permission": "android.permission.ACCESS_FINE_LOCATION", "granted": True},
        tags=["settings", "apps", "facebook", "permissions", "location"],
    ),

    # Task 41: Change screen timeout to 15 minutes
    CUATask(
        id="train_41_timeout_15min",
        name="Change Screen Timeout to 15 Minutes",
        description="Open Settings, navigate to Display, and set the screen timeout to 15 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=900000,
        tags=["settings", "display", "timeout"],
    ),

    # Task 42: Change media volume to 80%
    CUATask(
        id="train_42_media_volume_80",
        name="Set Media Volume to 80%",
        description="Open Sound settings or use volume controls to set media volume to 80%.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="media_volume",
        expected_result=80,
        tags=["settings", "sound", "volume", "media"],
    ),

    # Task 43: Clear all app notifications
    CUATask(
        id="train_43_clear_all_notifications",
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

    # Task 44: Enable Chrome microphone permission
    CUATask(
        id="train_44_enable_chrome_mic",
        name="Enable Chrome Microphone Permission",
        description="Go to Chrome app settings and enable the Microphone permission if it's disabled.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=14,
        validation_type="state",
        validation_query="app_permission",
        expected_result={"package": "com.android.chrome", "permission": "android.permission.RECORD_AUDIO", "granted": True},
        tags=["settings", "apps", "chrome", "permissions", "microphone"],
    ),

    # Task 45: Enable Chrome camera permission
    CUATask(
        id="train_45_enable_chrome_camera",
        name="Enable Chrome Camera Permission",
        description="Go to Chrome app settings and enable the Camera permission if it's disabled.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=14,
        validation_type="state",
        validation_query="app_permission",
        expected_result={"package": "com.android.chrome", "permission": "android.permission.CAMERA", "granted": True},
        tags=["settings", "apps", "chrome", "permissions", "camera"],
    ),

    # Task 46: Change media volume to 30%
    CUATask(
        id="train_46_media_volume_30",
        name="Set Media Volume to 30%",
        description="Open Sound settings or use volume controls to set media volume to 30%.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="media_volume",
        expected_result=30,
        tags=["settings", "sound", "volume", "media"],
    ),

    # Task 47: Change media volume to 60%
    CUATask(
        id="train_47_media_volume_60",
        name="Set Media Volume to 60%",
        description="Open Sound settings or use volume controls to set media volume to 60%.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=10,
        validation_type="state",
        validation_query="media_volume",
        expected_result=60,
        tags=["settings", "sound", "volume", "media"],
    ),

    # Task 48: Change screen timeout to 30 minutes
    CUATask(
        id="train_48_timeout_30min",
        name="Change Screen Timeout to 30 Minutes",
        description="Open Settings, navigate to Display, and set the screen timeout to 30 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="screen_timeout",
        expected_result=1800000,
        tags=["settings", "display", "timeout"],
    ),

]

# =============================================================================
# EVALUATION TASKS (16 tasks)
# =============================================================================

DEMO_EVAL_TASKS = [
    # Eval Task 1: Change display timeout to 5 minutes
    CUATask(
        id="eval_01_display_timeout_5min",
        name="Change Display Timeout to 5 Minutes",
        description="Open Settings, go to Display, and change the screen timeout to 5 minutes.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
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
        max_steps=10,
        validation_type="state",
        validation_query="dnd_enabled",
        expected_result=True,
        tags=["settings", "notifications", "dnd"],
    ),

    # Eval Task 3: Change system language
    CUATask(
        id="eval_03_change_language",
        name="Change System Language to Spanish",
        description="Open Language & input settings and change the system language to Spanish.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=24,
        validation_type="state",
        validation_query="system_language",
        expected_result="es",
        tags=["settings", "language", "system"],
    ),

    # Eval Task 4: Enable dark theme
    CUATask(
        id="eval_04_enable_dark_theme",
        name="Enable Dark Theme",
        description="Enable the system dark theme from Display settings or quick settings.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=12,
        validation_type="state",
        validation_query="dark_theme_enabled",
        expected_result=True,
        tags=["settings", "display", "theme", "dark"],
    ),


    # Eval Task 5: Enable battery saver
    CUATask(
        id="eval_05_enable_battery_saver",
        name="Enable Battery Saver",
        description="Open Battery settings and enable Battery Saver mode.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SETTINGS,
        max_steps=14,
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
        max_steps=12,
        validation_type="state",
        validation_query="app_installed",
        expected_result={"package": "com.instagram.android", "installed": False},
        tags=["system", "uninstall", "instagram"],
    ),

    # Eval Task 7: Disable Facebook contacts permission
    CUATask(
        id="eval_07_disable_facebook_contacts",
        name="Disable Facebook Contacts Permission",
        description="Go to Facebook app settings and disable the Contacts permission.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="app_permission",
        expected_result={"package": "com.facebook.katana", "permission": "android.permission.READ_CONTACTS", "granted": False},
        tags=["settings", "apps", "facebook", "permissions", "contacts"],
    ),

    # Eval Task 8: Clear Facebook cache
    CUATask(
        id="eval_08_clear_facebook_cache",
        name="Clear Facebook App Cache",
        description="Navigate to Facebook's App Info > Storage and clear only the app cache (not data).",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=18,
        validation_type="state",
        validation_query="facebook_cache_cleared",
        expected_result=True,
        tags=["settings", "apps", "facebook", "cache"],
    ),

    # Eval Task 9: Create folder in Downloads
    CUATask(
        id="eval_09_create_downloads_folder",
        name="Create Folder in Downloads",
        description="Create a new folder named 'abc' in the Downloads directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
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
        max_steps=12,
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
        max_steps=12,
        validation_type="api",
        validation_query="finish_message_contains",
        expected_result="51.54MB",
        tags=["settings", "apps", "storage", "chrome"],
    ),

    # Eval Task 13: Clear Instagram cache
    CUATask(
        id="eval_13_clear_instagram_cache",
        name="Clear Instagram App Cache",
        description="Navigate to Instagram's App Info > Storage and clear only the app cache.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=18,
        validation_type="state",
        validation_query="instagram_cache_cleared",
        expected_result=True,
        tags=["settings", "apps", "instagram", "cache"],
    ),

    # Eval Task 14: Disable Instagram location permission
    CUATask(
        id="eval_14_disable_instagram_location",
        name="Disable Instagram Location Permission",
        description="Go to Instagram app settings and disable the Location permission.",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.SYSTEM,
        max_steps=16,
        validation_type="state",
        validation_query="app_permission",
        expected_result={"package": "com.instagram.android", "permission": "android.permission.ACCESS_FINE_LOCATION", "granted": False},
        tags=["settings", "apps", "instagram", "permissions", "location"],
    ),

    # Eval Task 15: Create folder in Documents
    CUATask(
        id="eval_15_create_documents_folder",
        name="Create Folder in Documents",
        description="Create a new folder named 'test_folder' in the Documents directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
        validation_type="state",
        validation_query="file_exists",
        expected_result={"path": "/storage/emulated/0/Documents/test_folder", "exists": True},
        tags=["system", "file", "folder", "documents"],
    ),

    # Eval Task 16: Create folder in Downloads with different name
    CUATask(
        id="eval_16_create_downloads_folder_xyz",
        name="Create Folder in Downloads",
        description="Create a new folder named 'xyz' in the Downloads directory.",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.SYSTEM,
        max_steps=10,
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

