"""Base classes for task execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ApkConfig:
    """Configuration for APK installation and task environment."""
    
    # App name for identification (e.g., "demo", "airbnb", "instagram")
    app_name: str = "unknown"
    
    # CUA guide text for system prompt (how to describe what to operate)
    # Examples: 
    #   "You need to operate the android device to complete the task."
    #   "You need to operate the Airbnb app to complete the task."
    cua_guide: str = "You need to operate the system to complete the task."
    
    # Whether to install an APK
    requires_apk: bool = False
    
    # APK URL to download and install
    apk_url: Optional[str] = None
    
    # Package name of the app
    package_name: Optional[str] = None
    
    # Whether to launch the app after installation
    launch_after_install: bool = True
    
    # Main activity to launch (optional, will use default launcher if not specified)
    main_activity: Optional[str] = None


class Task(ABC):
    """Base class for tasks."""
    
    name: str
    description: str
    
    @abstractmethod
    def run(self, adb_client) -> bool:
        """Run the task.
        
        Args:
            adb_client: AdbClient instance to interact with device
            
        Returns:
            True if task completed successfully
        """
        pass
    
    @abstractmethod
    def get_validator(self):
        """Get validator for this task."""
        pass
    
    @abstractmethod
    def get_pre_hook(self) -> Optional[object]:
        """Get pre-hook for this task (optional)."""
        pass
    
    def get_apk_config(self) -> ApkConfig:
        """Get APK configuration for this task.
        
        By default, tasks import config from their parent module.
        Subclasses can override this method for custom behavior.
        
        Returns:
            ApkConfig object specifying APK installation requirements
        """
        # Try to import config from task's module
        import importlib
        import inspect
        
        # Get the module where the task class is defined
        task_module = inspect.getmodule(self.__class__)
        if task_module is None:
            # If module cannot be determined, return default (no APK)
            return ApkConfig()
        
        # Get the package path (e.g., tinker_cookbook.recipes.cua_rl.tasks.demo.01_open_settings.task)
        module_parts = task_module.__name__.split('.')
        
        # Find the config module in the task category
        # e.g., tinker_cookbook.recipes.cua_rl.tasks.demo.config
        if 'tasks' in module_parts:
            tasks_idx = module_parts.index('tasks')
            if tasks_idx + 1 < len(module_parts):
                # Get the category (demo, airbnb, instagram, etc.)
                category = module_parts[tasks_idx + 1]
                config_module_name = '.'.join(module_parts[:tasks_idx + 2]) + '.config'
                
                try:
                    config_module = importlib.import_module(config_module_name)
                    if hasattr(config_module, 'get_apk_config'):
                        return config_module.get_apk_config()
                except (ImportError, AttributeError):
                    pass
        
        # Default: no APK required
        return ApkConfig()

