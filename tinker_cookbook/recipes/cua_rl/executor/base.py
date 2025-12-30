"""Base classes for task execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


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

