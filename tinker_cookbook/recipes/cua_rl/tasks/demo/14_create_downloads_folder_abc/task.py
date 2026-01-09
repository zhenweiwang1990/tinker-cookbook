from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task14Validator


@dataclass
class Task14(Task):
    name: str = "14_create_downloads_folder_abc"
    description: str = """Create a new folder named 'abc' in the Downloads directory."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task14Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task14()
