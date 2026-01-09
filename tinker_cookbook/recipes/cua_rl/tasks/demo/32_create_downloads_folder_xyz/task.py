from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task32Validator


@dataclass
class Task32(Task):
    name: str = "32_create_downloads_folder_xyz"
    description: str = """Create a new folder named 'xyz' in the Downloads directory."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task32Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task32()
