from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task20Validator


@dataclass
class Task20(Task):
    name: str = "20_check_instagram_storage"
    description: str = """Check how much storage space the Instagram app is using. You must report the exact storage size (include units like KB, MB, or GB) in your finish message."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task20Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task20()
