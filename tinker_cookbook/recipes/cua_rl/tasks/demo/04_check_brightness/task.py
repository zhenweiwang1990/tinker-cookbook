from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task04Validator


@dataclass
class Task04(Task):
    name: str = "04_check_brightness"
    description: str = """Open Settings and navigate to the Brightness section to check the current brightness level."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task04Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task04()
