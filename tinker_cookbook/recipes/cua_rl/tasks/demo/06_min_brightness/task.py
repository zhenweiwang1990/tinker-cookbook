from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task06Validator


@dataclass
class Task06(Task):
    name: str = "06_min_brightness"
    description: str = """Open Settings and set the screen brightness to the minimum level."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task06Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task06()
