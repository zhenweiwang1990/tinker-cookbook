from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task02Validator


@dataclass
class Task02(Task):
    name: str = "02_max_brightness"
    description: str = """Open Settings and set the screen brightness to maximum."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task02Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task02()
