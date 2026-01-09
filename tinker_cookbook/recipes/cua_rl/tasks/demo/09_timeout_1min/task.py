from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task09Validator


@dataclass
class Task09(Task):
    name: str = "09_timeout_1min"
    description: str = """Open Settings, navigate to Display, and set the screen timeout to 1 minute."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task09Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task09()
