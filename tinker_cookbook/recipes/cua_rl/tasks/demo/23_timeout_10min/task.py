from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task23Validator


@dataclass
class Task23(Task):
    name: str = "23_timeout_10min"
    description: str = """Open Settings, navigate to Display, and set the screen timeout to 10 minutes."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task23Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task23()
