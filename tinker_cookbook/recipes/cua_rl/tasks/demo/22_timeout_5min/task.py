from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task22Validator


@dataclass
class Task22(Task):
    name: str = "22_timeout_5min"
    description: str = """Open Settings, navigate to Display, and set the screen timeout to 5 minutes."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task22Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task22()
