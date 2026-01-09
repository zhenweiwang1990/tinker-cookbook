from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task27Validator


@dataclass
class Task27(Task):
    name: str = "27_timeout_15min"
    description: str = """Open Settings, navigate to Display, and set the screen timeout to 15 minutes."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task27Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task27()
