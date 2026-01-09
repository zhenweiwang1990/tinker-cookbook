from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task08Validator


@dataclass
class Task08(Task):
    name: str = "08_timeout_30s"
    description: str = """Open Settings, navigate to Display, and set the screen timeout to 30 seconds."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task08Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task08()
