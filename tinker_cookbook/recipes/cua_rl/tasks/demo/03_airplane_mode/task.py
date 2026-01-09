from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task03Validator


@dataclass
class Task03(Task):
    name: str = "03_airplane_mode"
    description: str = """Go to Settings and turn on Airplane Mode."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task03Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task03()
