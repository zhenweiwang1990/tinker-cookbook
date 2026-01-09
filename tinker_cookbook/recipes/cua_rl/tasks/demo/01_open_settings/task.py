from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task01Validator


@dataclass
class Task01(Task):
    name: str = "01_open_settings"
    description: str = """Open the Settings app from the home screen."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task01Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task01()
