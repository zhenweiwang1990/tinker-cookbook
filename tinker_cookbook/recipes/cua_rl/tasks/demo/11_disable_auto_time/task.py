from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task11Validator


@dataclass
class Task11(Task):
    name: str = "11_disable_auto_time"
    description: str = """In Date & Time settings, disable automatic date & time."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task11Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task11()
