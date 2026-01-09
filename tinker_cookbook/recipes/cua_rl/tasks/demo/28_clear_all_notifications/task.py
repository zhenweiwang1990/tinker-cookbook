from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task28Validator


@dataclass
class Task28(Task):
    name: str = "28_clear_all_notifications"
    description: str = """Open the notifications shade and clear all existing notifications."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task28Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task28()
