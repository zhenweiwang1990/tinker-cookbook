from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task30Validator


@dataclass
class Task30(Task):
    name: str = "30_dnd_mode_enable"
    description: str = """Enable Do Not Disturb mode from quick settings or the Settings app."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task30Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task30()
