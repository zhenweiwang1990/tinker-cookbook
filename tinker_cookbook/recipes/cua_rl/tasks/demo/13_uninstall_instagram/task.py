from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task13Validator


@dataclass
class Task13(Task):
    name: str = "13_uninstall_instagram"
    description: str = """Uninstall the Instagram app from the device."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task13Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task13()
