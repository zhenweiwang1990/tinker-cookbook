from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task19Validator


@dataclass
class Task19(Task):
    name: str = "19_uninstall_facebook"
    description: str = """Uninstall the Facebook app from the device."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task19Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task19()
