from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task05Validator


@dataclass
class Task05(Task):
    name: str = "05_disable_wifi"
    description: str = """Go to Settings and turn off WiFi if it's on."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task05Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task05()
