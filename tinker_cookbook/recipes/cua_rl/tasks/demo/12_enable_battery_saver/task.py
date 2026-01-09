from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task12Validator


@dataclass
class Task12(Task):
    name: str = "12_enable_battery_saver"
    description: str = """Open Settings, go to Battery, and enable Battery Saver mode."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task12Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task12()
