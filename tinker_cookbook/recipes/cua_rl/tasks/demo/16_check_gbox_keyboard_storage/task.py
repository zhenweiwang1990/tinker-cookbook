from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task16Validator


@dataclass
class Task16(Task):
    name: str = "16_check_gbox_keyboard_storage"
    description: str = """Check how much storage space the GBOX Keyboard app is using. You must report the exact storage size (e.g., '73.73KB') in your finish message. The correct answer is 73.73KB."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task16Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task16()
