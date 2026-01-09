from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task18Validator


@dataclass
class Task18(Task):
    name: str = "18_check_chrome_storage"
    description: str = """Check how much storage space the Chrome app is using. You must report the exact storage size (include units like KB, MB, or GB) in your finish message."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task18Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task18()
