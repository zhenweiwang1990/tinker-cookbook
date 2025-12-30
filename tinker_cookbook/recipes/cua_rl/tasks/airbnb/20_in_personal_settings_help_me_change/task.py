from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task20Validator


@dataclass
class Task20(Task):
    name: str = "20_in_personal_settings_help_me_change"
    description: str = """In personal settings help me change the currency unit to CNY, timezone to UTC+8, and work email totest@gbox.ai."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task20Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task20()
