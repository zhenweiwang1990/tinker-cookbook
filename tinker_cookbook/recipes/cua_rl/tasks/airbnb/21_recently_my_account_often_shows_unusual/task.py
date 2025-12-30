from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task21Validator


@dataclass
class Task21(Task):
    name: str = "21_recently_my_account_often_shows_unusual"
    description: str = """Recently my account often shows unusual location logins; in personal settings help me enable two-step verification, turn on the login alert feature, and set the Security PIN to 1234."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task21Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task21()
