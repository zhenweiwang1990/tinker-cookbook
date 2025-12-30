from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task09Validator


@dataclass
class Task09(Task):
    name: str = "09_i_have_one_week_off_and"
    description: str = """I have one week off and will go alone, I want to experience some Airbnb unique rooms (Amazing views), help me find the cheapest listing and save it."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task09Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task09()
