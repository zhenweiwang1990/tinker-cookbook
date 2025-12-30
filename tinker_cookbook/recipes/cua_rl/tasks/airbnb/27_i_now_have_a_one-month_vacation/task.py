from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task04Validator


@dataclass
class Task04(Task):
    name: str = "04_i_now_have_a_one-month_vacation"
    description: str = """I now have a one-month vacation, plan to take my wife, my wife's parents, and children to travel, destination can be anywhere in United States, total accommodation budget for 1 month is within $7000, please help me find a listing that you think has the best cost performance and is relatively large, and save it."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task04Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task04()
