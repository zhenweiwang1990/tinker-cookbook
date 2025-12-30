from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task06Validator


@dataclass
class Task06(Task):
    name: str = "06_please_help_me_plan_a_2-week"
    description: str = """Please help me plan a 2-week vacation, taking my wife and child. The first week I plan to spend in United States, the second week I plan to go to Arizona, budget is $300 per night, location can be anywhere, please find listings that meet the conditions and save them."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task06Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task06()
