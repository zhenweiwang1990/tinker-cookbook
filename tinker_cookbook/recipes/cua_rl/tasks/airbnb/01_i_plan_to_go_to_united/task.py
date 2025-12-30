from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task01Validator


@dataclass
class Task01(Task):
    name: str = "01_i_plan_to_go_to_united"
    description: str = """I plan to go to United States, there are currently 2 adults and 1 infant, can depart at any time, plan to travel for 7 days, budget per night within $700, must have a swimming pool, help me save 3 listings that meet the conditions for me to compare."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task01Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task01()
