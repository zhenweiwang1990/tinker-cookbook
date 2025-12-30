from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task10Validator


@dataclass
class Task10(Task):
    name: str = "10_this_weekend_i_want_to_travel"
    description: str = """This weekend I want to travel to San Diego with my wife for two days, help me find a listing that has a swimming pool, is a Guest favourite, rating greater than or equal to 4.9 and is among the top three by price, no budget limit, please save it."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task10Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task10()
