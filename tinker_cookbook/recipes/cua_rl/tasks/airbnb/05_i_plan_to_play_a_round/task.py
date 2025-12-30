from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task05Validator


@dataclass
class Task05(Task):
    name: str = "05_i_plan_to_play_a_round"
    description: str = """I plan to play a round of golf this weekend with my wife, need to stay two nights, help me find listings that meet the conditions, each night price not exceeding $700, preferably with sea view nearby, rating must be greater than or equal to 4.7, and save them."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task05Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task05()
