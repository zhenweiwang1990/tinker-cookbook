from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task12Validator


@dataclass
class Task12(Task):
    name: str = "12_i_plan_to_play_near_national"
    description: str = """I plan to play near national parks in United States, I have 2 adults and 1 child, please help me find the 5 cheapest listings that meet the conditions, then send messages to the 5 hosts asking whether they have availability in the coming week."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task12Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task12()
