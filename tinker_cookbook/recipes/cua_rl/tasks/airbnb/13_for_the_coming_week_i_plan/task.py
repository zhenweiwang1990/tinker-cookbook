from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task13Validator


@dataclass
class Task13(Task):
    name: str = "13_for_the_coming_week_i_plan"
    description: str = """For the coming week I plan to stay in some more interesting rooms, please use the Trending category to pick the most popular rooms, select the 2 lowest-priced listings, and message the hosts to ask whether it's possible to check in during the coming week."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task13Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task13()
