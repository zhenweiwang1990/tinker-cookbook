from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task03Validator


@dataclass
class Task03(Task):
    name: str = "03_i_want_to_experience_unique_rooms"
    description: str = """I want to experience unique rooms, preferably with distinctive views (Amazing views), just me, my budget is $700 per night, rating requirement is greater than or equal to 4.7, and preferably Guest favourite, save 2 listings that meet the conditions for me to compare."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task03Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task03()
