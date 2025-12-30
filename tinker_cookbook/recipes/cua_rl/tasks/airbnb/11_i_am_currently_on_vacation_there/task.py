from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task11Validator


@dataclass
class Task11(Task):
    name: str = "11_i_am_currently_on_vacation_there"
    description: str = """I am currently on vacation, there are 3 adults, 1 infant and 1 pet, help me find listings that can be checked in from tomorrow for the next 3 days, sort by price, and save the lowest priced listing."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task11Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task11()
