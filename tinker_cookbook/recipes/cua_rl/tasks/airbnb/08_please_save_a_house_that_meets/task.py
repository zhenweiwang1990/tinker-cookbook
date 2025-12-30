from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task08Validator


@dataclass
class Task08(Task):
    name: str = "08_please_save_a_house_that_meets"
    description: str = """Please save a house that meets the following conditions: 2 adults, 1 child; has a farm, and can be checked in within the next 3 days, and is the lowest priced listing."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task08Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task08()
