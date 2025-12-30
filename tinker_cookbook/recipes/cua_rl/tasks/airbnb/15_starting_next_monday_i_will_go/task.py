from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task15Validator


@dataclass
class Task15(Task):
    name: str = "15_starting_next_monday_i_will_go"
    description: str = """Starting next Monday I will go to San Diego alone, help me book the cheapest listing, my credit card number 4242 4242 4242 4242, expiration date 05/2028, CVC 366, name GBOX."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task15Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task15()
