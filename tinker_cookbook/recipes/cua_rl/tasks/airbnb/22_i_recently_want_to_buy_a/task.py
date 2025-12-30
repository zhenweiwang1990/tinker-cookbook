from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task02Validator


@dataclass
class Task02(Task):
    name: str = "02_i_recently_want_to_buy_a"
    description: str = """I recently want to buy a vacation home, so I really want to stay in a tiny house for 7 days to experience it; location can be anywhere in United States, just me, price per night not exceeding $700, choose rooms with rating greater than or equal to 4.7 and save them."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task02Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task02()
