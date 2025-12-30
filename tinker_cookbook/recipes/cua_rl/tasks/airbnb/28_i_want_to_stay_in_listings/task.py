from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task28Validator


@dataclass
class Task28(Task):
    name: str = "28_i_want_to_stay_in_listings"
    description: str = """I want to stay in listings with Amazing pools, region not limited, sort by price from low to high, then among the top 5 results find the house closest to me and save it."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task28Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task28()
