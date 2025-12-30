from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task29Validator


@dataclass
class Task29(Task):
    name: str = "29_i_want_to_stay_in_listings"
    description: str = """I want to stay in listings with Farms, region not limited, sort by price from high to low, then among the top 5 results find the house closest to me and save it."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task29Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task29()
