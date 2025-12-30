from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task30Validator


@dataclass
class Task30(Task):
    name: str = "30_please_calculate_for_the_golfing_category"
    description: str = """Please calculate for the Golfing category the average rating (keep 1 decimal place) of the top 10 listings sorted by price from high to low (if a listing has no rating ignore it), then save all listings whose rating is above the average."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task30Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task30()
