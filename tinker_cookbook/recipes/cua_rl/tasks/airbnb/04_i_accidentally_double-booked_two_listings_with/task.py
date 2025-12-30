from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .pre_hook import Task27PreHook
from .validator import Task27Validator


@dataclass
class Task27(Task):
    name: str = "27_i_accidentally_double-booked_two_listings_with"
    description: str = """I accidentally double-booked two listings with overlapping dates and now I lack budget, please help me cancel the higher-priced listing."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task27Validator()

    def get_pre_hook(self):
        return Task27PreHook()



def create_task() -> Task:
    return Task27()
