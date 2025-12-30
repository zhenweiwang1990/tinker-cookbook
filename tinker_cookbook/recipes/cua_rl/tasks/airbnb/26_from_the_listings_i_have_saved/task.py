from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .pre_hook import Task26PreHook
from .validator import Task26Validator


@dataclass
class Task26(Task):
    name: str = "26_from_the_listings_i_have_saved"
    description: str = """From the listings I have saved, un-save the highest priced listing, and keep only one saved listing which is the lowest priced."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task26Validator()

    def get_pre_hook(self):
        return Task26PreHook()



def create_task() -> Task:
    return Task26()
