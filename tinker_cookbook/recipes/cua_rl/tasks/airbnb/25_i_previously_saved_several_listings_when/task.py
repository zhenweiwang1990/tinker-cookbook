from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .pre_hook import Task25PreHook
from .validator import Task25Validator


@dataclass
class Task25(Task):
    name: str = "25_i_previously_saved_several_listings_when"
    description: str = """I previously saved several listings when planning trips, please help me un-save all the saved listings now, I no longer have travel plans."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task25Validator()

    def get_pre_hook(self):
        return Task25PreHook()



def create_task() -> Task:
    return Task25()
