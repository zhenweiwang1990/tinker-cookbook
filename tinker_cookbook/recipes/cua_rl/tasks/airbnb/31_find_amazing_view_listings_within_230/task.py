from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task31Validator


@dataclass
class Task31(Task):
    name: str = "31_find_amazing_view_listings_within_230"
    description: str = """Find Amazing view listings within 230 kilometers and the lowest priced listing among them, save it."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task31Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task31()
