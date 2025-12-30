from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task07Validator


@dataclass
class Task07(Task):
    name: str = "07_i_have_a_one-week_vacation_to"
    description: str = """I have a one-week vacation to go anywhere, taking my wife, my child (high school student) and a pet, we hope to stay in a listing with a swimming pool, budget within $700, save 1 listing that meets the conditions."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task07Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task07()
