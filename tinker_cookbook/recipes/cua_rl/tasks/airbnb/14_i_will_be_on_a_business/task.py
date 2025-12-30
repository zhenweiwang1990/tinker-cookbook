from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task14Validator


@dataclass
class Task14(Task):
    name: str = "14_i_will_be_on_a_business"
    description: str = """I will be on a business trip to San Francisco for the next week, please try to find brand-new listings, they should be close to the beach, due to my limited budget keep the cost per night within $300, after finding the target listing send the host a message: "if there is WIFI network?"."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task14Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task14()
