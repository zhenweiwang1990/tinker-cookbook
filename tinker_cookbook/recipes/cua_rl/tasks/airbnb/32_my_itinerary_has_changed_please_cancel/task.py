from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .pre_hook import Task32PreHook
from .validator import Task32Validator


@dataclass
class Task32(Task):
    name: str = "32_my_itinerary_has_changed_please_cancel"
    description: str = """My itinerary has changed, please cancel all my already booked trips and listings, but before canceling send messages to all hosts to apologize and hope they understand."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task32Validator()

    def get_pre_hook(self):
        return Task32PreHook()



def create_task() -> Task:
    return Task32()
