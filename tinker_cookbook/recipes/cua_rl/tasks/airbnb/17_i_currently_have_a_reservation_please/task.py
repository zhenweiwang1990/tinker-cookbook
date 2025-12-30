from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .pre_hook import Task17PreHook
from .validator import Task17Validator


@dataclass
class Task17(Task):
    name: str = "17_i_currently_have_a_reservation_please"
    description: str = """I currently have a reservation, please help me cancel it, then rebook the cheapest listing in San Diego for 7 days starting tomorrow, my credit card number 4242 4242 4242 4242, expiration date 05/2028, CVC 366, name GBOX."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task17Validator()

    def get_pre_hook(self):
        return Task17PreHook()



def create_task() -> Task:
    return Task17()
