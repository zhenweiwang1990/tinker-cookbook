from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task16Validator


@dataclass
class Task16(Task):
    name: str = "16_tomorrow_i_will_go_to_san"
    description: str = """Tomorrow I will go to San Francisco to play golf, help me book a listing near a golf course, I have no budget limit, book the most expensive listing, during booking use credit card number 4242 4242 4242 4242, expiration date 05/2028, CVC 366, name GBOX."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task16Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task16()
