from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .pre_hook import Task22PreHook
from .validator import Task22Validator


@dataclass
class Task22(Task):
    name: str = "22_help_me_ask_the_host_of"
    description: str = """Help me ask the host of the listing I have already booked, send him a message asking whether I can bring my cat."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task22Validator()

    def get_pre_hook(self):
        return Task22PreHook()



def create_task() -> Task:
    return Task22()
