from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .pre_hook import Task18PreHook
from .validator import Task18Validator


@dataclass
class Task18(Task):
    name: str = "18_help_me_send_a_message_to"
    description: str = """Help me send a message to Carl asking him: "Where is the charger？"."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task18Validator()

    def get_pre_hook(self):
        return Task18PreHook()



def create_task() -> Task:
    return Task18()
