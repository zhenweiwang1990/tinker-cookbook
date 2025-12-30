from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .pre_hook import Task24PreHook
from .validator import Task24Validator


@dataclass
class Task24(Task):
    name: str = "24_previously_i_chatted_with_several_hosts"
    description: str = """Previously I chatted with several hosts, please help me delete the chat history with SOMA House, keep all other chat histories."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task24Validator()

    def get_pre_hook(self):
        return Task24PreHook()



def create_task() -> Task:
    return Task24()
