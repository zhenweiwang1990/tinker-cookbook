from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .pre_hook import Task23PreHook
from .validator import Task23Validator


@dataclass
class Task23(Task):
    name: str = "23_help_me_delete_the_chat_history"
    description: str = """Help me delete the chat history with Carl, delete all content."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task23Validator()

    def get_pre_hook(self):
        return Task23PreHook()



def create_task() -> Task:
    return Task23()
