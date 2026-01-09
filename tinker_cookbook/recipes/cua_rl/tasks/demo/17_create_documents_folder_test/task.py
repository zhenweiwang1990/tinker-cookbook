from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task17Validator


@dataclass
class Task17(Task):
    name: str = "17_create_documents_folder_test"
    description: str = """Create a new folder named 'test' in the Documents directory."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task17Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task17()
