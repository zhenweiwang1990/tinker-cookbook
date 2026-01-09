from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task25Validator


@dataclass
class Task25(Task):
    name: str = "25_create_documents_folder_work"
    description: str = """Create a new folder named 'work_folder' in the Documents directory."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task25Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task25()
