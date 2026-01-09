from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task24Validator


@dataclass
class Task24(Task):
    name: str = "24_create_documents_folder_project"
    description: str = """Create a new folder named 'project_folder' in the Documents directory."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task24Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task24()
