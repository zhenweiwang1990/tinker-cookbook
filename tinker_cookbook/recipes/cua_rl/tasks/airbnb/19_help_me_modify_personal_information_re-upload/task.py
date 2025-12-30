from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task19Validator


@dataclass
class Task19(Task):
    name: str = "19_help_me_modify_personal_information_re-upload"
    description: str = """Help me modify personal information: re-upload an avatar using the last photo in the album, then change my family name to GBOX and my given name to GBOX."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task19Validator()

    def get_pre_hook(self):
        return None



def create_task() -> Task:
    return Task19()
