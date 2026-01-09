from __future__ import annotations

from dataclasses import dataclass

from ....adb import AdbClient
from ....executor.base import Task
from .validator import Task15Validator


@dataclass
class Task15(Task):
    name: str = "15_download_gbox_logo"
    description: str = """Navigate to gbox.ai website and download the logo file. Verify the downloaded file is named logo.svg."""

    def run(self, adb_client: AdbClient) -> bool:
        # TODO: implement UI steps for this task.
        return True

    def get_validator(self):
        return Task15Validator()

    def get_pre_hook(self):
        return None


def create_task() -> Task:
    return Task15()
