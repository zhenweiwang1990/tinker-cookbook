from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task13Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check if app is uninstalled
        query = "pm list packages | grep com.instagram.android"
        output = adb_client.shell_command(query)
        return not bool(output.strip())

