from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task30Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check Do Not Disturb
        query = "settings get global zen_mode"
        output = adb_client.shell_command(query)
        return output.strip() != "0"  # non-zero means enabled

