from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task11Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check auto time setting
        query = "settings get global auto_time"
        output = adb_client.shell_command(query)
        return output.strip() == "0"  # 0 means disabled

