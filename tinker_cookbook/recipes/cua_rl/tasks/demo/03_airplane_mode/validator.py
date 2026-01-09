from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task03Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check airplane mode
        query = "settings get global airplane_mode_on"
        output = adb_client.shell_command(query)
        return output.strip() == "1"

