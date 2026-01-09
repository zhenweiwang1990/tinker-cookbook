from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task23Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check screen timeout
        query = "settings get system screen_off_timeout"
        output = adb_client.shell_command(query)
        try:
            timeout = int(output.strip())
            return timeout == 600000
        except ValueError:
            return False

