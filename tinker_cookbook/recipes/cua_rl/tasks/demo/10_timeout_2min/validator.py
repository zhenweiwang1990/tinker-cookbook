from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task10Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check screen timeout
        # Use full path for GBox compatibility
        query = "settings get system screen_off_timeout"
        output = adb_client._run("shell", query, capture_output=True)
        try:
            timeout = int(output.strip())
            return timeout == 120000
        except (ValueError, Exception):
            return False

