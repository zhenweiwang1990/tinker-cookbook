from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task03Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check airplane mode
        # Use full path for GBox compatibility
        query = "settings get global airplane_mode_on"
        try:
            output = adb_client._run("shell", query, capture_output=True)
            return output.strip() == "1"
        except Exception:
            return False

