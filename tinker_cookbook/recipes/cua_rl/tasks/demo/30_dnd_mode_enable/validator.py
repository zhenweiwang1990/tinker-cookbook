from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task30Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check Do Not Disturb
        # Use full path for GBox compatibility
        query = "settings get global zen_mode"
        try:
            output = adb_client._run("shell", query, capture_output=True)
            return output.strip() != "0"  # non-zero means enabled
        except Exception:
            return False

