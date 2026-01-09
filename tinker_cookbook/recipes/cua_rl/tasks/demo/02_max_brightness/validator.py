from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task02Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check brightness level
        # Use full path for GBox compatibility
        query = "settings get system screen_brightness"
        try:
            output = adb_client._run("shell", query, capture_output=True)
            brightness = int(output.strip())
            return brightness == 255
        except (ValueError, Exception):
            return False

