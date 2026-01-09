from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task06Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check brightness level is at minimum
        # Android minimum brightness is typically 0-10 depending on device
        # Use full path for GBox compatibility
        query = "settings get system screen_brightness"
        try:
            output = adb_client._run("shell", query, capture_output=True)
            brightness = int(output.strip())
            # Consider brightness as minimum if it's <= 10
            # This accounts for different Android versions and devices
            return brightness <= 10
        except (ValueError, Exception):
            # Fallback: if settings command fails, return False
            return False

