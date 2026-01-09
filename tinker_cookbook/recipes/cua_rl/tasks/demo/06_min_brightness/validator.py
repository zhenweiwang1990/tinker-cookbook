from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task06Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check brightness level is at minimum
        # Android minimum brightness is typically 0-10 depending on device
        query = "settings get system screen_brightness"
        output = adb_client.shell_command(query)
        try:
            brightness = int(output.strip())
            # Consider brightness as minimum if it's <= 10
            # This accounts for different Android versions and devices
            return brightness <= 10
        except ValueError:
            return False

