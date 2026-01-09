from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task02Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check brightness level
        query = "settings get system screen_brightness"
        output = adb_client.shell_command(query)
        try:
            brightness = int(output.strip())
            return brightness == 255
        except ValueError:
            return False

