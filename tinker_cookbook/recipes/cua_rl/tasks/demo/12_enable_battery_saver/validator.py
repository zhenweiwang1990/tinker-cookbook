from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task12Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check battery saver
        query = "settings get global low_power"
        output = adb_client.shell_command(query)
        return output.strip() == "1"

