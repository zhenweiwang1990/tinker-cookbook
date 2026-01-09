from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task05Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check wifi status
        query = "settings get global wifi_on"
        output = adb_client.shell_command(query)
        return output.strip() == "0"  # 0 means disabled

