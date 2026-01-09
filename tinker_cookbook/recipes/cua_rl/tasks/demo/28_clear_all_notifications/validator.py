from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task28Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check notification count
        query = "dumpsys notification | grep -c NotificationRecord"
        output = adb_client.shell_command(query)
        try:
            count = int(output.strip())
            return count == 0
        except ValueError:
            return False

