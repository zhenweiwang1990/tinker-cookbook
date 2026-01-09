from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task31Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check if file exists
        query = "test -e /storage/emulated/0/Documents/test_folder && echo 1 || echo 0"
        output = adb_client.shell_command(query)
        return output.strip() == "1"

