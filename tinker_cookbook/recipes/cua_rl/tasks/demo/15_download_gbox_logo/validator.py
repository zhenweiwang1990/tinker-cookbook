from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task15Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check if file exists
        query = "test -e /storage/emulated/0/Download/logo.svg && echo 1 || echo 0"
        output = adb_client._run("shell", query, capture_output=True)
        return output.strip() == "1"

