from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task01Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check current app
        query = "dumpsys window windows | grep -E 'mCurrentFocus'"
        output = adb_client.shell_command(query)
        return "com.android.settings" in output

