from __future__ import annotations

from ....adb import AdbClient


class Task01Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check current foreground app. Use AdbClient helper instead of a piped shell
        # command (pipes are not reliably supported across local ADB vs GBox modes).
        current_app = adb_client.get_current_app()
        return current_app == "com.android.settings"

