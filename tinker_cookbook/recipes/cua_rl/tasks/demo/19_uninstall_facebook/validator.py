from __future__ import annotations

from ....adb import AdbClient


class Task19Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check if app is uninstalled. Avoid `pm ... | grep ...` because piping
        # is not reliably supported across environments.
        return not adb_client.is_installed("com.facebook.katana")

