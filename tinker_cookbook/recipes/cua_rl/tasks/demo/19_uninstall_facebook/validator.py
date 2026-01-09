from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task19Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check if app is uninstalled
        query = "pm list packages | grep com.facebook.katana"
        output = adb_client.shell_command(query)
        return not bool(output.strip())

