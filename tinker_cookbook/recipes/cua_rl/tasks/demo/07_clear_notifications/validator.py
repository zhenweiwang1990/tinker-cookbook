from __future__ import annotations

from ....adb import AdbClient


class Task07Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Check notification count. Avoid `dumpsys ... | grep -c ...` because
        # piping is not reliably supported across environments.
        output = adb_client._run("shell", "dumpsys notification", capture_output=True)
        # Conservative: count NotificationRecord occurrences.
        count = output.count("NotificationRecord")
        return count == 0

