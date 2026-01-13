from __future__ import annotations

from ....adb import AdbClient


class Task16Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Validate by checking the agent's completion message (finish tool message).
        msg = getattr(adb_client, "result_message", None) or ""
        return "106" in msg.lower()

