from __future__ import annotations

from ....adb import AdbClient


class Task18Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # Validate by checking the agent's completion message (finish tool message).
        # Expected: finish message reports a storage size with units.
        import re

        msg = getattr(adb_client, "result_message", None) or ""
        if not msg:
            return False
        return "57" in msg.lower()

