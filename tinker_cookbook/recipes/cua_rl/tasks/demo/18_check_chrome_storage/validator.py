from __future__ import annotations

from ... import config
from ....adb import AdbClient


class Task18Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        # For API validation, check the finish message
        # This should be validated by the training system checking agent's finish message
        # Expected: finish message contains "storage_size_reported"
        return True  # Placeholder - actual validation done by system

