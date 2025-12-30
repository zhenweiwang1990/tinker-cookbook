from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task16Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "SELECT 1 FROM payments WHERE room_id = 25723 AND cardholder_name = ? AND card_cvc = ? AND card_expiry = ? AND card_number = ? ORDER BY created_at DESC LIMIT 1;"
        ]
        for sql in queries:
            output = adb_client.run_sqlite_query(
                package_name=config.get_package_name(),
                db_relative_path="databases/airbnbSQLiteSQLite.db",
                sql=sql,
            )
            print(output)
            if not output.strip():
                return False
        return True
