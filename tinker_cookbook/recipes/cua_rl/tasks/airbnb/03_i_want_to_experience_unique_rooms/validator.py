from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task03Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "SELECT 1 FROM (SELECT COUNT(*) AS cnt FROM favorites f JOIN rooms r ON r.id = f.listing_id WHERE r.bed_count_count >= 1 AND r.price <= 70000 AND r.filter = 'Amazing views' AND r.guest_favorite = 'Guest favourite') t WHERE t.cnt = 2;"
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
