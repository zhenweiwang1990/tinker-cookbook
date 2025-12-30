from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task05Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "SELECT 1 FROM favorites f JOIN rooms r ON r.id = f.listing_id WHERE r.bed_count_count >= 1 AND r.price <= 70000 AND r.filter = 'Golfing' AND r.house_rating >= 4.7 LIMIT 1;"
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
