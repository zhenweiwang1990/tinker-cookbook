from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task13Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "SELECT 1 FROM (SELECT listing_id FROM messages ORDER BY created_at DESC LIMIT 5) m HAVING COUNT(*) = 5 AND SUM(listing_id = 25239) >= 1 AND SUM(listing_id = 25233) >= 1 AND SUM(CASE WHEN listing_id NOT IN (25239,25233) THEN 1 ELSE 0 END) = 0;"
        ]
        for sql in queries:
            output = adb_client.run_sqlite_query(
                package_name=config.get_package_name(),
                db_relative_path="/data/data/"+config.get_package_name()+"/databases/airbnbSQLiteSQLite.db",
                sql=sql,
            )
            print(output)
            if not output.strip():
                return False
        return True
