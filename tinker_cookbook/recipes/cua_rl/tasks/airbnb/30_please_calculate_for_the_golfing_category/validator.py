from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task30Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "SELECT 1 FROM (SELECT COUNT(*) AS cnt, COUNT(DISTINCT listing_id) AS distinct_cnt FROM favorites WHERE listing_id IN (25723,25719,25724,25716)) t WHERE t.cnt = 4 AND t.distinct_cnt = 4;"
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
