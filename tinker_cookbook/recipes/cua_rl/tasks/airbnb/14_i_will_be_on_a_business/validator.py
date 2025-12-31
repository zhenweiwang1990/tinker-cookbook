from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task14Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "SELECT 1 FROM messages m JOIN rooms r ON r.id = m.listing_id WHERE m.created_at = (SELECT MAX(created_at) FROM messages) AND r.city = 'San Francisco' AND r.price <= 30000 AND r.filter = 'New' AND r.house_title LIKE '%Beach%' LIMIT 1;"
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
