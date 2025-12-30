from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task25PreHook:
    """Prepare the database with seed data before running the UI flow."""

    def run(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "INSERT INTO favorites (listing_id, created_at) VALUES ('25936',1766126401438),('24803',1766126402656);"
        ]
        for sql in queries:
            output = adb_client.run_sqlite_query(
                package_name=config.get_package_name(),
                db_relative_path="databases/airbnbSQLiteSQLite.db",
                sql=sql,
            )
            print(output)
        return True
