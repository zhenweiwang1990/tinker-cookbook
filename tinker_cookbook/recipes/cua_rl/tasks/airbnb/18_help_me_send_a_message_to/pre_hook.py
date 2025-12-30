from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task18PreHook:
    """Prepare the database with seed data before running the UI flow."""

    def run(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "INSERT INTO messages (id, thread_id, listing_id, host_name, host_id, user_email, body, created_at) VALUES ('1766124291409-d8fo6u5g','25936-Carl','25936','Carl','Carl','local@airbnb.com','111',1766124291409);"
        ]
        for sql in queries:
            output = adb_client.run_sqlite_query(
                package_name=config.get_package_name(),
                db_relative_path="databases/airbnbSQLiteSQLite.db",
                sql=sql,
            )
            print(output)
        return True
