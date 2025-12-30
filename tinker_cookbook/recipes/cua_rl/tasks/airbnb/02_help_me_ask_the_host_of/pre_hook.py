from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task22PreHook:
    """Prepare the database with seed data before running the UI flow."""

    def run(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "INSERT INTO payments (id, room_id, user_email, start_date, end_date, guest, amount, created_at, card_number, card_expiry, card_cvc, cardholder_name) VALUES ('1766119602001-lfihh7f2','25936','local@airbnb.com','Fri Dec 19, 2025','Thu Dec 25, 2025','1 guest',933,1766119602001,'4242 4242 4242 4242','02/28','567','wa');"
        ]
        for sql in queries:
            output = adb_client.run_sqlite_query(
                package_name=config.get_package_name(),
                db_relative_path="databases/airbnbSQLiteSQLite.db",
                sql=sql,
            )
            print(output)
        return True
