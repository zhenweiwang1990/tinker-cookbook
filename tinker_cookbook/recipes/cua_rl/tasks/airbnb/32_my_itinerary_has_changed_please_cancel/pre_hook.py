from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task32PreHook:
    """Prepare the database with seed data before running the UI flow."""

    def run(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "INSERT INTO payments (id, room_id, user_email, start_date, end_date, guest, amount, created_at, card_number, card_expiry, card_cvc, cardholder_name) VALUES ('1766126735166-ctytj98d','25936','local@airbnb.com','Fri Dec 19, 2025','Fri Dec 26, 2025','1 guest',1074,1766126735166,'4242 4242 4242 4242','02/28','567','111'),('1766126761551-b6bor0s3','24803','local@airbnb.com','Fri Dec 19, 2025','Fri Dec 26, 2025','1 guest',820,1766126761551,'4242 4242 4242 4242','02/28','567','wang');"
        ]
        for sql in queries:
            output = adb_client.run_sqlite_query(
                package_name=config.get_package_name(),
                db_relative_path="databases/airbnbSQLiteSQLite.db",
                sql=sql,
            )
            print(output)
        return True
