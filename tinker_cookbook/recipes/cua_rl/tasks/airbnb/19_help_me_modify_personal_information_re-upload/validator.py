from __future__ import annotations

from typing import List

from ... import config
from ....adb import AdbClient


class Task19Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        queries: List[str] = [
            "SELECT 1 FROM user_profile WHERE json_extract(profile_json,'$.first_name') IS NOT NULL AND json_extract(profile_json,'$.first_name') != '' AND json_extract(profile_json,'$.last_name') IS NOT NULL AND json_extract(profile_json,'$.last_name') != '' AND avatar_base64 IS NOT NULL AND avatar_base64 != '' LIMIT 1;"
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
