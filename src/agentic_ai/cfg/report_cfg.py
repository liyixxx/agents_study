from pathlib import Path

from src.util.env_util import (
    _deep_get,
    _resolve_path,
    get_app_config,
    get_postgres_connection_string,
)

class Report():
    def get_report_data_mode(self) -> str:
        config = get_app_config()
        return str(_deep_get(config, "report", "data_mode", default="hybrid")).lower()

    def get_report_thread_id(self) -> str:
        config = get_app_config()
        return str(_deep_get(config, "report", "thread_id", default="report_agent_1"))

    def get_report_recursion_limit(self) -> int:
        config = get_app_config()
        return int(_deep_get(config, "report", "recursion_limit", default=40))

    def get_report_max_rows(self) -> int:
        config = get_app_config()
        return int(_deep_get(config, "report", "db_max_rows", default=3))

    def get_report_max_review_rounds(self) -> int:
        config = get_app_config()
        return int(_deep_get(config, "report", "max_review_rounds", default=2))

    def get_report_seed_sql_path(self) -> Path:
        config = get_app_config()
        raw_path = _deep_get(
            config,
            "report",
            "seed_sql_path"
        )
        return _resolve_path(raw_path)

    def get_postgres_connection_string(self, database: str | None = None) -> str:
        return get_postgres_connection_string(database=database)
