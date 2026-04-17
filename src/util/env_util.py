from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import tomllib
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch

# 配置文件路径
ROOT_DIR = Path(__file__).resolve().parents[2]
RESOURCE_DIR = ROOT_DIR / "resource"
DEFAULT_CONFIG_PATH = RESOURCE_DIR / "runtime_config.toml"
DEFAULT_REPORT_SEED_SQL_PATH = RESOURCE_DIR / "report_agent_knowledge_seed.sql"

def _deep_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """从配置字典中递归获取嵌套键的值"""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_path(value: str | Path, *, default_base: Path = ROOT_DIR) -> Path:
    """解析配置文件路径"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (default_base / path).resolve()


@lru_cache(maxsize=1)
def get_app_config() -> dict[str, Any]:
    """获取APP基础配置"""
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {DEFAULT_CONFIG_PATH}")
    with DEFAULT_CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)


# 获取 LLM 实例
def get_llm(model: Optional[str] = None, temperature: Optional[float] = None) -> ChatDeepSeek:
    config = get_app_config()
    llm_config = _deep_get(config, "llm", default={}) or {}
    llm_kwargs = {
        "model": model or llm_config.get("model", "deepseek-chat"),
        "api_key": llm_config.get("api_key", ""),
    }
    final_temperature = temperature if temperature is not None else llm_config.get("temperature")
    if final_temperature is not None:
        llm_kwargs["temperature"] = final_temperature
    return ChatDeepSeek(**llm_kwargs)


# 获取 Tavily 搜索工具实例
def get_search_tool(max_results: Optional[int] = None):
    config = get_app_config()
    tavily_config = _deep_get(config, "tavily", default={}) or {}
    if TavilySearch is None or not tavily_config.get("api_key"):
        return None
    return TavilySearch(
        max_results=max_results or tavily_config.get("max_results", 3),
        tavily_api_key=tavily_config["api_key"],
    )


# 获取 PostgreSQL 连接
def get_postgres_connection_string(database: Optional[str] = None) -> str:
    config = get_app_config()
    postgres_config = _deep_get(config, "postgres", default={}) or {}
    user = quote_plus(str(postgres_config.get("user", "postgres")))
    password = quote_plus(str(postgres_config.get("password", "postgres")))
    host = str(postgres_config.get("host", "localhost"))
    port = int(postgres_config.get("port", 5432))
    db_name = database or str(postgres_config.get("database", "langgraph_db"))
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
