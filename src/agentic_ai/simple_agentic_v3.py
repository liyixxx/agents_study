"""
Agentic AI
解决 v1/v2 版本中的配置分散与数据源能力不足问题
1. 抽取公共配置到 env_util。
2. 将知识库与 graph checkpoint 统一切换到 PostgreSQL。
"""

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, TypedDict
from uuid import uuid4

import psycopg
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agentic_ai.cfg.report_cfg import Report

from src.util.env_util import (
    get_llm,
    get_postgres_connection_string,
    get_search_tool,
)

default_report_cfg = Report()

class ReportAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_requirement: str
    raw_data: str
    evidence_items: List[Dict[str, str]]
    draft_report: str
    check_result: str
    quality_defects: List[str]
    is_satisfied: bool
    review_round: int


class DataProvider:
    name = "base"

    def query(self, requirement: str) -> List[Dict[str, str]]:
        raise NotImplementedError


class PostgresProvider(DataProvider):
    name = "postgres"

    def __init__(self, connection_string: str, seed_sql_path: Path, max_rows: int = 3):
        self.connection_string = connection_string
        self.seed_sql_path = seed_sql_path
        self.max_rows = max_rows
        self.available = True
        self._init_db()

    def _init_db(self) -> None:
        try:
            with psycopg.connect(self.connection_string, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS report_knowledge (
                            id BIGSERIAL PRIMARY KEY,
                            topic TEXT NOT NULL,
                            content TEXT NOT NULL,
                            source TEXT NOT NULL,
                            url TEXT DEFAULT '',
                            published_at TEXT DEFAULT '',
                            tags TEXT DEFAULT '',
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_report_knowledge_topic ON report_knowledge(topic)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_report_knowledge_tags ON report_knowledge(tags)"
                    )
                    cur.execute("SELECT COUNT(1) FROM report_knowledge")
                    count = cur.fetchone()[0]
                    if count == 0 and self.seed_sql_path.exists():
                        seed_sql = self.seed_sql_path.read_text(encoding="utf-8")
                        for statement in _split_sql_statements(seed_sql):
                            cur.execute(statement)
        except psycopg.Error:
            self.available = False

    def query(self, requirement: str) -> List[Dict[str, str]]:
        if not self.available:
            return []

        terms = _extract_query_terms(requirement)
        where_clauses: List[str] = []
        params: List[Any] = []

        for term in terms:
            like_term = f"%{term}%"
            where_clauses.append(
                "(topic ILIKE %s OR tags ILIKE %s OR content ILIKE %s)"
            )
            params.extend([like_term, like_term, like_term])

        sql = (
            "SELECT topic, content, source, url, published_at "
            "FROM report_knowledge "
        )
        if where_clauses:
            sql += "WHERE " + " OR ".join(where_clauses) + " "
        sql += "ORDER BY updated_at DESC LIMIT %s"
        params.append(self.max_rows)

        try:
            with psycopg.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        except psycopg.Error:
            return []

        items: List[Dict[str, str]] = []
        for topic, content, source, url, published_at in rows:
            items.append(
                {
                    "topic": topic or "",
                    "content": content or "",
                    "source": source or self.name,
                    "url": url or "",
                    "published_at": str(published_at) if published_at else "",
                }
            )
        return items


class WebProvider(DataProvider):
    name = "web"

    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.client = get_search_tool(max_results=max_results)
        self.enabled = self.client is not None

    def query(self, requirement: str) -> List[Dict[str, str]]:
        if not self.enabled or self.client is None:
            return []
        try:
            result = self.client.invoke(requirement)
        except Exception:
            return []

        raw_items = result.get("results", []) if isinstance(result, dict) else []
        items: List[Dict[str, str]] = []
        for item in raw_items[: self.max_results]:
            items.append(
                {
                    "topic": item.get("title", ""),
                    "content": item.get("content", ""),
                    "source": self.name,
                    "url": item.get("url", ""),
                    "published_at": item.get("published_date", ""),
                }
            )
        return items


def _split_sql_statements(sql_text: str) -> List[str]:
    return [statement.strip() for statement in sql_text.split(";") if statement.strip()]


def _extract_query_terms(requirement: str) -> List[str]:
    requirement_lower = requirement.lower()
    terms: List[str] = []

    if "langgraph" in requirement_lower:
        terms.extend(["langgraph", "发布", "发布时间"])
    if "销售" in requirement:
        terms.extend(["销售", "同比增长"])
    if "用户" in requirement:
        terms.extend(["用户", "留存"])

    if not terms:
        tokens = [token for token in re.split(r"[\s,，。；;]+", requirement_lower) if len(token) >= 2]
        terms.extend(tokens[:4])

    seen = set()
    dedup_terms = []
    for term in terms:
        if term not in seen:
            dedup_terms.append(term)
            seen.add(term)
    return dedup_terms


def _compose_raw_data(evidence_items: List[Dict[str, str]]) -> str:
    if not evidence_items:
        return ""
    return "；".join(item.get("content", "") for item in evidence_items if item.get("content"))


@lru_cache(maxsize=1)
def _get_data_providers() -> List[DataProvider]:
    mode = default_report_cfg.get_report_data_mode()
    connection_string = default_report_cfg.get_postgres_connection_string()
    seed_sql_path = default_report_cfg.get_report_seed_sql_path()
    max_rows = default_report_cfg.get_report_max_rows()

    providers: List[DataProvider] = []
    if mode in {"db", "postgres", "hybrid"}:
        providers.append(
            PostgresProvider(
                connection_string=connection_string,
                seed_sql_path=seed_sql_path,
                max_rows=max_rows,
            )
        )
    if mode in {"web", "hybrid"}:
        providers.append(WebProvider(max_results=max_rows))
    return providers


@tool
def fetch_business_data(
        requirement: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """获取报告所需的业务数据内容。"""

    evidence_items: List[Dict[str, str]] = []
    source_used = "none"
    for provider in _get_data_providers():
        evidence_items = provider.query(requirement)
        if evidence_items:
            source_used = provider.name
            break

    raw_data = _compose_raw_data(evidence_items)
    if not raw_data:
        raw_data = f"未查询到「{requirement}」相关数据，请补充关键词或检查 PostgreSQL/Web 配置。"

    source_summary = ", ".join(
        f"{item.get('source', '')}:{item.get('topic', '')}" for item in evidence_items
    ) or "无"

    return Command(
        update={
            "raw_data": raw_data,
            "evidence_items": evidence_items,
            "messages": [
                ToolMessage(
                    f"已获取数据（source={source_used}）：{raw_data}\n证据来源：{source_summary}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def save_draft_report(
        draft_report: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """将生成的草稿转换为图状态。"""
    return Command(
        update={
            "draft_report": draft_report,
            "check_result": "",
            "quality_defects": [],
            "is_satisfied": False,
            "messages": [ToolMessage(content="报告草稿已保存", tool_call_id=tool_call_id)],
        }
    )


@tool
def check_report_quality(
        draft_report: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """对报告进行规则化质量检查。"""

    text = draft_report.strip()
    text_lower = text.lower()
    defects: List[str] = []

    if len(text) < 120:
        defects.append("报告内容过短，需至少120字")

    has_background = "背景" in text or "background" in text_lower
    has_analysis = "分析" in text or "analysis" in text_lower
    has_conclusion = "结论" in text or "conclusion" in text_lower
    if not (has_background and has_analysis and has_conclusion):
        defects.append("报告结构不完整，需包含背景/分析/结论")

    release_time_keywords = [
        "发布时间",
        "发布日期",
        "发布于",
        "release date",
        "released on",
        "release time",
    ]
    if "langgraph" in text_lower and not any(keyword in text_lower for keyword in release_time_keywords):
        defects.append("LangGraph报告缺少发布时间信息")

    capability_keywords = [
        "状态管理",
        "工具调用",
        "循环执行",
        "人机协同",
        "state management",
        "tool calling",
        "cyclic execution",
        "human in the loop",
    ]
    capability_hits = sum(1 for keyword in capability_keywords if keyword in text_lower)
    if capability_hits < 2:
        defects.append("核心能力覆盖不足，至少覆盖2项关键特性")

    if defects:
        check_res = "不达标：" + ";".join(defects)
        is_satisfied = False
    else:
        check_res = "达标：报告内容符合要求"
        is_satisfied = True

    return Command(
        update={
            "check_result": check_res,
            "quality_defects": defects,
            "is_satisfied": is_satisfied,
            "messages": [ToolMessage(f"报告检测结果:{check_res}", tool_call_id=tool_call_id)],
        }
    )


@tool
def human_review_report(
        draft_report: str,
        check_result: str,
        review_round: int,
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """请求人工进行审核报告。"""

    human_response = interrupt(
        {
            "question": "请审核当前报告，approved=y 表示通过；否则可给 revised_report。",
            "draft_report": draft_report,
            "check_result": check_result,
            "review_round": review_round,
        }
    )
    approved = str(human_response.get("approved", "")).lower().startswith("y")
    revised_report = human_response.get("revised_report", draft_report).strip()

    if approved:
        next_check_result = "人工审核通过"
        is_satisfied = True
        feedback = "人工审核通过，流程结束"
        defects = []
    else:
        next_check_result = ""
        is_satisfied = False
        feedback = "人工提出修改意见，回到质检环节"
        defects = []

    return Command(
        update={
            "draft_report": revised_report,
            "check_result": next_check_result,
            "quality_defects": defects,
            "is_satisfied": is_satisfied,
            "review_round": review_round + 1,
            "messages": [ToolMessage(feedback, tool_call_id=tool_call_id)],
        }
    )


tools = [
    fetch_business_data,
    save_draft_report,
    check_report_quality,
    human_review_report,
]


@lru_cache(maxsize=1)
def _get_agent_llm():
    return get_llm()


def _need_human_review(requirement: str, review_round: int) -> bool:
    requirement_lower = requirement.lower()
    return review_round == 0 and (
            "human" in requirement_lower
            or "review" in requirement_lower
            or "人工" in requirement
            or "审核" in requirement
    )


def _manual_tool_call_message(*, tool_name: str, args: Dict[str, Any]) -> AIMessage:
    """构造稳定的工具调用消息，避免模型未按预期触发 tool_call。"""
    return AIMessage(
        content=f"调用工具：{tool_name}",
        tool_calls=[
            {
                "id": f"manual_{tool_name}_{uuid4().hex[:8]}",
                "name": tool_name,
                "args": args,
                "type": "tool_call",
            }
        ],
    )


def report_chatbot(state: ReportAgentState):
    """核心 chatbot 节点，根据状态进行下一步动作决策。"""
    user_requirement = state.get("user_requirement") or state["messages"][-1].content
    raw_data = state.get("raw_data", "")
    evidence_items = state.get("evidence_items", [])
    draft_report = state.get("draft_report", "")
    check_result = state.get("check_result", "")
    quality_defects = state.get("quality_defects", [])
    is_satisfied = state.get("is_satisfied", False)
    review_round = state.get("review_round", 0)
    max_human_review_rounds = default_report_cfg.get_report_max_review_rounds()

    if not raw_data:
        return {
            "user_requirement": user_requirement,
            "messages": [
                _manual_tool_call_message(
                    tool_name="fetch_business_data",
                    args={"requirement": user_requirement},
                )
            ],
        }

    if not draft_report:
        evidence_text = "\n".join(
            [
                f"- source={item.get('source', '')}, topic={item.get('topic', '')}, "
                f"published_at={item.get('published_at', '')}, url={item.get('url', '')}, "
                f"content={item.get('content', '')}"
                for item in evidence_items
            ]
        ) or "无可用证据"
        prompt = (
            f"用户需求：{user_requirement}\n"
            f"业务数据：{raw_data}\n"
            f"证据列表：\n{evidence_text}\n"
            "请基于证据输出一份结构化报告（含背景、分析、结论，至少120字），并尽量引用可核对信息。"
        )
        report_text = _get_agent_llm().invoke([HumanMessage(content=prompt)]).content
        return {
            "user_requirement": user_requirement,
            "messages": [
                _manual_tool_call_message(
                    tool_name="save_draft_report",
                    args={"draft_report": str(report_text).strip()},
                )
            ],
        }

    if not check_result:
        return {
            "user_requirement": user_requirement,
            "messages": [
                _manual_tool_call_message(
                    tool_name="check_report_quality",
                    args={"draft_report": draft_report},
                )
            ],
        }

    if not is_satisfied:
        if review_round >= max_human_review_rounds:
            return {
                "user_requirement": user_requirement,
                "messages": [
                    AIMessage(
                        content=(
                            "流程结束：已达到最大人工审核轮次，仍未达标。\n"
                            f"当前质检结果：{check_result}\n"
                            f"缺陷清单：{'; '.join(quality_defects) if quality_defects else '无'}\n"
                            f"当前草稿：\n{draft_report}"
                        )
                    )
                ],
            }
        return {
            "user_requirement": user_requirement,
            "messages": [
                _manual_tool_call_message(
                    tool_name="human_review_report",
                    args={
                        "draft_report": draft_report,
                        "check_result": check_result,
                        "review_round": review_round,
                    },
                )
            ],
        }

    if _need_human_review(user_requirement, review_round):
        return {
            "user_requirement": user_requirement,
            "messages": [
                _manual_tool_call_message(
                    tool_name="human_review_report",
                    args={
                        "draft_report": draft_report,
                        "check_result": check_result,
                        "review_round": review_round,
                    },
                )
            ],
        }

    return {
        "user_requirement": user_requirement,
        "messages": [AIMessage(content=f"任务完成，最终报告如下：\n{draft_report}")],
    }


def build_graph(checkpointer: Any):
    graph_builder = StateGraph(ReportAgentState)
    graph_builder.add_node("report_chatbot", report_chatbot)
    graph_builder.add_node("tools", ToolNode(tools=tools))

    graph_builder.set_entry_point("report_chatbot")
    graph_builder.add_conditional_edges(
        "report_chatbot",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    graph_builder.add_edge("tools", "report_chatbot")
    return graph_builder.compile(checkpointer=checkpointer)


@contextmanager
def graph_session() -> Iterator[Any]:
    """保持 Postgres checkpointer 连接在整个 graph 执行期间有效。"""
    connection_string = get_postgres_connection_string()
    with PostgresSaver.from_conn_string(connection_string) as checkpointer:
        checkpointer.setup()
        yield build_graph(checkpointer)


def _print_events(stream_events):
    for event in stream_events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


def _has_pending_interrupt(snapshot) -> bool:
    for task in getattr(snapshot, "tasks", []) or []:
        if getattr(task, "interrupts", None):
            return True
    return False


def _build_resume_payload(snapshot) -> Dict[str, Any]:
    review_round = snapshot.values.get("review_round", 0)
    if review_round == 0:
        return {
            "approved": "n",
            "revised_report": (
                "LangGraph官方发布报告\n"
                "【背景】LangGraph由LangChain于2024年1月17日发布，定位于复杂智能体流程编排。\n"
                "【分析】该框架强调状态管理、工具调用、循环执行与人机协同，解决多步骤任务中状态难维护、流程难回溯的问题。\n"
                "在企业应用中，可用于复杂对话系统、自动化流程决策和跨工具任务执行。\n"
                "【结论】LangGraph的发布时间明确、核心能力完整，适合作为有状态Agent系统的基础设施。"
            ),
            "reason": "根据质检意见补充发布时间信息，并完善结构与数据解读。",
        }
    return {
        "approved": "y",
        "revised_report": snapshot.values.get("draft_report", ""),
        "reason": "人工确认通过。",
    }


if __name__ == "__main__":
    config = {
        "recursion_limit": default_report_cfg.get_report_recursion_limit(),
        "configurable": {
            "thread_id": default_report_cfg.get_report_thread_id(),
        }
    }
    user_input = "请查询LangGraph发布时间，生成报告，并在最终交付前让human review一次。"

    with graph_session() as graph:
        events = graph.stream(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_requirement": user_input,
                "raw_data": "",
                "evidence_items": [],
                "draft_report": "",
                "check_result": "",
                "quality_defects": [],
                "is_satisfied": False,
                "review_round": 0,
            },
            config,
            stream_mode="values",
        )
        _print_events(events)

        print("\n===== 人工审核阶段 =====")
        max_resume_attempts = 3
        for i in range(max_resume_attempts):
            snapshot = graph.get_state(config)
            if not _has_pending_interrupt(snapshot):
                break
            human_command = Command(resume=_build_resume_payload(snapshot))
            print(f"\n--- 第{i + 1}次人工恢复 ---")
            events = graph.stream(human_command, config, stream_mode="values")
            _print_events(events)
        else:
            print("达到最大人工恢复次数，流程仍未完成。")

        print("\n===== 最终状态 =====")
        snapshot = graph.get_state(config)
        final_state = {
            key: value
            for key, value in snapshot.values.items()
            if key in {
                "user_requirement",
                "raw_data",
                "evidence_items",
                "draft_report",
                "check_result",
                "quality_defects",
                "is_satisfied",
            }
        }
        for key, value in final_state.items():
            print(f"{key}: {value}")

        for history_state in graph.get_state_history(config):
            print("Num Messages: ", len(history_state.values["messages"]), "Next: ", history_state.next)
