"""
Agentic AI
解决v1版本graph运行结构问题现象
1. 添加了quality_defects字段来进行结果分析。
2. 添加了max_human_review_rounds来控制审核轮次,避免无线节点循环
"""

from typing import TypedDict, List, Dict, Annotated, Any
from uuid import uuid4
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import AnyMessage, AIMessage, ToolMessage, HumanMessage
from langgraph.types import interrupt, Command
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


# 自定义状态
class ReportAgentState(TypedDict):
    # message
    messages: Annotated[List[AnyMessage], add_messages]
    # 报告状态
    user_requirement: str  # 用户报告需求
    raw_data: str  # 业务数据
    draft_report: str  # 报告草稿
    check_result: str  # 校验结果
    quality_defects: List[str]  # 质检缺陷
    is_satisfied: bool  # 是否满足要求
    review_round: int  # 评审轮次


# 工具定义
@tool
def fetch_business_data(requirement: str,
                        tool_call_id: Annotated[str, InjectedToolCallId]
                        ) -> Command:
    """ 获取报告所需的业务数据内容 """

    # MOCK 业务数据
    data_map: Dict[str, str] = {
        "2024年销售报告": (
            "2024年1-6月销售额：100万、200万、150万、300万、280万、350万；"
            "同比增长20%；华东区域贡献45%。"
        ),
        "用户分析报告": (
            "用户总数：50万；新增用户：8万；留存率：75%；核心用户占比：30%；"
            "月均活跃时长提升12%。"
        ),
        "LangGraph发布报告": (
            "LangGraph发布时间：2024年1月17日；所属公司：LangChain；"
            "核心特性：状态管理、工具调用、循环执行、人机协同。"
        ),
    }
    matched_key = requirement
    if "LangGraph" in requirement and ("发布时间" in requirement or "发布" in requirement):
        matched_key = "LangGraph发布报告"
    elif "销售" in requirement:
        matched_key = "2024年销售报告"
    elif "用户" in requirement:
        matched_key = "用户分析报告"

    raw_data = data_map.get(matched_key, f"未查询到「{requirement}」相关数据")

    # 状态构造更新
    state_update = {
        "raw_data": raw_data,
        "messages": [ToolMessage(f"已获取数据：{raw_data}", tool_call_id=tool_call_id)]
    }

    return Command(update=state_update)


@tool
def save_draft_report(
        draft_report: str,
        tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """ 将生成的草稿转换为图状态 """
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
        tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """ 对报告进行质量检查 """

    text = draft_report.strip()
    text_lower = text.lower()
    defects: List[str] = []

    # 长度校验
    if len(text) < 120:
        defects.append("报告内容过短，需至少120字")

    # 结构校验,添加中英文符号
    has_background = "背景" in text or "background" in text_lower
    has_analysis = "分析" in text or "analysis" in text_lower
    has_conclusion = "结论" in text or "conclusion" in text_lower
    if not (has_background and has_analysis and has_conclusion):
        defects.append("报告结构不完整，需包含背景/分析/结论")

    release_time_keywords = [
        "发布时间", "发布日期", "发布于", "发布于", "release date", "released on", "release time"
    ]
    if "langgraph" in text_lower and not any(k in text_lower for k in release_time_keywords):
        defects.append("LangGraph报告缺少发布时间信息")

    # 核心能力覆盖校验
    capability_keywords = [
        "状态管理", "工具调用", "循环执行", "人机协同",
        "state management", "tool calling", "cyclic execution", "human in the loop"
    ]
    capability_hits = sum(1 for k in capability_keywords if k in text_lower)
    if capability_hits < 2:
        defects.append("核心能力覆盖不足，至少覆盖2项关键特性")

    if defects:
        check_res = "不达标：" + ";".join(defects)
        is_satisfied = False
    else:
        check_res = "达标：报告内容符合要求"
        is_satisfied = True

    # 状态更新
    state_update = {
        "check_result": check_res,
        "quality_defects": defects,
        "is_satisfied": is_satisfied,
        "messages": [ToolMessage(f"报告检测结果:{check_res}", tool_call_id=tool_call_id)]
    }

    return Command(update=state_update)


@tool
def human_review_report(
        draft_report: str,
        check_result: str,
        review_round: int,
        tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """ 请求人工进行审核报告 """
    # 暂停输入,等待人工审核(human in the loop)

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

    # 状态更新
    state_update = {
        "draft_report": revised_report,
        "check_result": next_check_result,
        "quality_defects": defects,
        "is_satisfied": is_satisfied,
        "review_round": review_round + 1,
        "messages": [ToolMessage(feedback, tool_call_id=tool_call_id)]
    }

    return Command(update=state_update)


# LLM 配置
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-8f6367ea6d3748578985f9bc16dbfa50"
)

# 工具绑定
tools = [
    fetch_business_data,
    save_draft_report,
    check_report_quality,
    human_review_report,
]


def _need_human_review(
        requirement: str,
        review_round: int
) -> bool:
    requirement_lower = requirement.lower()
    return (
            review_round == 0
            and (
                    "human" in requirement_lower
                    or "review" in requirement_lower
                    or "人工" in requirement
                    or "审核" in requirement
            )
    )


def _manual_tool_call_message(
        *,
        tool_name: str,
        args: Dict
) -> AIMessage:
    """构造稳定的工具调用消息，避免模型未按预期触发tool_call。"""
    return AIMessage(
        content=f"调用工具：{tool_name}",
        tool_calls=[{
            "id": f"manual_{tool_name}_{uuid4().hex[:8]}",
            "name": tool_name,
            "args": args,
            "type": "tool_call"
        }]
    )


# Agents 节点的定义
def report_chatbot(state: ReportAgentState):
    """ 核心chatbot节点,根据状态进行下一步动作的决策(工具调用&报告生成) """
    # 提取状态信息
    user_requirement = state.get("user_requirement") or state["messages"][-1].content
    raw_data = state.get("raw_data", "")
    draft_report = state.get("draft_report", "")
    check_result = state.get("check_result", "")
    quality_defects = state.get("quality_defects", [])
    is_satisfied = state.get("is_satisfied", False)
    review_round = state.get("review_round", 0)
    max_human_review_rounds = 2

    # 决策逻辑
    if not raw_data:
        return {
            "user_requirement": user_requirement,
            "messages": [
                _manual_tool_call_message(
                    tool_name="fetch_business_data",
                    args={"requirement": user_requirement}
                )
            ]
        }
    elif not draft_report:
        prompt = (
            f"用户需求：{user_requirement}\n"
            f"业务数据：{raw_data}\n"
            "请输出一份结构化报告（含背景、分析、结论，至少120字）。"
        )
        report_text = llm.invoke([HumanMessage(content=prompt)]).content
        return {
            "user_requirement": user_requirement,
            "messages": [
                _manual_tool_call_message(
                    tool_name="save_draft_report",
                    args={"draft_report": str(report_text).strip()}
                )
            ]
        }
    elif not check_result:
        return {
            "user_requirement": user_requirement,
            "messages": [
                _manual_tool_call_message(
                    tool_name="check_report_quality",
                    args={"draft_report": draft_report}
                )
            ]
        }
    elif not is_satisfied:
        # 超过人工审核轮次后强制终止并输出缺陷，避免流程卡死
        if review_round >= max_human_review_rounds:
            return {
                "user_requirement": user_requirement,
                "messages": [AIMessage(
                    content=(
                        "流程结束：已达到最大人工审核轮次，仍未达标。\n"
                        f"当前质检结果：{check_result}\n"
                        f"缺陷清单：{'; '.join(quality_defects) if quality_defects else '无'}\n"
                        f"当前草稿：\n{draft_report}"
                    )
                )]
            }
        else:
            return {
                "user_requirement": user_requirement,
                "messages": [
                    _manual_tool_call_message(
                        tool_name="human_review_report",
                        args={
                            "draft_report": draft_report,
                            "check_result": check_result,
                            "review_round": review_round
                        }
                    )
                ]
            }
    elif _need_human_review(user_requirement,review_round):
        return {
            "user_requirement": user_requirement,
            "messages": [
                _manual_tool_call_message(
                    tool_name="human_review_report",
                    args={
                        "draft_report": draft_report,
                        "check_result": check_result,
                        "review_round": review_round
                    }
                )
            ]
        }
    else:
        return {
            "user_requirement": user_requirement,
            "messages": [AIMessage(content=f"任务完成，最终报告如下：\n{draft_report}")]
        }


# 构建Graph
def build_graph():
    graph_builder = StateGraph(ReportAgentState)
    graph_builder.add_node("report_chatbot", report_chatbot)
    graph_builder.add_node("tools", ToolNode(tools=tools))

    graph_builder.set_entry_point("report_chatbot")
    graph_builder.add_conditional_edges(
        "report_chatbot",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    graph_builder.add_edge("tools", "report_chatbot")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# 运行Agent
if __name__ == "__main__":
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
        # 第1轮给出完整修订稿，尽量一次通过质检；后续轮次默认通过
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
                "reason": "根据质检意见补充发布时间信息，并完善结构与数据解读。"
            }
        return {
            "approved": "y",
            "revised_report": snapshot.values.get("draft_report", ""),
            "reason": "人工确认通过。"
        }


    graph = build_graph()
    config = {
        "recursion_limit": 40,
        "configurable": {
            "thread_id": "report_agent_1"
        }
    }
    user_input = (
        "请查询LangGraph发布时间，生成报告，并在最终交付前让human review一次。"
    )

    events = graph.stream({
        "messages": [HumanMessage(content=user_input)],
        "user_requirement": user_input,
        "raw_data": "",
        "draft_report": "",
        "check_result": "",
        "quality_defects": [],
        "is_satisfied": False,
        "review_round": 0,
    }, config, stream_mode="values")

    _print_events(events)

    # 模拟人工审核：持续resume直到没有待处理中断
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

    # 查看最终的执行状态
    print("\n===== 最终状态 =====")
    shapshot = graph.get_state(config)
    final_state = {
        k: v for k, v in shapshot.values.items()
        if k in {"user_requirement", "raw_data", "draft_report", "check_result", "quality_defects", "is_satisfied"}
    }
    for k, v in final_state.items():
        print(f"{k}: {v}")

    for state in graph.get_state_history(config):
        print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
