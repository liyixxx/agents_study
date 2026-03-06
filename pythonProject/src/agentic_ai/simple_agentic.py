"""
Agentic AI
运用智能体生成报告示例
"""

from typing import TypedDict, List, Dict, Annotated
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

    # Mock检测逻辑,添加一些业务逻辑
    if len(draft_report) < 100:
        check_res = "不达标：报告内容过短，缺少核心数据解读"
        is_satisfied = False
    elif "同步增长" not in draft_report and "销售" in draft_report:
        check_res = "不达标：销售报告缺少同步增长分析"
        is_satisfied = False
    elif "发布时间" not in draft_report and "LangGraph" in draft_report:
        check_res = "不达标：LangGraph报告缺少发布时间"
        is_satisfied = False
    else:
        check_res = "达标：报告内容符合要求"
        is_satisfied = True

    # 状态更新
    state_update = {
        "check_result": check_res,
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
    else:
        next_check_result = ""
        is_satisfied = False
        feedback = "人工提出修改意见，回到质检环节"

    # 状态更新
    state_update = {
        "draft_report": revised_report,
        "check_result": next_check_result,
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
    is_satisfied = state.get("is_satisfied", False)
    review_round = state.get("review_round", 0)

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
        if review_round >= 2:
            prompt = (
                f"当前草稿：{draft_report}\n"
                f"质检结果：{check_result}\n"
                "请改写报告，补全缺失信息，输出完整修订版报告（至少120字）。"
            )
            revised_text = llm.invoke([HumanMessage(content=prompt)]).content
            return {
                "user_requirement": user_requirement,
                "messages": [
                    _manual_tool_call_message(
                        tool_name="save_draft_report",
                        args={"draft_report": str(revised_text).strip()}
                    )
                ]
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
        "is_satisfied": False,
        "review_round": 0,
    }, config, stream_mode="values")

    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # 模拟人工审核
    print("\n===== 人工审核阶段 =====")

    human_command = Command(
        resume={
            "approved": "n",  # approved=y 表示通过; n表示需要修正
            "revised_report": "LangGraph Official Release Report\n1. Release Date: Jan 17, 2024\n2. Developer: LangChain\n3. Core Features: State management, tool calling, cyclic execution\n4. Application Scenario: Agent development",
            "reason": "补充核心特性和应用场景，优化报告结构"
        }
    )
    snapshot_before_resume = graph.get_state(config)
    if snapshot_before_resume.next:
        events = graph.stream(
            human_command,
            config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
    else:
        print("当前没有待恢复的interrupt，Command(resume=...) 未执行。")

    # 查看最终的执行状态
    print("\n===== 最终状态 =====")
    shapshot = graph.get_state(config)
    final_state = {
        k: v for k, v in shapshot.values.items()
        if k in {"user_requirement", "raw_data", "draft_report", "is_satisfied"}
    }
    for k, v in final_state.items():
        print(f"{k}: {v}")

    for state in graph.get_state_history(config):
        print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)