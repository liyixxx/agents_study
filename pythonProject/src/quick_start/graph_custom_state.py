"""
langgraph学习: 自定义状态
https://langgraph.com.cn/tutorials/get-started/5-customize-state/index.html
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, InjectedToolCallId
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph
from langchain_core.messages import ToolMessage
from langgraph.types import interrupt, Command


# 自定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


@tool
def human_assistance(
        name: str,
        birthday: str,
        tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    # interrupt: 暂停图的执行,等待人工输入确认
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        }
    )
    # 如果信息正确，则按原样更新状态。
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)]
    }

    return Command(update=state_update)


llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-8f6367ea6d3748578985f9bc16dbfa50"
)
search_tool = TavilySearch(max_results=2, tavily_api_key="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")

tools = [human_assistance, search_tool]

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert (len(message.tool_calls) <= 1)
    return {"messages": [message]}


tool_node = ToolNode(tools=tools)

# 构建graph信息
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")

# 添加检查点
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
)
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# 添加人工信息确认
human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    }
)
# 这次将人工信息进行输入
events = graph.stream(
    human_command,
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# 查看状态
shapshot = graph.get_state(config)
print({k: v for k, v in shapshot.values.items() if k in {"name", "birthday"}})


# 手动更新状态
graph.update_state(config,{"name":"LangGraph(library)"})
shapshot_update = graph.get_state(config)
print({k: v for k, v in shapshot_update.values.items() if k in {"name", "birthday"}})


