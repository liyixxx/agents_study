"""
langgraph 学习: 时间旅行
https://langgraph.com.cn/tutorials/get-started/6-time-travel/index.html
"""

from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langchain_tavily import TavilySearch
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearch(max_results = 4, tavily_api_key="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")

tools = [search_tool]

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-8f6367ea6d3748578985f9bc16dbfa50"
)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state:State):
    return {
        "messages":[llm_with_tools.invoke(state["messages"])]
    }

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",ToolNode(tools))
graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools","chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# 重播完整历史,graph每一步都会保存检查点,跨越调用,因此可以再完整线程的历史中回溯
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # 选择一个状态,这里选择的是第六个message
        # Num Messages:  6 Next:  ('tools',)
        to_replay = state

# print(to_replay.next)
# print(to_replay.config)

# 从to_replay加载执行
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()

# print(to_replay.next)

