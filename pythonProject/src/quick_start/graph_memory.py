"""
langgraph学习, 记忆:
https://langgraph.com.cn/tutorials/get-started/3-add-memory/index.html

langgraph 的记忆采取检查点的设计思路,而非传统的记忆窗口。可以保存完整的置校状态,支持恢复和调式。
"""

from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langchain_deepseek import ChatDeepSeek

class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()

graph_builder = StateGraph(State)

# 初始化 DeepSeek 大模型客户端
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-8f6367ea6d3748578985f9bc16dbfa50"
)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile(checkpointer=memory)

config = {
    "configurable": {"thread_id": "1"}
}

user_input = "Hi there ! My name is Will"

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()

# 验证memory
user_input = "Remember my name?"
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    # {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()


shapshot = graph.get_state(config)
print(shapshot)