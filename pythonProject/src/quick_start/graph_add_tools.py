"""
langgraph学习,添加工具:
https://langgraph.com.cn/tutorials/get-started/2-add-tools/index.html#8-ask-the-bot-questions
"""

from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langchain_tavily import TavilySearch
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import ToolNode, tools_condition

# 定义状态类型,包含消息列表
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 定义工具
search_tool = TavilySearch(max_results = 2, tavily_api_key="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")

tools = [search_tool]

# 定义模型并绑定工具
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-8f6367ea6d3748578985f9bc16dbfa50"
)
llm_with_tools = llm.bind_tools(tools)

# 定义chatbot
def chatbot(state:State):
    return {
        "messages":[llm_with_tools.invoke(state["messages"])]
    }

# 定义图
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",ToolNode(tools))
# start -> chatbot
graph_builder.set_entry_point("chatbot")
# chatbot -> tools
graph_builder.add_conditional_edges("chatbot",tools_condition)
# tools -> chatbot
graph_builder.add_edge("tools","chatbot")

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    messages = [
        {"role": "system", "content": "你是一个准确、简洁的助手。"},
        {"role": "user", "content": user_input},
    ]

    for event in graph.stream({"messages":messages}):
        for node_name, node_output in event.items():
            last_msg = node_output["messages"][-1]
            content = getattr(last_msg, "content", str(last_msg))
            print(f"[{node_name}] {content}")


if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except Exception as e:
            print("发生错误，尝试默认问题...")
            print(e)
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)


