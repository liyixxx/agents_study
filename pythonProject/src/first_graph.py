from typing import Annotated
from langchain_deepseek import ChatDeepSeek
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 初始化状态图
graph_builder = StateGraph(State)

# 初始化 DeepSeek 大模型客户端
llm = ChatDeepSeek(
    model="deepseek-chat",  # 指定 DeepSeek 的模型名称
    api_key="sk-8f6367ea6d3748578985f9bc16dbfa50"  # 替换为您自己的 DeepSeek API 密钥
)

def chatbot(state: State):
    # 调用 DeepSeek 大模型生成回复
    return {"messages": [llm.invoke(state["messages"])]}

# 添加节点到状态图
graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("chatbot")  # 设置入口点
graph_builder.set_finish_point("chatbot")  # 设置结束点

# 编译状态图
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    # 构造对话历史，包含系统提示词和用户输入
    messages = [
        {"role": "system", "content": "你是一个有创意的助手，擅长根据用户问题提供有趣且相关的内容。输出内容长度不超过100个字。"},
        {"role": "user", "content": user_input},
    ]

    # 流式输出模型生成的回复
    for event in graph.stream({"messages": messages}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        # 获取用户输入
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # 调用流式更新函数
        stream_graph_updates(user_input)
    except Exception as e:
        # 捕获异常并提供默认输入
        print("发生错误，尝试默认问题...")
        print(e)
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break