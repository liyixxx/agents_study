"""
langgraph 记忆部分增强,使用 PostgreSQL 存储 checkpoint
"""

from pathlib import Path
import sys
from typing import TypedDict

from typing_extensions import Annotated

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph import add_messages


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.util.env_util import get_llm, get_postgres_connection_string

class State(TypedDict):
    messages: Annotated[list, add_messages]

@classmethod
def build_pg_graph(check_point: PostgresSaver):
    """创建持久化记忆的 graph。"""
    graph_builder = StateGraph(State)
    llm = get_llm()

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.set_entry_point("chatbot")
    graph_builder.set_finish_point("chatbot")

    return graph_builder.compile(checkpointer=check_point)


def use_pg_graph():
    """使用示例。"""
    session_config = {"configurable": {"thread_id": "user_session_001"}}
    db_uri = get_postgres_connection_string()

    print("开始会话")

    # 第一次运行：写入 checkpoint
    with PostgresSaver.from_conn_string(db_uri) as check_point:
        check_point.setup()
        graph = build_pg_graph(check_point)
        r1 = graph.invoke(
            {"messages": [{"role": "user", "content": "你好！我的名字是张三"}]},
            config=session_config,
        )
        print("Assistant:", r1["messages"][-1].content)

    # 第二次运行：使用新的连接读取相同 thread_id 的历史状态
    with PostgresSaver.from_conn_string(db_uri) as check_point:
        check_point.setup()
        graph2 = build_pg_graph(check_point)
        r2 = graph2.invoke(
            {"messages": [{"role": "user", "content": "还记得我的名字吗？"}]},
            config=session_config,
        )
        print("Assistant:", r2["messages"][-1].content)


if __name__ == "__main__":
    use_pg_graph()
