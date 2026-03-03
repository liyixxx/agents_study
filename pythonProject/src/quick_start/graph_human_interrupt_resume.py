"""
LangGraph Human-in-the-loop demo:
1) graph runs to an interrupt point
2) human provides review data
3) graph resumes from checkpoint
"""

from typing import Any
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    task: str
    draft: str
    approved: bool
    reviewer_comment: str
    result: str


def generate_plan(state: State) -> dict[str, Any]:
    task = state["task"]
    draft = (
        f"任务: {task}\n"
        "初稿方案:\n"
        "1. 使用 StateGraph 定义状态\n"
        "2. 添加模型节点和工具节点\n"
        "3. 用条件边控制流程\n"
        "4. 加入 checkpoint 实现可恢复执行"
    )
    return {"draft": draft}


def human_review(state: State) -> dict[str, Any]:
    review_payload = {
        "stage": "human_review",
        "instruction": "请返回 {approved: bool, comment: str}",
        "task": state["task"],
        "draft": state["draft"],
    }
    review = interrupt(review_payload)
    return {
        "approved": bool(review.get("approved", False)),
        "reviewer_comment": str(review.get("comment", "")),
    }


def route_after_review(state: State) -> str:
    return "approved" if state.get("approved") else "rejected"


def finalize(state: State) -> dict[str, Any]:
    return {
        "result": (
            "审批通过，进入执行。\n"
            f"人工意见: {state.get('reviewer_comment', '')}\n"
            f"最终方案:\n{state['draft']}"
        )
    }


def rejected(state: State) -> dict[str, Any]:
    return {
        "result": (
            "审批未通过，流程结束。\n"
            f"人工意见: {state.get('reviewer_comment', '')}"
        )
    }


def build_graph():
    builder = StateGraph(State)
    builder.add_node("generate_plan", generate_plan)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", finalize)
    builder.add_node("rejected", rejected)

    builder.add_edge(START, "generate_plan")
    builder.add_edge("generate_plan", "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {"approved": "finalize", "rejected": "rejected"},
    )
    builder.add_edge("finalize", END)
    builder.add_edge("rejected", END)

    return builder.compile(checkpointer=MemorySaver())


def run_once(thread_id: str, approved: bool, comment: str) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Session: {thread_id} ===")
    first = graph.invoke(
        {"task": "为 Java 开发者制定 LangGraph 学习路线"},
        config=config,
    )

    interrupts = first.get("__interrupt__", [])
    if interrupts:
        print("触发 interrupt，等待人工输入:")
        for item in interrupts:
            print(item.value)

    snapshot = graph.get_state(config)
    print("当前暂停节点:", snapshot.next)
    print("当前草稿摘要:", snapshot.values.get("draft", "").splitlines()[0])

    second = graph.invoke(
        Command(resume={"approved": approved, "comment": comment}),
        config=config,
    )
    print("恢复后结果:")
    print(second["result"])


if __name__ == "__main__":
    run_once("hil-demo-approved", True, "可以执行，补充测试用例。")
    run_once("hil-demo-rejected", False, "方案过于笼统，请先细化每周计划。")
