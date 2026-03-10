# LangGraph 基础学习文档（基于 quick_start 实践）

## 适用范围
- 代码材料：`/pythonProject/src/quick_start/` 下全部 demo
- 官方文档：LangGraph 基础概念与 get-started 教程（中文站）

## 1. 核心概念解析

### 1.1 Graph（图）
**定义**：Graph 是工作流运行时容器，负责组织 Node、Edge 和 State。  
在 LangGraph 中，先定义图结构，再 `compile()` 成可执行对象。

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list

builder = StateGraph(State)
# add_node / add_edge / add_conditional_edges
graph = builder.compile()
```

### 1.2 Node（节点）
**定义**：Node 是状态转换函数。输入当前状态，输出状态增量。  
官方低层概念强调：节点第一个位置参数是 `state`；可选第二个参数是 `config`。

```python
def chatbot(state: State):
    # 读取 state -> 调用模型 -> 返回更新
    return {"messages": [llm.invoke(state["messages"])]}
```

### 1.3 Edge（边）
**定义**：Edge 决定执行路径。
- 普通边：固定流向（`add_edge(a, b)`）
- 条件边：动态路由（`add_conditional_edges(...)`）
- 入口/结束：`START` / `END`（或 `set_entry_point` / `set_finish_point`）

```python
from langgraph.graph import START, END

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
# 或 builder.add_edge("chatbot", END)
```

### 1.4 State（状态）
**定义**：State 是全图共享上下文。  
`TypedDict` 定义字段；每个字段可配置 reducer（合并策略）。

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    # add_messages: 消息追加/合并，不是覆盖
    messages: Annotated[list, add_messages]
    name: str
```

### 1.5 Checkpoint / Thread（检查点与线程）
**定义**：Checkpoint 是每一步执行后的状态快照；Thread 是同一会话轨迹标识。  
持久化能力依赖 checkpointer（`MemorySaver` / `PostgresSaver`）。

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "u-001"}}
result = graph.invoke({"messages": [...]}, config=config)
```

### 1.6 Human-in-the-loop（人工在环）
**定义**：通过 `interrupt()` 暂停图执行，等待人工输入，再用 `Command(resume=...)` 恢复。

```python
from langgraph.types import interrupt, Command

# 在节点或工具内部
review = interrupt({"question": "Approve?"})

# 外部恢复
graph.invoke(Command(resume={"approved": True}), config=config)
```

### 1.7 Time Travel（时间旅行）
**定义**：从历史 checkpoint 继续执行（回放/分叉调试）。

```python
to_replay = None
for s in graph.get_state_history(config):
    if len(s.values["messages"]) == 6:
        to_replay = s

for event in graph.stream(None, to_replay.config, stream_mode="values"):
    ...
```

---

## 2. 基础工作流构建步骤

### 2.1 标准步骤
1. 定义 `State`（字段 + reducer）。
2. 编写 Node 函数（输入 state，返回增量）。
3. `StateGraph(State)` 初始化图。
4. 添加节点（`add_node`）。
5. 连接边（`add_edge` / `add_conditional_edges` / 入口结束）。
6. `compile()`（可选传 checkpointer）。
7. 用 `invoke()` 或 `stream()` 运行（可选传 `thread_id`）。

### 2.2 基础流程图（工具调用模式）

```text
START
  |
  v
chatbot --(tools_condition)--> tools
  |                            |
  |---- no tool calls ---------| 
  |                            v
  +-------------------------- chatbot
  |
  v
 END
```

### 2.3 人工在环两段式流程

```text
Phase-1: 运行到 interrupt -> 暂停并保存 checkpoint
Phase-2: Command(resume=...) -> 从暂停点继续到 END
```

---

## 3. 代码实践总结（quick_start 关键模式）

### 模式A：单节点最小图（`graph_first.py`）
- 结构：`START -> chatbot -> END`
- 作用：理解 StateGraph 最小闭环。

### 模式B：ReAct 工具循环（`graph_add_tools.py`）
- 结构：`chatbot -> tools -> chatbot`
- 关键组件：`ToolNode` + `tools_condition`
- 作用：让模型按需调用外部工具。

### 模式C：短期记忆与线程隔离（`graph_memory.py`）
- 关键：`compile(checkpointer=MemorySaver())`
- `thread_id` 相同 -> 共享会话轨迹；不同 -> 独立轨迹。

### 模式D：持久化记忆（`graph_memory_db.py`）
- 关键：`PostgresSaver.from_conn_string(...)`
- 实践点：数据库连接生命周期必须覆盖 graph 执行期。

### 模式E：人工在环（`graph_human_assistance.py` + `graph_human_interrupt_resume.py`）
- `interrupt()` 负责暂停；`Command(resume=...)` 负责恢复。
- 典型场景：审批、高风险动作确认、人工纠偏。

### 模式F：自定义状态与显式更新（`graph_custom_state.py`）
- 在工具内部返回 `Command(update=...)`，同时更新业务字段和 `messages`。
- 也可在外部 `graph.update_state(config, {...})` 做人工修正。

### 模式G：时间旅行（`graph_time_travel.py`）
- `get_state_history` 浏览历史快照。
- 选中 checkpoint 后 `stream(None, checkpoint_config)` 继续执行。
- 价值：回放调试、分叉实验、问题复现。

---

## 4. 易错点与注意事项（基于编码实践）

### 4.1 工具相关
1. 只绑定 `human_assistance`，却期望先搜索 -> 模型无法检索。  
修正：把搜索工具也加入 `tools`。
2. 以为 `bind_tools` 后一定会调用工具。  
修正：是否调用由模型决策，提示词要明确工具使用条件。

### 4.2 记忆与检查点
1. 配了 checkpointer 但没传 `thread_id` -> 无法稳定复用同一轨迹。  
2. 用 `MemorySaver` 期待跨进程持久化 -> 不成立。  
修正：跨进程用 `PostgresSaver` 等持久化后端。

### 4.3 人工在环
1. 只跑了 `interrupt` 前半段，没有 `resume`。  
修正：必须两段式运行。
2. 误以为恢复从“中断行的下一行”继续。  
修正：恢复按节点语义继续，输出可能看起来有重复（尤其 `stream_mode="values"`）。

### 4.4 状态更新
1. 忽略 reducer 规则，导致消息被覆盖或重复不符合预期。  
2. 只更新了 `name/birthday`，但节点逻辑不读取这些字段 -> 行为不变。

### 4.5 工程安全
1. API Key 明文写入代码。  
修正：改为环境变量读取。
2. `with PostgresSaver...` 内返回对象后在外部继续用，可能连接已关闭。  
修正：执行逻辑放在有效上下文内，或自行管理连接生命周期。

---

## 5. 术语表（中英文对照）

| 中文 | English | 说明 |
|---|---|---|
| 图 | Graph | 工作流执行结构 |
| 节点 | Node | 状态转换单元 |
| 边 | Edge | 节点间流转规则 |
| 条件边 | Conditional Edge | 基于函数返回进行路由 |
| 状态 | State | 全局共享上下文 |
| 状态模式 | State Schema | `TypedDict`/`BaseModel` 定义 |
| 归约器 | Reducer | 字段合并策略（如 `add_messages`） |
| 入口点 | Entry Point | 起始执行节点 |
| 结束点 | Finish Point / END | 终止执行节点 |
| 工具节点 | ToolNode | 执行 AI 工具调用 |
| 检查点 | Checkpoint | 每步保存的状态快照 |
| 线程 | Thread | 一条会话执行轨迹（`thread_id`） |
| 人工在环 | Human-in-the-loop | 人工审阅/干预执行 |
| 中断 | Interrupt | 暂停图运行等待外部输入 |
| 恢复命令 | Command(resume) | 从中断点恢复执行 |
| 状态更新命令 | Command(update) | 在运行中显式写入状态 |
| 时间旅行 | Time Travel | 从历史 checkpoint 恢复/分叉 |
| 持久化执行 | Durable Execution | 可恢复、可回放的长流程执行 |

---

## 进阶主题推荐（下一阶段 5 选）

以下主题均来自官方文档导航，适合你完成基础 6 章后的下一阶段。

1. **低层图 API 深入（Graph API + Low-level）**  
学习价值：掌握 `Send`、并行分发、复杂条件路由和可控状态更新，能独立设计中等复杂 Agent 图。  
参考：
- https://langgraph.com.cn/how-tos/graph-api.1.html
- https://langgraph.com.cn/concepts/low_level.1.html

2. **持久化与记忆体系（Persistence + Memory）**  
学习价值：从 demo 级记忆升级到生产级线程存储、短期/长期记忆治理（成本和效果平衡）。  
参考：
- https://langgraph.com.cn/concepts/persistence.1.html
- https://langgraph.com.cn/concepts/memory.1.html

3. **子图与多智能体编排（Subgraphs + Multi-agent）**  
学习价值：把复杂业务拆成可复用子流程，提升维护性，支持多角色协作。  
参考：
- https://langgraph.com.cn/concepts/subgraphs.1.html
- https://langgraph.com.cn/tutorials/multi_agent/multi-agent-collaboration/index.html

4. **函数式 API（Functional API）**  
学习价值：在保留 LangGraph 持久化/中断能力的同时，用更轻量方式嵌入现有工程。  
参考：
- https://langgraph.com.cn/concepts/functional_api.1.html

5. **平台化与部署（Platform / Auth / Deployment）**  
学习价值：把本地实验迁移到可观测、可鉴权、可运维的服务化环境。  
参考：
- https://langgraph.com.cn/concepts/langgraph_platform/index.html
- https://langgraph.com.cn/concepts/auth/index.html
- https://langgraph.com.cn/concepts/deployment_options/index.html

---

## 官方对齐参考链接
- 首页：https://langgraph.com.cn/index.html
- 基础教程：https://langgraph.com.cn/tutorials/get-started/
- 低层概念：https://langgraph.com.cn/concepts/low_level.1.html
- 人工在环：https://langgraph.com.cn/concepts/human_in_the_loop.1.html
- 持久化：https://langgraph.com.cn/concepts/persistence.1.html
- 记忆：https://langgraph.com.cn/concepts/memory.1.html
- 时间旅行：https://langgraph.com.cn/concepts/time-travel/index.html
