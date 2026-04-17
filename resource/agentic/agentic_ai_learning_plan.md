# Agentic-AI 实战学习计划（LangGraph路线）

## 目标说明
- 最终目标：完成一个可运行、可评估、可治理的 Agentic-AI 应用。
- 学习策略：目标驱动，按阶段交付可运行版本，不走“只学概念不落地”。

## 阶段0：场景与指标定稿（1-2天）

### 学习重点
- 明确业务问题与边界
- 定义输入、输出、工具、风险点
- 定义评估指标

### 交付物
- 一页场景说明（PRD-lite）
- 指标清单：正确率、时延、工具成功率、人工介入率

### 验收标准
- 能清楚回答：这个 Agent 替代人工哪一步、如何衡量效果

---

## 阶段1：单Agent可用版 MVP（3-5天）

### 学习重点
- Graph / Node / Edge / State
- ToolNode + 条件边 (`tools_condition`)

### 典型图结构
```text
START -> chatbot -> (tools or END)
              tools -> chatbot
```

### 交付物
- 最小可运行 Agent（至少2个工具）
- 支持结构化输出

### 验收标准
- 可稳定多轮调用工具
- 对同类问题输出一致格式结果

---

## 阶段2：记忆与持久化（3-5天）

### 学习重点
- Checkpointer 机制
- `thread_id` 与会话隔离
- `MemorySaver` 与 `PostgresSaver` 的区别

### 交付物
- InMemory 与 Postgres 两套运行模式
- 跨会话恢复演示

### 验收标准
- 同一 `thread_id` 可延续历史
- 不同 `thread_id` 状态隔离

---

## 阶段3：Agentic核心能力（4-6天）

### 学习重点
- 任务分解（Plan）
- 执行（Act）
- 反思修正（Critique/Revise）
- 迭代上限控制（防无限循环）

### 典型图结构
```text
plan -> act -> critique -> (revise or END)
```

### 交付物
- 带反思循环的工作流
- 可配置最大迭代次数

### 验收标准
- 经过反思后答案质量可提升（用自定义评分验证）

---

## 阶段4：Human-in-the-loop治理（3-5天）

### 学习重点
- `interrupt`
- `Command(resume=...)`
- `Command(update=...)`
- 人工审批节点设计

### 交付物
- 高风险步骤前人工审批
- 审批结果回写状态

### 验收标准
- 流程可暂停、可恢复、可审计

---

## 阶段5：Time Travel与调试（3-5天）

### 学习重点
- `get_state_history`
- 从 checkpoint 恢复执行
- 分叉调试与问题复现

### 交付物
- 历史状态回放脚本
- 分支结果对比案例

### 验收标准
- 至少复现并修复1个历史问题

---

## 阶段6：评估与轻量上线（1-2周）

### 学习重点
- 观测与评估（日志、trace、指标）
- 稳定性（超时、重试、限流、幂等）
- 服务化封装（CLI/API）

### 交付物
- 可对外调用的最小服务
- 回归测试样本集
- 指标看板（哪怕是简版）

### 验收标准
- 在固定测试集上稳定运行
- 指标可持续跟踪

---

## 相比旧计划的调整点

1. 从“能力清单式学习”调整为“阶段交付式学习”。
2. 将 Reflection 与评估提前，避免只停留在“工具调用”。
3. 将多智能体和复杂平台能力后置到基础稳定后。

---

## 官方文档对应学习入口

1. 基础教程：
   - https://langgraph.com.cn/tutorials/get-started/
2. 低层图与图API：
   - https://langgraph.com.cn/concepts/low_level.1.html
   - https://langgraph.com.cn/how-tos/graph-api.1.html
3. 持久化与记忆：
   - https://langgraph.com.cn/concepts/persistence.1.html
   - https://langgraph.com.cn/concepts/memory.1.html
4. 人工在环与时间旅行：
   - https://langgraph.com.cn/concepts/human_in_the_loop.1.html
   - https://langgraph.com.cn/concepts/time-travel/index.html
5. 平台与部署（后置）：
   - https://langgraph.com.cn/concepts/langgraph_platform/index.html
   - https://langgraph.com.cn/concepts/auth/index.html
   - https://langgraph.com.cn/concepts/deployment_options/index.html

---

## 执行建议

- 每阶段结束都做一次“可演示复盘”：
  - 演示脚本
  - 问题清单
  - 下一阶段改进点
- 不要跨阶段跳跃：先可用，再增强，再治理，再上线。
