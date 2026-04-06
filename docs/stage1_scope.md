---

# `docs/stage1_scope.md`

```md
# 第一阶段实现范围说明（stage1_scope）

## 1. 文档目的

本文档用于明确第一阶段代码实现的任务边界，避免在环境刚搭建时过早引入复杂功能，导致结构混乱、调试困难和迭代成本升高。

第一阶段的目标不是实现完整的 Agentic Graph RAG 系统，也不是实现最终论文方法，而是：

> 搭建一个最小可运行的 Gymnasium 风格图推理环境原型，使其能够在 toy graph 上完成 `reset -> 多步选择候选边 -> 终止回答/停止` 的完整闭环。

---

## 2. 第一阶段总目标

第一阶段必须完成以下能力：

1. 能表示一个小规模有向多关系图
2. 能围绕一个 query 初始化环境
3. 能生成候选动作集合
4. 能接受 `candidate_id` 形式的动作输入
5. 能执行 `EXPAND_EDGE`、`ANSWER`、`STOP`
6. 能维护工作子图与访问历史
7. 能返回 observation、reward、terminated、truncated、info
8. 能记录完整 episode 轨迹
9. 能提供一个最小 demo 脚本跑通一局

---

## 3. 第一阶段必须实现的模块

第一阶段必须至少包含如下模块：

- `envs/cf_graph_env.py`
- `graph/graph_store.py`
- `memory/working_memory.py`
- `candidates/generator.py`
- `observation/renderer.py`
- `reward/reward_engine.py`
- `answer/answer_engine.py`
- `memory/trajectory_logger.py`

可以根据需要增加：

- `types.py`
- `schemas.py`
- `utils.py`
- `toy_data.py`

但不得破坏上述主模块职责。

---

## 4. 第一阶段图层范围

### 4.1 必须做的内容

- 使用 `NetworkX.MultiDiGraph`
- 定义边记录对象
- 支持：
  - 添加节点
  - 添加边
  - 通过 `edge_id` 查询边
  - 查询邻接边
  - 判断边是否存在
  - 导出小子图或子图摘要

### 4.2 推荐实现方式

建议定义：

- `EdgeRecord`
- `GraphStore`

其中 `GraphStore` 作为底层图操作统一入口。

### 4.3 第一阶段不要求的内容

- 不要求支持复杂图持久化
- 不要求支持数据库后端
- 不要求接入真实大规模知识图谱
- 不要求支持高性能索引优化

---

## 5. 第一阶段状态与记忆范围

### 5.1 必须做的内容

`WorkingMemory` 至少维护：

- 当前工作子图中的边集合
- 已访问节点集合
- 已访问边集合
- 当前 frontier 节点集合
- 最近动作历史
- 当前已用步数
- 最大步数
- 当前候选动作缓存

### 5.2 第一阶段允许的简化

- 历史摘要可以只保留最近若干步
- frontier 可以简单定义为最新扩展边的终点节点
- 不要求复杂图回溯逻辑
- 不要求路径级缓存优化

---

## 6. 第一阶段候选动作范围

### 6.1 必须做的内容

每个 step 必须生成一个候选动作列表，至少包括：

- 若干 `EXPAND_EDGE`
- 一个 `ANSWER`
- 一个 `STOP`

每个候选动作必须具备唯一的 `candidate_id`。

### 6.2 第一阶段允许的候选生成方式

第一阶段不接入 BM25、FAISS、Milvus。  
候选器可使用以下简化逻辑：

1. 根据 query 对图中节点名或关系名做简单字符串匹配，召回 seed entities
2. 从 seed entities 或 frontier 进行一跳邻边扩展
3. 去除重复边
4. 做简单排序
5. 取 top-k 作为 `EXPAND_EDGE`
6. 再附加 `ANSWER` 和 `STOP`

### 6.3 第一阶段不要求的内容

- 不要求真实语义检索系统
- 不要求向量数据库
- 不要求多路召回融合
- 不要求复杂 rerank
- 不要求 relation schema 推断

---

## 7. 第一阶段 observation 范围

### 7.1 必须做的内容

必须同时支持：

- 结构化 observation
- 文本 observation

### 7.2 结构化 observation 最低要求

结构化 observation 至少包含：

- `query`
- `working_edges`
- `frontier_nodes`
- `candidate_actions`
- `steps_left`
- `history_summary`

### 7.3 文本 observation 最低要求

文本 observation 至少包含：

- 当前 query
- 当前工作子图摘要
- 最近若干步动作摘要
- 当前候选动作编号与简要说明
- 当前剩余步数

### 7.4 第一阶段不要求的内容

- 不要求专门适配某个具体大模型模板
- 不要求复杂 prompt engineering
- 不要求多视图 observation 融合
- 不要求 token 长度控制优化到极致

---

## 8. 第一阶段动作范围

### 8.1 必须实现的动作类型

只实现以下三种：

- `EXPAND_EDGE`
- `ANSWER`
- `STOP`

### 8.2 动作输入格式要求

环境 `step()` 只接受以下两种形式之一：

- `int`（直接表示 `candidate_id`）
- `{"candidate_id": int}`

不接受自由生成的 JSON 三元组动作。

### 8.3 第一阶段不允许实现的主接口

以下动作类型可以定义占位，但不得作为第一阶段主流程强制实现：

- `ADD_EDGE`
- `DELETE_EDGE`
- `UPDATE_EDGE_WEIGHT`
- `CHANGE_FRONTIER`

---

## 9. 第一阶段状态转移范围

### 9.1 `EXPAND_EDGE`

必须实现：

- 校验候选合法性
- 将边加入工作子图
- 更新 frontier
- 更新记忆
- 重新生成候选动作

### 9.2 `ANSWER`

必须实现：

- 基于当前工作子图生成答案
- 给出最终奖励
- 结束 episode

### 9.3 `STOP`

必须实现：

- 直接结束 episode
- 返回固定终止结果

---

## 10. 第一阶段回答模块范围

### 10.1 必须做的内容

提供一个最小回答模块 `AnswerEngine`，允许先用占位逻辑实现，例如：

- 根据当前工作子图拼接简短答案
- 或返回固定 mock answer
- 或用简单规则判断是否答对

### 10.2 第一阶段允许的简化

- ground truth evaluator 可非常简单
- 不要求真实 LLM 调用
- 不要求复杂答案生成模板
- 不要求多答案比较

---

## 11. 第一阶段奖励范围

### 11.1 必须做的内容

实现一个最小 `RewardEngine`，至少支持：

- 合法扩展奖励
- 非法动作惩罚
- 重复扩展惩罚
- `ANSWER` 终局奖励
- `STOP` 终止奖励

### 11.2 推荐默认值

下面只是默认建议，可后续再调：

- 合法扩展：`+0.1`
- 重复扩展：`-0.2`
- 非法动作：`-1.0`
- `STOP`：`-0.5`
- `ANSWER`：由 evaluator 返回 `0/1` 或更细分数值

### 11.3 第一阶段明确不做的内容

- 不做 step-wise counterfactual reward
- 不做因果 credit assignment
- 不做 advantage 估计
- 不做 value function
- 不做 reward model

---

## 12. 第一阶段终止条件

必须支持：

- 选择 `ANSWER`
- 选择 `STOP`
- 达到最大步数
- 候选动作不可扩展
- 连续非法动作超过阈值（可选）

---

## 13. 第一阶段日志与调试要求

### 13.1 必须记录的内容

`TrajectoryLogger` 至少记录：

- query
- 每一步候选动作列表
- 每一步选择的动作
- 每一步 reward
- 每一步工作子图摘要
- 最终终止原因
- 最终答案
- 最终得分

### 13.2 必须提供的调试能力

必须至少有一个 debug 脚本可以：

- 构造 toy graph
- reset 环境
- 手工或规则选择若干步
- 打印每一步 observation、action、reward、terminated 状态

---

## 14. 第一阶段必须交付的运行结果

完成第一阶段后，项目应至少能做到：

1. 成功导入所有核心模块
2. 成功构造 toy graph
3. 成功初始化环境
4. 成功执行多步 `step()`
5. 成功终止 episode
6. 成功输出完整轨迹日志

---

## 15. 第一阶段明确不做的内容总表

以下内容全部不属于第一阶段范围：

- PPO / GRPO / DQN / A2C 等强化学习训练器
- 反事实 reward
- 因果 credit assignment
- 真正图编辑（增边/删边/改边权）主流程
- 向量数据库接入
- BM25 检索系统接入
- 多智能体
- 分布式采样
- 并行 rollout
- 真正线上知识图谱
- 复杂评测 benchmark
- UI 界面
- 可视化面板

---

## 16. 第一阶段验收标准

只有在以下条件都满足时，才算第一阶段完成：

### 16.1 代码结构验收

- 目录结构清晰
- 模块职责明确
- 类型定义基本完整
- 关键类有 docstring

### 16.2 功能验收

- toy graph 环境可运行
- `reset()` 和 `step()` 行为正确
- 候选动作编号机制生效
- 非法动作处理正确
- 工作子图会随 `EXPAND_EDGE` 更新
- `ANSWER` 和 `STOP` 可正确终止

### 16.3 可调试性验收

- 能打印轨迹
- 能定位某一步选了哪条边
- 能看到当前工作子图
- 能看出终止原因和最终奖励

---