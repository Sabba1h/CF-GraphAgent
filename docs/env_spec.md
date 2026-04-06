# 面向 Agentic Graph RAG 的强化学习环境规格说明（env_spec）

## 1. 文档目的

本文档用于定义一个面向 Agentic Graph RAG 的强化学习环境原型，其主要用途不是直接写入论文，而是为代码实现、模块拆分、环境搭建、后续 reward 扩展和实验复现实验提供统一规格。

本文档服务于如下研究目标：

> 让大语言模型智能体（LLM Agent）能够在知识图谱上进行多步检索、扩展与可选更新，并通过强化学习学习更优的图决策策略。后续将引入基于反事实比较的 step-wise dense causal reward，以解决长过程图推理中的奖励稀疏问题。

当前阶段的重点不是实现完整算法，而是先构建一个**结构清晰、接口标准、易于扩展**的 Gymnasium 风格环境。

---

## 2. 设计总原则

本项目环境设计遵循以下核心原则：

### 2.1 结构状态优先于文本状态

环境内部的真实状态必须是结构化图状态，而不是简单的 prompt 文本。  
文本 observation 只是结构状态的视图，不应被当作环境内部唯一状态表示。

### 2.2 动作必须受控

动作不能设计为完全开放的自然语言或自由 JSON 生成。  
环境必须先生成带编号的候选动作集合，Agent 再从中选择。

### 2.3 检索与图编辑在语义上区分

环境设计上必须为两类行为预留独立语义：

- 检索行为：从现有图谱中选择边、扩展工作子图
- 编辑行为：向图谱中增加、删除、修改边

第一阶段可以只实现检索，但接口层必须支持未来加入图编辑。

### 2.4 环境先于算法

在第一阶段，不实现 PPO、GRPO、反事实 reward 或复杂训练循环。  
优先实现最小闭环环境，使得状态、动作、状态转移、奖励接口、终止逻辑和日志系统能够跑通。

### 2.5 可解释与可回放

环境应能记录完整轨迹，使得后续任一步动作都能够被重新定位、重放、删除或替换，用于反事实 reward 计算。

---

## 3. 任务定义

每个 episode 对应一个图推理任务，输入包括：

- 一个用户查询 `query`
- 一个底层知识图谱 `base_graph`
- 可选标准答案 `ground_truth`
- 可选初始种子实体 `seed_entities`

智能体需要在有限步数内对图进行多步操作，构建当前 query 的工作子图，并在合适时机输出答案。

---

## 4. 环境形式化定义

环境被建模为一个序列决策过程：

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \mathcal{T})
\]

其中：

- \(\mathcal{S}\)：状态空间
- \(\mathcal{A}\)：动作空间
- \(P\)：状态转移函数
- \(R\)：奖励函数
- \(\mathcal{T}\)：终止条件

尽管环境形式上是标准 MDP，但动作粒度应为“**高层图操作**”，而不是 token 粒度。

---

## 5. 状态空间定义

在时间步 \(t\)，环境内部状态定义为：

\[
s_t = (q, G^{base}, G_t^{work}, F_t, M_t, B_t, \mathcal{C}_t)
\]

各字段含义如下：

### 5.1 `q`
当前用户查询，在整个 episode 中保持不变。

### 5.2 `G_base`
底层知识图谱，是智能体可检索的图对象。  
第一阶段默认只读；后续阶段可引入可写副本。

### 5.3 `G_work_t`
当前工作子图。  
表示在当前 episode 中已经被接纳并纳入推理上下文的边与节点集合。

### 5.4 `F_t`
当前 frontier，表示下一步可扩展的前沿节点或边集合。

### 5.5 `M_t`
结构化过程记忆，包括但不限于：

- 已访问节点集合
- 已访问边集合
- 已接纳边集合
- 已观察但未接纳边集合
- 最近若干步动作历史
- 节点访问次数
- 边访问次数
- 当前累计成本
- 当前已用步数

### 5.6 `B_t`
当前剩余预算。  
第一阶段至少包含剩余步数 `steps_left`。

### 5.7 `C_t`
当前候选动作集合。  
必须由环境显式生成与维护，而不是由模型自由生成。

---

## 6. 图数据结构规范

### 6.1 图类型

第一阶段建议使用 `NetworkX.MultiDiGraph` 作为底层图结构。

原因：

- 知识图谱天然是有向图
- 同一对节点之间可能存在多种 relation
- 需要保留边属性

### 6.2 边对象规范

每条边至少应包含以下字段：

- `edge_id: str`
- `src: str`
- `dst: str`
- `relation: str`
- `confidence: float`
- `source: str`
- `timestamp: Optional[str]`

其中 `edge_id` 必须全局唯一，并作为边级 credit assignment 的基本索引。

### 6.3 工作子图管理要求

工作子图 `G_work_t` 不应简单等于所有访问过的边集合。  
必须区分：

- `accepted edges`：被纳入工作子图的边
- `observed but rejected edges`：已看见但未纳入工作子图的边

---

## 7. 观测（Observation）设计

环境对外暴露的 observation 应分为两层：

### 7.1 结构化 observation

用于训练代码、调试脚本、非 LLM 模型或日志系统，格式可为 Python 字典。建议包含：

- `query`
- `current_working_nodes`
- `current_working_edges`
- `frontier_nodes`
- `candidate_actions`
- `steps_left`
- `history_summary`

### 7.2 文本 observation

用于直接喂给 LLM。由独立模块从结构状态渲染生成。  
文本 observation 应包括：

- 当前 query
- 当前工作子图摘要
- 最近若干步关键动作摘要
- 当前候选动作及其编号
- 剩余预算信息

文本 observation 不应无限拼接完整历史，应只保留压缩后的可读摘要。

---

## 8. 动作空间定义

### 8.1 基本思想

动作必须被建模为对候选动作集合的选择，而不是自由生成的自然语言或 JSON。

### 8.2 统一动作表示

推荐动作格式为：

\[
a_t = (\texttt{action\_type}, \texttt{candidate\_id})
\]

第一阶段中，外部策略可直接只输出 `candidate_id`。

### 8.3 第一阶段必须支持的动作类型

- `EXPAND_EDGE`：接纳某条候选边并扩展工作子图
- `ANSWER`：基于当前工作子图回答问题并结束 episode
- `STOP`：终止探索并结束 episode

### 8.4 后续阶段预留动作类型

- `ADD_EDGE`
- `DELETE_EDGE`
- `UPDATE_EDGE_WEIGHT`
- `CHANGE_FRONTIER`

第一阶段不要求实现这些动作，但类型体系需要预留扩展空间。

---

## 9. 候选动作生成机制

### 9.1 总体目标

由于底层图规模可能很大，环境必须在每个 step 对动作空间进行裁剪，将开放世界搜索转化为有限候选选择问题。

### 9.2 双向约束候选器

候选动作集合由两部分约束共同决定：

#### 全局语义相关性约束

根据 query，从全图召回与问题相关的节点或边。  
第一阶段可采用简单规则或轻量检索方法；后续阶段可接入 BM25、FAISS、Milvus 等系统。

记为：

\[
\mathcal{C}^{global}(q)
\]

#### 局部结构可达性约束

根据当前工作子图和 frontier，只保留局部可扩展边。

记为：

\[
\mathcal{C}^{local}(G_t^{work}, F_t)
\]

#### 最终候选集

\[
\mathcal{C}_t = \text{Fuse}\left(\mathcal{C}^{global}(q), \mathcal{C}^{local}(G_t^{work}, F_t)\right)
\]

其中 `Fuse` 可以是交集、排序融合或先过滤再重排。

### 9.3 第一阶段可接受的简化

第一阶段不强制接入向量库。  
允许使用：

- query 与实体名/关系名的简单字符串匹配
- query 召回种子节点
- frontier 一跳邻边扩展
- 基于简单打分规则排序 top-k

### 9.4 候选动作对象规范

候选动作应为结构化对象，建议至少包括：

- `candidate_id`
- `action_type`
- `edge_id`（若适用）
- `src`
- `dst`
- `relation`
- `score_global`
- `score_local`
- `score_final`

此外，每个 step 必须始终包含终止类动作：

- `ANSWER`
- `STOP`

---

## 10. 状态转移机制

### 10.1 `EXPAND_EDGE`

执行流程：

1. 校验当前 `candidate_id` 是否合法
2. 取出对应候选边
3. 将边加入工作子图
4. 更新 frontier
5. 更新结构化记忆
6. 重新生成下一步候选动作集合

### 10.2 `ANSWER`

执行流程：

1. 基于当前工作子图构造回答上下文
2. 调用回答模块生成答案
3. 对答案打分
4. 终止 episode

### 10.3 `STOP`

执行流程：

1. 终止当前 episode
2. 可选输出空答案或固定“无法确定”

第一阶段建议语义尽量简单稳定，不要混入复杂停止策略。

---

## 11. 非法动作处理

非法动作处理必须写入环境定义，不可留作后续再补。

第一阶段推荐规则：

- 若 `candidate_id` 越界或不存在，则视为非法动作
- 不改变图状态
- 给固定负奖励
- 在 `info` 中记录 `invalid_action=True`
- 若连续非法动作超过阈值，可提前终止 episode

对于重复选择已接纳边：

- 不重复加入工作子图
- 视为冗余动作
- 给较小负奖励

---

## 12. 奖励接口设计

### 12.1 第一阶段奖励结构

建议统一写成：

\[
r_t = r_t^{task} + r_t^{process} + r_t^{constraint}
\]

其中：

- `r_task`：任务奖励，主要在 `ANSWER` 或 `STOP` 时给出
- `r_process`：过程奖励，如合理扩展、无效扩展等
- `r_constraint`：约束惩罚，如非法动作、重复扩展等

### 12.2 第一阶段最小可行奖励

第一阶段允许采用非常简化的 reward，例如：

- 合法扩展：`+0.1`
- 重复扩展：`-0.2`
- 非法动作：`-1.0`
- `ANSWER`：调用占位 evaluator 得到最终奖励
- `STOP`：返回固定较低终止奖励

### 12.3 反事实奖励接口预留

环境中必须从第一阶段起预留如下接口：

```python
def compute_stepwise_counterfactual_reward(
    state_before,
    action_taken,
    state_after,
    answer_evaluator,
    mode="remove_action"
) -> float:
    ...