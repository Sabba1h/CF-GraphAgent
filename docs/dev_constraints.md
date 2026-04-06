# 开发约束说明（dev_constraints）

## 1. 文档目的

本文档用于约束代码实现过程，确保生成的代码符合当前研究原型的需求，不因过早优化、随意扩展或错误抽象而偏离项目目标。

本文档主要面向代码生成代理、自动补全代理或协作开发者。

---

## 2. 总体开发原则

### 2.1 先搭骨架，后补细节

当前阶段优先构建稳定、清晰、最小可运行的项目结构。  
不要一开始就试图实现完整论文系统。

### 2.2 先保证可运行，再保证可扩展

所有代码首先要保证最小 demo 能跑通，其次才考虑二阶段和三阶段扩展。

### 2.3 优先模块边界清晰

不要把图操作、环境逻辑、奖励逻辑、候选生成、观测渲染、回答生成全部写进一个类。

### 2.4 优先可解释与可调试

代码应便于打印中间状态、候选动作、轨迹信息，而不是过度追求“简洁隐藏”。

---

## 3. 目录与模块约束

必须优先采用模块化目录结构。  
至少应包含以下主模块：

- `envs/`
- `graph/`
- `memory/`
- `candidates/`
- `observation/`
- `reward/`
- `answer/`
- `scripts/`

允许增加：

- `tests/`
- `utils/`
- `data/`
- `docs/`

不得把全部逻辑只写在一个脚本文件中。

---

## 4. 类型与数据结构约束

### 4.1 必须使用类型标注

所有公开函数、核心类方法、主要 dataclass 字段都必须有类型标注。

### 4.2 优先使用 dataclass

以下对象优先定义为 dataclass：

- `EdgeRecord`
- `CandidateAction`
- `StepRecord`
- `EpisodeSummary`
- 其他稳定结构对象

### 4.3 避免弱结构化字典泛滥

除非是 observation 或 info 返回值，否则不要到处传递未定义 schema 的松散字典。  
内部核心数据尽量使用 dataclass 或清晰的类对象。

---

## 5. 环境设计约束

### 5.1 环境类职责必须克制

`CFGraphEnv` 负责协调流程，但不应承担所有业务逻辑。  
它应该调用下列模块：

- `GraphStore`
- `WorkingMemory`
- `CandidateGenerator`
- `ObservationRenderer`
- `RewardEngine`
- `AnswerEngine`
- `TrajectoryLogger`

### 5.2 环境状态不能等同于文本 prompt

禁止把完整环境状态只保存在文本字符串中。  
环境内部必须有结构化状态对象。

### 5.3 `step()` 不接受自由生成的开放动作

禁止把主动作接口设计成：

- 任意自然语言
- 任意 JSON 三元组
- 任意实体名字符串拼接

第一阶段只允许：

- `int`
- `{"candidate_id": int}`

---

## 6. 图层实现约束

### 6.1 使用 `NetworkX.MultiDiGraph`

第一阶段默认图对象必须支持：

- 有向边
- 多关系边
- 边属性

### 6.2 必须有稳定 `edge_id`

所有边都必须具有唯一 `edge_id`。  
后续的日志、回放、反事实 reward 都依赖此约束。

### 6.3 不得把 relation 信息丢失

同一 `src-dst` 对之间的不同 relation 必须能够同时存在。  
不能因使用错误图结构而覆盖掉边。

---

## 7. 候选动作生成约束

### 7.1 必须先生成候选动作，再由 Agent 选择

禁止把“候选动作生成”和“动作选择”混在一起。  
`CandidateGenerator` 的职责是输出合法动作列表，而不是直接做决策。

### 7.2 每一步必须提供编号候选集

所有候选动作必须有唯一 `candidate_id`，且该编号在当前 step 内有效。

### 7.3 每一步必须始终包含终止动作

无论有无候选边，每一步都应包含：

- `ANSWER`
- `STOP`

除非明确处于异常终止逻辑。

---

## 8. Observation 设计约束

### 8.1 必须同时支持 structured observation 与 text observation

不能只实现 prompt 文本，不实现结构化 observation。

### 8.2 文本 observation 应可读但不能无限增长

不得无限拼接所有历史。  
必须做历史摘要或窗口裁剪。

### 8.3 ObservationRenderer 独立存在

禁止把文本 observation 的拼接逻辑散落在 `env.step()` 各处。  
必须集中在 `ObservationRenderer` 中实现。

---

## 9. Reward 设计约束

### 9.1 第一阶段 reward 只做最小闭环

第一阶段 reward 仅实现：

- 合法扩展
- 重复扩展
- 非法动作
- `ANSWER`
- `STOP`

### 9.2 必须为反事实 reward 预留接口

即使第一阶段不实现，也必须在 `RewardEngine` 中留有明确扩展位。  
不得把 reward 写死成不可扩展的散落 if-else。

### 9.3 禁止提前接入复杂 RL 逻辑

第一阶段不得实现：

- PPO
- GRPO
- value network
- advantage estimation
- policy gradient 训练循环

---

## 10. 回答模块约束

### 10.1 第一阶段允许 mock，但接口要稳定

`AnswerEngine` 可以先用简单规则或 mock 逻辑实现。  
但对外接口应保持稳定，后续可替换为真实 LLM 或更复杂 evaluator。

### 10.2 回答逻辑不能硬编码进环境类

禁止在 `CFGraphEnv.step()` 内直接写大量回答逻辑。  
必须通过 `AnswerEngine` 间接调用。

---

## 11. 日志与调试约束

### 11.1 必须实现轨迹日志

环境必须能够记录每一步：

- 候选动作
- 选中的动作
- reward
- 工作子图摘要
- 终止状态

### 11.2 必须保留 step 级调试能力

不允许所有状态变化都只存在于内部对象而不可查看。  
至少应能通过 debug 脚本打印关键中间结果。

### 11.3 错误信息要明确

非法动作、重复动作、候选为空等情况，必须在 `info` 或日志中给出清晰原因，方便后续调试。

---

## 12. 测试与运行约束

### 12.1 每轮开发后应至少完成基本运行检查

至少应验证：

- 模块可导入
- toy graph 可构建
- 环境可 reset
- 环境可 step
- episode 可终止

### 12.2 优先提供最小 demo

项目中必须有一个可直接运行的脚本，例如：

- `scripts/run_env_demo.py`
- `scripts/debug_single_episode.py`

### 12.3 鼓励增加轻量测试

允许添加 `tests/` 目录，对以下模块优先覆盖：

- `GraphStore`
- `CandidateGenerator`
- `CFGraphEnv`

但第一阶段不强制高覆盖率测试框架。

---

## 13. 代码风格约束

### 13.1 函数尽量短小、职责单一

一个函数只做一件事。  
避免写一个 200 行的大函数同时完成候选生成、状态转移、日志和 reward。

### 13.2 docstring 必须清楚

核心类与关键公开方法必须写简洁 docstring，说明：

- 输入
- 输出
- 作用
- 关键副作用

### 13.3 命名必须语义明确

避免使用模糊命名，如：

- `data1`
- `tmp_edges`
- `handler2`

应优先使用：

- `candidate_actions`
- `working_edges`
- `frontier_nodes`
- `edge_record`

### 13.4 不要过度抽象

第一阶段不需要设计复杂插件系统、注册表工厂体系、过深继承层次。  
优先使用直接、清晰、易读的实现。

---

## 14. 本阶段明确禁止事项

以下行为在第一阶段中明确禁止：

1. 把全部逻辑塞进一个 notebook 或单个脚本
2. 把动作主接口写成自由文本或自由 JSON
3. 把环境状态只保存在 prompt 字符串中
4. 在第一阶段强行接入 PPO / GRPO
5. 直接接入 FAISS / Milvus / BM25 等外部复杂检索系统
6. 过早实现多智能体
7. 过早实现分布式 rollout
8. 为了“通用性”引入明显超出当前需求的复杂抽象
9. 为了“看起来完整”一次性实现二阶段和三阶段功能
10. 隐藏关键中间状态，导致环境难以调试

---

## 15. 推荐开发顺序

建议按以下顺序实现：

1. 项目骨架与目录结构
2. 类型对象与 dataclass
3. `GraphStore`
4. `WorkingMemory`
5. `CandidateGenerator`
6. `ObservationRenderer`
7. `RewardEngine`
8. `AnswerEngine`
9. `TrajectoryLogger`
10. `CFGraphEnv`
11. demo 脚本
12. 轻量测试

禁止跳过前面基础层，直接硬写环境主类。

---

## 16. 成功交付的判断标准

只有同时满足以下标准，才说明本轮开发是合格的：

- 项目结构清晰
- 核心模块独立
- toy graph 可运行
- 候选动作机制正确
- action 通过 `candidate_id` 选择
- `EXPAND_EDGE / ANSWER / STOP` 行为正确
- 环境日志完整
- 代码易读、可继续迭代

---