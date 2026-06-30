# 自动化多智能体故障归因 (Who & When) 核心诊断框架汇编

本文档汇编了针对大语言模型多智能体系统（LLM Multi-Agent Systems）自动化故障归因的三种核心诊断框架。重点阐述了我们全新提出的 **AgentJury (或 CRAFT)** 架构，以及与之对比的两大基线大模型归因方案 (CHIEF 与 ECHO)。

## 0.项目架构以及关键文件和关键命令

一、 关键文件与目录层级

1. 核心系统代码 (Automated_FA/)
   inference.py (系统总控入口)
   作用：接收命令行参数，加载底层多模态/大语言模型（如兼容 SiliconFlow API, Llama, Qwen）。作为主路由，调度不同数据集的流转，并分发到对应的归因策略模块运算。
   Lib/api_utils.py
   作用：项目的承重墙。封装了所有对外通信逻辑（\_make_api_call_with_retry 网络容错连线）、各类统一的文件 I/O 加载器。
   关键点：不仅是工具库，还包含了处理 Algorithmic (相对规整的算法生成) 数据集的标准 DAO 审查主循环逻辑（在此数据集上达成了超 90% 的探员定位率）。
   Lib/api_utils_4_handcrafted.py (Hand-Crafted 恶劣数据定制适配器)
   作用：专门为高度发散、超长噪音（HTML、OCR碎片）的人工手工测试集定制的“加强版审查流”。
   关键点：含有 [TRUNCATION] 防崩溃截断规则、重写的 ECHO 摘要提词、强制红队对抗提示词（判定 Strict Liability 严格责任制）以及定制化三权分立 DAO 提示词。
   extract_hc_summaries.py (独立摘要提炼工具)
   作用：为了排查“Token 上下文淹没”问题单独剥离的工具脚本，负责将 ECHO 第一阶段对冗长步骤压缩的结构化三键 JSON 摘要（指令、动作、反馈）独立提取出来进行质检。
   api_utils_melting.py
   预期的消融实验文件夹

2. 数据与实验仓库
   outputs/
   完整的Algorithm-Generated实验结果和部分hc数据集实验结果
   Who&When/ (评测目标数据集)
   Algorithm-Generated 相关子文件夹：标准基准对齐测试集。
   Hand-Crafted / mvp*test_hc*\* 相关子文件夹：夹杂着噪音和复杂人类意图的“真实世界”魔鬼测试集。
   extract_of_error/ (知识规则库驱动)
   包含通过 extract_failed_reasons.py 对海量失败样本执行 K-means 聚类产生的洞察结论，指导了 DAO 三名评审专家的思维模型。

---

## 1. 提议方案：AgentJury

**AgentJury** 是我们针对多智能体系统故障归因中，大模型普遍存在的**“上帝视角溯源综合症 (Upstream Blame Bias)”**而设计的突破性理论与工程框架。

该方案摒弃了传统的全局因果图检索，通过 **基于混合分辨率的滑动探针 + 数据驱动的涌现式 DAO 仲裁庭 + 双层递归定位与 Top-K 容错解码** 的组合拳，在 `Who&When` 复杂基准测试集上实现了极高的零样本诊断准确率（Top-2 召回率突破 90%）。

### 系统级联架构与算法实现

AgentJury 框架将时序扫描、混合分辨率上下文构造与多点共识融合为一套紧密耦合的诊断流水线：

```text
Phase 1 — Probe Gatekeeper（前置滑动窗口探针）
  ↓ 将全量日志进行k分片，每个分片中包含 日志的高分辨率+远端近点结构化摘要
  ↓ k次探针层llm调用，判断分片中的Agent是否有可能是错误的步骤
  ↓ 用置信度规则标记当前步骤
  ↓ 以可疑步骤为中心，展开+-上下文窗口，在上下文窗口外的步骤压缩为结构化摘要发送给DAO

Phase 2 — Data-Driven DAO Arbitrators (先验增强的多实体独立法庭)
  ├── Arbitrator_1 (事实与检索错误视角)
  ├── Arbitrator_2 (逻辑与规划错误视角)
  └── Arbitrator_3 (执行与协作错误视角)

Phase 3 — Dense Soft Voting（全概率分布软投票中心+真实场景容错鲁棒性top-2指标）
  ↓ 获取每位法官对所有特工的细粒度量刑打分 (agent_evaluations)
  ↓ 加总概率密度，对冲单体大模型的上游首因偏见
  → 输出最终判决（Top-1 Final Culprit & Mistake Step）+（Top-2 Final Culprit & Mistake Step）
```

### 核心法理与机制剖析

1. 基于混合分辨率的滑动探针 (Sliding Spotlight Probing with Mixed Resolution, SSP-MR)物理机制：打破了现有方法（如 ECHO）“以目标为中心、向外围物理衰减”的静态视野限制。通过构建 [中心高分辨率全量] + [外围低分辨率摘要] 的滑动窗口，赋予系统动态感受野。学术价值（核心防御点）：本质上实现了一种外部硬注意力机制（External Hard-Attention）。它在长程交互轨迹中强制“收紧”了大模型（LLM）的注意力权重，大幅提高了决策窗口的信噪比（SNR），有效克服了长文本导致的“中间迷失（Lost in the Middle）”效应。
2. 数据驱动的涌现式 DAO 仲裁庭 (Data-Driven Emergent DAO Arbitration)物理机制：摒弃了严重依赖人工启发式设定的固定专家库（如 ECHO 中的保守型、怀疑论者等通用性格）。通过对历史失败日志执行非参数拓扑发现（如 BERTopic），系统动态实例化特化专家。学术价值：实现了诊断系统从“人工先验（Human-crafted Prior）”向“数据驱动自适应（Data-driven Adaptive）”的范式转移，使得专家维度能够随目标多智能体系统的底层错误分布自动演进。
3. 双层递归定位与 Top-K 容错解码 (Two-tier Recursive Localization & Top-K Decoding)物理机制：采用“Phase 2 探针粗筛定位 $\rightarrow$ Phase 3 焦点定向仲裁”的递归管线，并在输出端引入置信度衰减软投票。学术价值：将传统的“一次性全局硬判决”转化为契合工业界真实排错流的“高置信度嫌疑区间推荐”。Top-2 机制不仅保护了少数派的高优信号，更为开发者提供了极具操作性的灰度调试方案。

---

## 2. 基线方案一：CHIEF (分层因果图与反事实回溯)

CHIEF (Causal Hierarchical Failure Attribution) 框架旨在解决传统方法将执行日志视为线性序列而导致因果关系错乱的问题。

- **分层因果图构建 (HCG)**：将平铺长文本转换为图拓扑，基于控制流与数据流连接节点。
- **分层预言机引导回溯 (Oracle-Guided Backtracking)**：合成“虚拟预言机”，自顶向下比对预期与实际结果。
- **局限性**：在大规模、高自由度长文本系统中，构建全局因果图的 Token 消耗和算力成本呈现指数级上升；同时，虚拟预言机的幻觉极易导致整棵搜索树的方向偏离。

---

## 3. 基线方案二：ECHO (分层上下文与同构共识投票)

ECHO 框架的核心目的是克服大模型在处理极长、极其复杂的智能体交互序列时的“中间遗忘（Lost-in-the-middle）”现象。

- **静态分层上下文提取**：基于距离目标步骤的位置远近，将整个日志划分为四个层级（直接上下文、局部决策、远端摘要、全局里程碑）。
- **系统局限性**：ECHO 缺乏对特工底层权利义务的定义（缺少 System Prompt 校验）。长文本切片仅仅是数据降维，当面对复杂的“上游出图、中游眼瞎、下游乱干”的连环车祸时，依然无力进行真正的逻辑阻断。而 AgentJury 则通过岗位职责对齐完美填补了这一空白。

## 4.已经完成的工作

项目概况
该项目是一个多智能体系统自动故障归因（Automated Failure Attribution）工具，目标是自动识别多 Agent 协作任务中"是哪个 Agent 在哪一步出了错"。核心架构为探针（Probe）+ DAO 三专家裁决的两阶段流程。

已完成的工作

1. Bug 修复

修复了 api_utils.py 中因手动删减代码引发的 IndentationError（缩进错误），并重构了 Agent-步骤绑定逻辑，改用动态变量 index_agent 替代硬编码的 "name"。

2. 探针盲区兜底（Step 0 Failsafe）

发现探针 Agent 无法对 Step 0 发起指控（因为 Step 0 的错误属于规划失误，执行链表面无误），导致整个归因流程漏报。
解决方案：当探针扫描完所有步骤仍未发起指控（error_found == False），系统自动将责任归咎于 Step 0 及其执行 Agent，日志输出 Planner Failsafe 提示。

3. DAO 聚合逻辑优化（因果时序规则）

原有的 Max Vote 聚合存在"多数压倒少数、错误被高分淹没"的问题。
新规则：从三位专家的判决中，筛选 confidence > 0.85 的高置信度候选，再取其中**步骤序号最小（最早发生）**的 Step 作为 Root Cause，并从真实日志中精确捞取该 Step 对应的 Agent 名称（避免专家写错名字的问题）。
明确区分了两种场景的处理逻辑：探针漏报 → 直接输出 Step 0；DAO 分歧 → 仍走高置信度+时序优先的聚合，不强行覆盖为 Step 0。

4. Hand-Crafted 数据集支持（ECHO 架构化压缩）

新增了对 Hand-Crafted（HC）格式数据集的支持，引入两阶段 ECHO 流程：

Phase 1（结构化压缩）：将每步日志压缩为 Node_Action / Result_and_Feedback / Key_Insight 三键 JSON 摘要，减少 Token 压力。
Phase 3（DAO 仲裁）：对当前被审判步骤保留原文（截断），对远处历史步骤使用压缩摘要，实现"细节定罪 + 全局对齐"。

将架构化压缩逻辑剥离为独立脚本 extract_hc_summaries.py，便于单独评估压缩质量。

5. 压缩效果验证

在 mvp_test_hc_1example 数据集上运行验证，输出 structured_abstract_hc_1.json，确认：

索引从 1 开始，与原始日志 history 的 0-based 索引对齐（Step 0 为 human 提问，跳过）。
压缩摘要准确提炼了关键错误信号（如 WebSurfer 跳转到错误页面），有效增强了 DAO 的归因精度，并规避了长文本导致注意力分散的问题。

## 5.已经取得的实验成果

who&when测试集一共有两个分别是Algorithm-Generated（126）和hand-crafted（58）

Algorithm-Generated（126）with GT结果
归因维度 命中记录 (正确数/总数) 综合准确率
Top-1 Agent (单一代理) 79 / 126 62.70%
Top-2 Agent (代理候选) 114 / 126 90.48%
Top-1 Step (单一步骤) 63 / 126 50%
Top-2 Step (步骤候选) 89 / 126 70.63%

Algorithm-Generated（126）without GT结果
归因维度 命中记录 (正确数/总数) 综合准确率
Top-1 Agent (单一代理) 76 / 126 60.32%
Top-2 Agent (代理候选) 100 / 126 79.37%
Top-1 Step (单一步骤) 53 / 126 42.06%
Top-2 Step (步骤候选) 77 / 126 61.11%

hc（58）with GT 数据集结果

Top-1 Agent 准确率: 62.07% (36/58)
Top-2 Agent 准确率: 87.93% (51/58)
Top-1 Step 准确率: 29.31% (17/58)
Top-2 Step 准确率: 36.21% (22/58)

hc（58）without GT 数据集结果

Top-1 Agent 准确率: 62.07% (36/58)
Top-2 Agent 准确率: 87.93% (51/58)
Top-1 Step 准确率: 29.31% (17/58)
Top-2 Step 准确率: 36.21% (22/58)

## 6.待进行的实验（消融实验）

消融实验使用数据集Who&When/alg_ablation_subset，消融实验参考的主方法：api_utils.py里面的 AgentJury_alg_enhance函数

变体 1：退化为相对距离视野 (w/o Sliding Spotlight)设置：取消 Phase 2 的探针滑动扫描。模仿 ECHO，直接选定目标节点，仅提供 $\pm 1$ 步全量和其余摘要，交由 DAO 判定。验证目标：证明如果没有“探针滑动带来的绝对平等的高分辨率审查”，隐藏在全局背景中的因果谬误会被直接漏过。函数名：melting_window.py

变体 2：退化为静态预设专家 (w/ Static Heuristic Experts)设置：关闭 Phase 3 的涌现机制。直接使用预设的通用专家 Prompt（如逻辑专家、代码专家）。验证目标：证明在处理复杂系统的特有错误类型时，由真实数据“涌现”出的专家比人工拍脑门的通用专家具有更高的诊断敏锐度。函数名:melting_dao.py

变体 3：移除探针粗筛 (w/o Recursive Localization)设置：跳过 Phase 2，将全局摘要和尽可能多的全量日志一次性喂给 Phase 3 的 DAO。验证目标：证明“先收紧注意力区域，再进行深度推演”是防止 LLM 在长文本中算力崩溃的必须步骤。
函数名:melting_probe.py

## 附录 1. 聚类结果解读

MAS 故障涌现机制与 K=3 合理性：双算法交叉验证分析
一、两种算法的聚类结果精确比对
以下是在 Alg-Only（n=126） 数据集上，K-Means(K=3) 与 BERTopic(min_cluster_size=15) 的详细样本级映射：

K-Means Cluster 1 (n=60, 执行层缺陷) → BERTopic 将其拆分为两个
BERTopic 在这里揭示了 K-Means Cluster 1 内部存在的次级分界：

BERTopic 分裂结果 关键词 n 代表性样本
Topic 0 agent data provided 41 "Propagation of incorrect data from upstream" / "Agent generated inaccurate factual data" / "Inaccurate foundational data to downstream agents"
Topic 1 incorrect result task execution 37 "Incorrect algorithm implementation" / "Task objective misalignment" / "Premature solution synthesis"
K-Means 的视角：把这 60 个样本归为同一类——"执行过程出错"；BERTopic 的视角：进一步区分了"输出端数据质量错误"（T0）和"决策过程逻辑错误"（T1）。两者都正确，只是粒度不同。

K-Means Cluster 0 (n=32, 认知缺陷) + Cluster 2 (n=34, 验证缺陷) → BERTopic 将其合并为一个
这是最关键的发现。BERTopic Topic 2 (n=31, failure data external information) 的样本横跨了 K-Means 的两个簇：

K-Means Cluster 0 中的样本（认知层）：
"Hallucinated content without source verification" ←┐
"Reliance on outdated static internal knowledge" ←┤ 都进入了
"Failure to retrieve external information before processing"←┤ BERTopic
┤ Topic 2
K-Means Cluster 2 中的样本（验证层）： ←┤
"Failure to verify critical external data source" ←┤
"Reliance on unverified external data" ←┤
"Failure to parse information from a provided resource" ←┘
为什么 BERTopic 会合并这两类？ 因为从密度-语义空间的角度，"幻觉（hallucination）"和"未验证外部数据（unverified external data）"在 sentence embedding 中具有相近的语义向量——它们共享一个核心词素：agent 与外部真实世界之间的信息鸿沟。BERTopic 捕捉的是这个底层语义共性；K-Means 则进一步区分了"主动幻觉"与"被动接收但不验证"两个不同的责任归因方向。

BERTopic NOISE (n=17)：极端执行崩溃类
"Agent exhausted interaction limits through repeated incorrect actions"
"Agent exceeded computational resource constraints"
"Agent role boundary violation causing process termination"
"Agent execution logic contains programming errors"
这 17 个被标记为噪声的样本，在 K-Means 中均落在 Cluster 1（执行层），但密度不足以在 BERTopic 中形成独立簇——它们是"代码级崩溃"这一细分类别，语义上介于 Topic 0 和 Topic 1 之间的边缘地带。

二、Combined 结果揭示的 HC 特有维度
合并跑（n=184）时 BERTopic 自发多出了第 4 个主题：

| Topic 3 (n=17) | 关键词：retrieval information retrieval search | HC 占 11/17（64.7%） |

代表性样本全部来自 HC WebSurfer 失败场景：

"Insufficient search parameters for targeted information retrieval"
"Incomplete data retrieval due to partial page processing"
"Temporal scope mismatch between query and data retrieval"
"Insufficient query refinement leading to cascading information retrieval failure"

结论：这个第 4 主题是 HC 数据集引入的特有错误模式——WebSurfer 的搜索/导航执行失败，在 Alg 数据集中不存在对应的密度峰。这也解释了为什么 HC-only 只能找到 2 个干净的簇（31% 噪声）：HC 的错误分布更加异质，有一部分与 Alg 共享三类主错误，另有一部分属于 HC 特有的搜索导航失败。

三、K=3 的合理性：来自两种方法的三角验证
将两种方法的结论整合为一个统一的论证框架：

┌──────────────────────────────────────────────────────────────────┐
│ Alg-Only 故障空间的三角验证 │
├─────────────────────┬────────────────────────────────────────────┤
│ K-Means (K=3) │ BERTopic (min_cluster_size=15) │
├─────────────────────┼────────────────────────────────────────────┤
│ Cluster 0 (n=32) │ ↘ │
│ 认知缺陷 │ Topic 2 (n=31): "failure external data"│
│ Cluster 2 (n=34) │ ↗ [BERTopic 将二者合并为"外部信息边界" │
│ 验证缺陷 │ 因为底层语义向量相近] │
├─────────────────────┼────────────────────────────────────────────┤
│ Cluster 1 (n=60) │ → Topic 0 (n=41): "agent data provided" │
│ 执行缺陷 │ → Topic 1 (n=37): "incorrect task exec" │
│ [最大类] │ [BERTopic 将其拆分为数据输出错误/决策错误]│
│ │ + NOISE (17): 代码崩溃类边缘样本 │
├─────────────────────┴────────────────────────────────────────────┤
│ 两种方法收敛于：Alg 数据 = 3 个主维度，无论算法如何划分边界 │
└──────────────────────────────────────────────────────────────────┘
K=3 合理性的三重论据：

K-Means 惯性边际递减：K=2→3 降幅（-2.33）> K=3→4 降幅（-1.71）> K=4→5 降幅（-1.13），在高维语义空间中肘部平滑是正常的，K=3 是边际收益最高的单步

BERTopic 密度收敛：在 min_cluster_size=15 的条件下，HDBSCAN 在 Alg 数据的语义密度空间中自发找到且仅找到 3 个高密度中心——与 K=3 完全对应，这是两种在数学上完全独立的方法的自然收敛

Combined 的第 4 主题解释：当加入 HC 数据时出现第 4 个主题，但这个主题（HC 搜索导航失败）是数据集特有的噪声维度，不属于 MAS 通用故障分类——它的出现反而反证了 K=3 在 Alg 上的完备性
