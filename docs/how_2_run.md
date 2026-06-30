# AgentJury 当前实验运行手册

本文档记录当前代码环境下的 AgentJury 实验入口。仓库复用了 Who&When 的数据格式与部分基线代码，但本文工作的最终主方法是 AgentJury，不是早期的 `step_by_step_dao_echo` 或 `step_by_step_dao_hc`。

所有命令默认在 `Automated_FA/` 目录下执行：

```bash
cd Automated_FA
```

---

## 1. 当前最终主方法

| 数据集 | 命令行 `--method` | 实际函数 | 代码位置 | 说明 |
|---|---|---|---|---|
| Algorithm-Generated | `agentjury_alg_enhance` | `AgentJury_alg_enhance` | `Lib/api_utils.py` | 当前 Alg 最终主方法 |
| Hand-Crafted | `agentjury_hc_enhance` | `AgentJury_hc_enhance` | `Lib/api_utils_4_handcrafted.py` | 当前 HC 最终主方法 |
| Algorithm-Generated noGT | `agentjury_alg_enhance_nogt` | `AgentJury_alg_enhance_noGT` | `Lib/api_utils.py` | 无真值辅助对照设置 |

不要把 `step_by_step_dao_echo` 和 `step_by_step_dao_hc` 当作当前最终主实验入口；它们是较早的中间版本或基线式实现。

---

## 2. 数据集路径

| 数据集 | 路径 | 样本数 | 用途 |
|---|---|---:|---|
| Alg-Gen 全集 | `../Who&When/Algorithm-Generated` | 126 | Algorithm-Generated 主实验 |
| HC 全集 | `../Who&When/Hand-Crafted` | 58 | Hand-Crafted 主实验 |
| Alg 消融子集 | `../Who&When/alg_ablation_subset` | 60 | 消融、参数敏感性、留出验证 |
| Alg-Gen 1-20 | `../Who&When/mvp_test_1_20example` | 20 | 分批运行/调试 |
| Alg-Gen 21-40 | `../Who&When/mvp_test_21_40example` | 20 | 分批运行/调试 |
| Alg-Gen 41-60 | `../Who&When/mvp_test_41_60example` | 20 | 分批运行/调试 |
| Alg-Gen 61-80 | `../Who&When/mvp_test_61_80example` | 20 | 分批运行/调试 |
| Alg-Gen 81-126 | `../Who&When/mvp_test_81_126example` | 46 | 分批运行/调试 |
| HC 1 样本 | `../Who&When/mvp_test_hc_1example` | 1 | 冒烟测试 |
| HC 1-5 | `../Who&When/mvp_test_hc_1_5example` | 5 | 小批量测试 |
| HC 6-20 | `../Who&When/mvp_test_hc_6_20example` | 15 | 分批运行 |
| HC 21-30 | `../Who&When/mvp_test_hc_21_30example` | 10 | 分批运行 |
| HC 31-58 | `../Who&When/mvp_test_hc_31_58example` | 28 | 分批运行 |

---

## 3. 模型别名

`inference.py` 当前支持的 API 模型别名来自 `Lib/api_utils.py::API_MODEL_MAP`：

| 别名 | 实际模型 |
|---|---|
| `ds-v3.2` | `deepseek-ai/DeepSeek-V3.2` |
| `ds-r1` | `deepseek-ai/DeepSeek-R1` |
| `glm-4.7` | `Pro/zai-org/GLM-4.7` |
| `minimax-2.5` | `Pro/MiniMaxAI/MiniMax-M2.5` |
| `kimi-2.5` | `Pro/moonshotai/Kimi-K2.5` |
| `qwen-3.5` | `Qwen/Qwen3.5-397B-A17B` |

当前已完成主实验主要使用 `ds-v3.2`。基座模型适配性分析中还使用过 `glm-4.7`、`kimi-2.5`、`qwen-3.5`。

---

## 4. 主实验命令

### 4.1 Algorithm-Generated 主方法

```bash
python inference.py \
  --method agentjury_alg_enhance \
  --model ds-v3.2 \
  --directory_path "../Who&When/Algorithm-Generated" \
  --max_tokens 1500
```

分批运行示例：

```bash
python inference.py \
  --method agentjury_alg_enhance \
  --model ds-v3.2 \
  --directory_path "../Who&When/mvp_test_1_20example" \
  --max_tokens 1500
```

### 4.2 Algorithm-Generated 无真值辅助版本

```bash
python inference.py \
  --method agentjury_alg_enhance_nogt \
  --model ds-v3.2 \
  --directory_path "../Who&When/Algorithm-Generated" \
  --max_tokens 1500
```

### 4.3 Hand-Crafted 主方法

```bash
python inference.py \
  --method agentjury_hc_enhance \
  --model ds-v3.2 \
  --directory_path "../Who&When/Hand-Crafted" \
  --is_handcrafted True \
  --max_tokens 1500
```

HC 小批量运行示例：

```bash
python inference.py \
  --method agentjury_hc_enhance \
  --model ds-v3.2 \
  --directory_path "../Who&When/mvp_test_hc_1_5example" \
  --is_handcrafted True \
  --max_tokens 1500
```

---

## 5. 当前主方法内部关键参数

这些参数当前在函数内部硬编码。如果要做参数敏感性分析，建议先把它们参数化后再运行，避免手动改代码造成记录混乱。

| 数据集 | 文件 | 函数 | 参数 | 当前值 |
|---|---|---|---|---:|
| Alg | `Lib/api_utils.py` | `AgentJury_alg_enhance` | `WIN_HALF` | 2 |
| Alg | `Lib/api_utils.py` | `AgentJury_alg_enhance` | `PROBE_CONF_THRESH` | 0.3 |
| Alg | `Lib/api_utils.py` | `AgentJury_alg_enhance` | `TOP_K_FLAGGED` | 3 |
| Alg | `Lib/api_utils.py` | `AgentJury_alg_enhance` | `CONVICTION_THRESHOLD` | 0.3 |
| HC | `Lib/api_utils_4_handcrafted.py` | `AgentJury_hc_enhance` | `WIN_HALF` | 3 |
| HC | `Lib/api_utils_4_handcrafted.py` | `AgentJury_hc_enhance` | `PROBE_CONF_THRESH` | 0.5 |
| HC | `Lib/api_utils_4_handcrafted.py` | `AgentJury_hc_enhance` | `TOP_K_FLAGGED` | 3 |
| HC | `Lib/api_utils_4_handcrafted.py` | `AgentJury_hc_enhance` | `CONVICTION_THRESHOLD` | 0.5 |

`inference.py` 已有以下与探针相关的命令行参数：

| 参数 | 作用 | 默认值 | 当前适用 |
|---|---|---:|---|
| `--probe_k` | Alg k-partition probe 分片数 | 4 | `agentjury_alg_enhance`、`agentjury_alg_enhance_nogt` |
| `--target_seg_size` | HC 自适应探针每段目标步数 | 15 | `agentjury_hc_enhance` |
| `--max_probe_k` | HC 自适应探针分片上限 | 8 | `agentjury_hc_enhance` |

---

## 6. 消融实验

当前三类主要消融在 `Lib/alg_melting.py` 中实现，参考主方法为 `AgentJury_alg_enhance`。

| 消融目标 | 命令行 `--method` | 实际函数 | 说明 |
|---|---|---|---|
| w/o Sliding Spotlight | `ablation_no_spotlight` | `AgentJury_ablation_no_spotlight` | 取消滑动探针的选择性高分辨率审查 |
| w/ Static Heuristic Experts | `ablation_static_experts` | `AgentJury_ablation_static_experts` | 将数据聚类专家替换为静态通用专家 |
| w/o Recursive Localization | `ablation_no_probe` | `AgentJury_ablation_no_probe` | 跳过探针粗筛，直接给 DAO 全局摘要 |

建议统一在 `alg_ablation_subset` 上运行：

```bash
python inference.py \
  --method ablation_no_spotlight \
  --model ds-v3.2 \
  --directory_path "../Who&When/alg_ablation_subset" \
  --max_tokens 1500

python inference.py \
  --method ablation_static_experts \
  --model ds-v3.2 \
  --directory_path "../Who&When/alg_ablation_subset" \
  --max_tokens 1500

python inference.py \
  --method ablation_no_probe \
  --model ds-v3.2 \
  --directory_path "../Who&When/alg_ablation_subset" \
  --max_tokens 1500
```

---

## 7. 参数敏感性与留出验证

### 7.1 参数敏感性建议

论文退修阶段建议只做单因素敏感性分析，避免二维网格放大成本。

推荐设置：

| 参数 | 建议取值 | 固定其他参数 |
|---|---|---|
| 窗口邻域半径 `WIN_HALF` / `Delta` | `{1,2,3,4,5}` | 固定置信度阈值 |
| 置信度阈值 `PROBE_CONF_THRESH` | `{0.3,0.5,0.7,0.9}` | 固定窗口半径 |

当前代码尚未把 `WIN_HALF` 和 `PROBE_CONF_THRESH` 暴露为 CLI 参数。若要系统运行敏感性实验，建议新增：

```text
--context_radius
--probe_conf_threshold
```

并传入 `AgentJury_alg_enhance` / `AgentJury_hc_enhance`，不要每次手动修改函数内部常量。

### 7.2 领域内留出验证建议

为回应“聚类先验与评估集同源”问题，可在同一数据池内做领域内留出验证：

1. 按故障类型分层切分数据。
2. 一半样本只用于构造聚类先验、主题词、仲裁者提示或向量库。
3. 另一半样本只用于测试和评估。
4. 正文中称为“领域内留出验证”，不要表述为跨系统泛化证明。

---

## 8. 双轨聚类与聚类先验代码分布

本节对应论文中“数据驱动 DAO 仲裁者 / 聚类先验”的代码链路。这里的“双轨聚类”指两条用于发现和交叉验证错误类型结构的聚类路线：

1. **K-Means 路线**：形成 K=3 专家维度，是最终 AgentJury DAO 专家提示词的主要来源。
2. **BERTopic 路线**：用于主题发现、关键词解释和与 K-Means 结果交叉验证；它主要服务论文解释与合理性论证，不是最终推理时的在线模块。

### 8.1 聚类脚本组织

所有聚类分析脚本集中在 `../extract_of_error/`：

| 路线 | 文件 | 输入 | 输出 | 当前定位 |
|---|---|---|---|---|
| K-Means | `extract_D.py` | `extracted_reasons_all.csv` 或数据集 JSON | `cluster_results.txt` | 早期总管脚本：LLM 标准化失败原因 + K=2..5 聚类。注意其默认数据路径带有历史痕迹，复跑前应检查 `DATASETS` 路径。 |
| K-Means | `cluster_raw_reasons.py` | Alg/HC 原始 JSON 的 `mistake_reason` | `cluster_raw_alg.txt`、`cluster_raw_hc.txt`、`cluster_raw_all.txt` | 当前更直接的原始错误原因聚类脚本。 |
| K-Means | `plot_cluster_scatter.py` | Alg/HC 原始 JSON 的 `mistake_reason` | `cluster_scatter_alg.*`、`cluster_scatter_hc.*`、`cluster_scatter_all.*`、`emb_cache_*.npy` | 生成 K=3 UMAP 散点图，并缓存 BGE-M3 嵌入。 |
| K-Means | `plot_elbow_curves.py` | `emb_cache_alg.npy`、`emb_cache_hc.npy` | `elbow_curves.*` | 基于 K=2..5 惯量绘制肘部曲线。需先运行 `plot_cluster_scatter.py`。 |
| BERTopic | `bertopic_clustering.py` | `extracted_reasons_all.csv` | `bertopic_results/*_topic_info.txt`、`*_topics.csv`、`*_umap2d.*` | BERTopic 主题聚类主脚本，支持 combined / alg_only / hc_only。 |
| BERTopic | `plot_topic_keywords.py` | `extracted_reasons_all.csv` | `bertopic_results/topic_keywords_*.*` | 绘制 c-TF-IDF 主题关键词图。 |

已有关键聚类输出：

| 文件或目录 | 含义 |
|---|---|
| `../extract_of_error/cluster_results.txt` | 标准化错误原因的 K-Means K=2..5 聚类结果，`K=3` 段被用于构建三类专家先验。 |
| `../extract_of_error/cluster_raw_alg.txt` | Algorithm-Generated 原始 `mistake_reason` 的 K-Means 聚类结果。 |
| `../extract_of_error/cluster_raw_hc.txt` | Hand-Crafted 原始 `mistake_reason` 的 K-Means 聚类结果。 |
| `../extract_of_error/cluster_raw_all.txt` | Alg + HC 合并后的原始错误原因聚类结果。 |
| `../extract_of_error/cluster_scatter_alg.*` | Alg K=3 聚类散点图，支持论文作图。 |
| `../extract_of_error/cluster_scatter_hc.*` | HC K=3 聚类散点图，支持论文作图。 |
| `../extract_of_error/elbow_curves.*` | K=2..5 肘部曲线图。 |
| `../extract_of_error/bertopic_results/` | BERTopic 主题表、主题分配 CSV、UMAP 图和关键词图。 |

### 8.2 K-Means 路线复跑命令

从仓库根目录进入聚类目录：

```bash
cd extract_of_error
```

直接对原始 `mistake_reason` 聚类：

```bash
python3 cluster_raw_reasons.py
```

生成 K=3 聚类散点图和嵌入缓存：

```bash
python3 plot_cluster_scatter.py
```

生成肘部曲线：

```bash
python3 plot_elbow_curves.py
```

早期标准化原因聚类脚本：

```bash
python3 extract_D.py --phase 2
```

如果要重新运行 `extract_D.py --phase 1` 或 `cluster_raw_reasons.py`，需要注意这些脚本会调用 SiliconFlow/OpenAI-compatible API，并且代码里存在历史硬编码 API 配置；公开仓库前应改为环境变量或本地私有配置。

### 8.3 BERTopic 路线复跑命令

BERTopic 依赖 `extracted_reasons_all.csv`。默认使用 `standardized` 字段：

```bash
cd extract_of_error

python3 bertopic_clustering.py \
  --field standardized \
  --min_cluster_size 3
```

如果要用原始错误原因字段：

```bash
python3 bertopic_clustering.py \
  --field mistake_reason \
  --min_cluster_size 3
```

绘制主题关键词图：

```bash
python3 plot_topic_keywords.py
```

论文解释中应将 BERTopic 表述为“对 K-Means 聚类语义结构的交叉验证和补充解释”。不要写成最终推理阶段实时调用 BERTopic。

### 8.4 聚类先验如何进入 AgentJury

K-Means K=3 的语义维度被固化为 AgentJury 的三个 DAO 专家视角：

| 专家维度 | 含义 | 主要代码位置 |
|---|---|---|
| `factual` | 事实、检索、幻觉、未验证信息错误 | `Lib/api_utils.py` 的 Alg DAO prompt；`Lib/api_utils_4_handcrafted.py` 的 HC DAO prompt |
| `logic` | 逻辑推理、算法、规划、决策错误 | `Lib/api_utils.py`；`Lib/api_utils_4_handcrafted.py` |
| `agentexec` | 执行、协作、角色边界、动作遗漏错误 | `Lib/api_utils.py`；`Lib/api_utils_4_handcrafted.py` |

最终主方法中直接使用这些聚类先验的位置：

| 数据集 | 函数 | 文件 | 说明 |
|---|---|---|---|
| Alg | `AgentJury_alg_enhance` | `Automated_FA/Lib/api_utils.py` | 使用 Alg 聚类分类体系构造 factual / logic / execution 三名 DAO 专家。 |
| HC | `AgentJury_hc_enhance` | `Automated_FA/Lib/api_utils_4_handcrafted.py` | 在 HC 场景中转写为 physical action / decision-logic / information-quality 三个更适合 WebSurfer 场景的专家视角。 |

### 8.5 对论文表述的边界

可以在论文中表述：

- K-Means 与 BERTopic 构成双轨聚类验证，支持 K=3 专家维度的合理性。
- K-Means K=3 聚类结果是最终 DAO 专家视角的主要工程来源。
- BERTopic 揭示了 Alg 与 HC 错误分布的异质性，特别是 HC 的搜索/导航失败更分散，因此它更适合用于解释而非直接决定专家数量。

不建议表述：

- “BERTopic 自动发现了所有最终专家”。HC 中第三类专家包含人工语义归并，必须说明人工确认规则。
- “双轨聚类证明跨系统泛化”。它只能支持领域内错误结构的合理性，不能替代跨系统或跨数据集验证。

---

## 10. 已有关键输出文件

| 文件 | 含义 |
|---|---|
| `outputs/AgentJury_alg_enhance_all.txt` | Alg 主方法全集结果 |
| `outputs/AgentJury_alg_enhance_noGT_all.txt` | Alg noGT 全集结果 |
| `outputs/_merged_alg_enhance_1_126.txt` | Alg 分批合并结果 |
| `outputs/_merged_alg_enhance_1_126_nogt.txt` | Alg noGT 分批合并结果 |
| `outputs/_merged_AgentJury_hc_enhance_0415_all.txt` | HC 主方法合并结果 |
| `outputs/_merged_AgentJury_hc_enhance_all_noGT.txt` | HC noGT 合并结果 |
| `outputs/ablation_no_spotlight_ds-v3.2_alg_alg_ablation_subset_0414_233214.txt` | w/o Sliding Spotlight 消融 |
| `outputs/ablation_static_experts_ds-v3.2_alg_alg_ablation_subset_0414_233306.txt` | Static Experts 消融 |
| `outputs/ablation_no_probe_ds-v3.2_alg_alg_ablation_subset_0414_233357.txt` | w/o Recursive Localization 消融 |

---

## 11. 历史入口与中间版本

以下方法保留在代码中，但不是当前论文最终主方法：

| `--method` | 当前定位 |
|---|---|
| `step_by_step_dao_echo` | 早期 ECHO/DAO 结构版本 |
| `step_by_step_dao_echo_noGT` | 早期无真值辅助版本 |
| `step_by_step_dao_hc` | HC 早期适配版本 |
| `agentJury_enhance` | 中间增强版本 |
| `ablation_generic`、`ablation_k2`、`ablation_k4`、`ablation_k5` | 早期 DAO 提示词/专家数量消融 |
| `all_at_once`、`step_by_step`、`binary_search` | Who&When 复用基线入口 |

后续写论文、跑退修补实验或整理复现脚本时，应优先使用第 1 节列出的最终主方法。
