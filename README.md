# AgentJuryCore

Official implementation of **AgentJury**, a multi-agent failure attribution framework for localizing the responsible agent and step in collaborative agent traces.

AgentJury combines mixed-resolution probing, data-driven DAO-style expert arbitration, and top-k tolerant decoding. This release keeps the public inference pipeline and clustering utilities, while omitting evaluation scripts and retrieval-augmented components.

## Codebase Description

The repository is organized around two Who&When data regimes. Algorithm-generated
traces use agent names and system-prompt dictionaries, while hand-crafted traces
use composite role strings from Magnetic-One-style conversations. Both pipelines
share the same four-stage design:

1. compress long trajectory steps into structured summaries;
2. locate suspicious regions with partitioned probes;
3. expand high-resolution windows around the strongest candidates;
4. aggregate three specialized DAO judges with dense soft voting.

`Automated_FA/inference.py` is the only public command-line entry point. Core
Algorithm-Generated logic lives in `Automated_FA/Lib/api_utils.py`; Hand-Crafted
adaptation lives in `Automated_FA/Lib/api_utils_4_handcrafted.py`; controlled
ablations live in `Automated_FA/Lib/alg_melting.py`.

## Overview

Given a multi-agent trajectory, AgentJury predicts:

- the agent most responsible for the final failure;
- the earliest step where the failure was introduced;
- a natural-language rationale for the attribution.

The public CLI exposes only the current paper methods:

| Method | Split | Description |
|---|---|---|
| `agentjury_alg_enhance` | Algorithm-Generated | Main AgentJury method for algorithm-generated traces |
| `agentjury_alg_enhance_nogt` | Algorithm-Generated | No-ground-truth-control variant |
| `agentjury_hc_enhance` | Hand-Crafted | Main AgentJury method for hand-crafted WebSurfer-style traces |
| `agentjury_hc_enhance_nogt` | Hand-Crafted | No-ground-truth-control variant for hand-crafted traces |
| `ablation_no_spotlight` | Algorithm-Generated | Removes selective sliding spotlight expansion |
| `ablation_static_experts` | Algorithm-Generated | Replaces data-driven experts with static generic experts |
| `ablation_no_probe` | Algorithm-Generated | Removes recursive localization probe |

Historical baselines and development variants remain in `Automated_FA/Lib/` for auditability, but they are not exposed as public CLI entry points.

## Paper-to-Code Mapping

| Paper experiment | CLI method | Python function | Source file |
|---|---|---|---|
| Algorithm-Generated main result | `agentjury_alg_enhance` | `AgentJury_alg_enhance` | `Automated_FA/Lib/api_utils.py` |
| Algorithm-Generated noGT control | `agentjury_alg_enhance_nogt` | `AgentJury_alg_enhance_noGT` | `Automated_FA/Lib/api_utils.py` |
| Hand-Crafted main result | `agentjury_hc_enhance` | `AgentJury_hc_enhance` | `Automated_FA/Lib/api_utils_4_handcrafted.py` |
| Hand-Crafted noGT control | `agentjury_hc_enhance_nogt` | `AgentJury_hc_enhance_noGT` | `Automated_FA/Lib/api_utils_4_handcrafted.py` |
| w/o Sliding Spotlight | `ablation_no_spotlight` | `AgentJury_ablation_no_spotlight` | `Automated_FA/Lib/alg_melting.py` |
| Static Heuristic Experts | `ablation_static_experts` | `AgentJury_ablation_static_experts` | `Automated_FA/Lib/alg_melting.py` |
| w/o Recursive Localization | `ablation_no_probe` | `AgentJury_ablation_no_probe` | `Automated_FA/Lib/alg_melting.py` |

## Repository Layout

```text
AgentJuryCore/
├── Automated_FA/
│   ├── inference.py                 # public experiment entry point
│   ├── Lib/
│   │   ├── api_utils.py             # Algorithm-Generated AgentJury logic
│   │   ├── api_utils_4_handcrafted.py
│   │   ├── alg_melting.py           # ablation methods
│   │   ├── local_model.py
│   │   └── utils.py
│   ├── extract_ablation_subset.py
│   ├── extract_hc_summaries.py
│   └── sample_ablation_subset.py
├── extract_of_error/                # clustering and analysis utilities
├── docs/                            # project notes and runbook
├── requirements.txt
└── .env.example
```

## Setup

```bash
conda create -n agentjury python=3.10 -y
conda activate agentjury
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your OpenAI-compatible API credentials:

```bash
SILICON_API_KEY=your_api_key_here
SILICON_BASE_URL=https://api.siliconflow.cn/v1
```

The code uses model aliases defined in `Automated_FA/Lib/api_utils.py`:

| Alias | Model |
|---|---|
| `ds-v3.2` | `deepseek-ai/DeepSeek-V3.2` |
| `ds-r1` | `deepseek-ai/DeepSeek-R1` |
| `glm-4.7` | `Pro/zai-org/GLM-4.7` |
| `minimax-2.5` | `Pro/MiniMaxAI/MiniMax-M2.5` |
| `kimi-2.5` | `Pro/moonshotai/Kimi-K2.5` |
| `qwen-3.5` | `Qwen/Qwen3.5-397B-A17B` |

## Data

Who&When is released with the ICML 2025 Spotlight paper
[Which Agent Causes Task Failures and When?](https://openreview.net/forum?id=GazlTYxZss).
Clone the official benchmark repository:

```bash
git clone --depth 1 \
  https://github.com/ag2ai/Agents_Failure_Attribution.git \
  data/Agents_Failure_Attribution
```

The paths used by this repository are then:

```text
data/Agents_Failure_Attribution/Who&When/Algorithm-Generated
data/Agents_Failure_Attribution/Who&When/Hand-Crafted
```

The official release contains 126 Algorithm-Generated and 58 Hand-Crafted JSON
traces. The local `data/` directory is ignored by Git.

Prepare datasets as JSON directories. The expected fields are:

- Algorithm-Generated: `history`, `question`, `system_prompt`, `mistake_agent`, `mistake_step`, `mistake_reason`; `ground_truth` is optional for `agentjury_alg_enhance_nogt`.
- Hand-Crafted: `history`, `question`, `mistake_agent`, `mistake_step`, `mistake_reason`; `ground_truth` is optional for `agentjury_hc_enhance_nogt`.

Example layout:

```text
data/Agents_Failure_Attribution/Who&When/
├── Algorithm-Generated/
│   ├── 1.json
│   └── ...
└── Hand-Crafted/
    ├── 1.json
    └── ...
```

## Running AgentJury

Run commands from the repository root.

Algorithm-Generated main method:

```bash
python Automated_FA/inference.py \
  --method agentjury_alg_enhance \
  --model ds-v3.2 \
  --directory_path "data/Agents_Failure_Attribution/Who&When/Algorithm-Generated" \
  --max_tokens 1500
```

Algorithm-Generated no-ground-truth variant:

```bash
python Automated_FA/inference.py \
  --method agentjury_alg_enhance_nogt \
  --model ds-v3.2 \
  --directory_path "data/Agents_Failure_Attribution/Who&When/Algorithm-Generated" \
  --max_tokens 1500
```

Hand-Crafted main method:

```bash
python Automated_FA/inference.py \
  --method agentjury_hc_enhance \
  --model ds-v3.2 \
  --directory_path "data/Agents_Failure_Attribution/Who&When/Hand-Crafted" \
  --is_handcrafted True \
  --max_tokens 1500
```

Hand-Crafted no-ground-truth variant:

```bash
python Automated_FA/inference.py \
  --method agentjury_hc_enhance_nogt \
  --model ds-v3.2 \
  --directory_path "data/Agents_Failure_Attribution/Who&When/Hand-Crafted" \
  --is_handcrafted True \
  --max_tokens 1500
```

Run logs are written to `outputs/` by default.

## Smoke Test

The repository includes an offline end-to-end smoke test that reads the official
`Algorithm-Generated/1.json` and `Hand-Crafted/1.json` traces and mocks only
remote model responses. The test exercises dataset parsing, structured summaries, partitioned probes, DAO arbitration,
dense voting, and final attribution output:

```bash
python -m unittest -v tests.test_alg_nogt_smoke tests.test_hc_nogt_smoke
```

For a live API run, set `SILICON_API_KEY` in `.env` and run the desired command above.
The offline smoke test validates code integration without API cost; it does not
measure model quality.

## Ablations

```bash
python Automated_FA/inference.py \
  --method ablation_no_spotlight \
  --model ds-v3.2 \
  --directory_path data/alg_ablation_subset

python Automated_FA/inference.py \
  --method ablation_static_experts \
  --model ds-v3.2 \
  --directory_path data/alg_ablation_subset

python Automated_FA/inference.py \
  --method ablation_no_probe \
  --model ds-v3.2 \
  --directory_path data/alg_ablation_subset
```

## Clustering Utilities

The `extract_of_error/` directory contains scripts for deriving and visualizing failure-mode clusters.

Cluster raw failure reasons:

```bash
python extract_of_error/cluster_raw_reasons.py \
  --alg_dir data/Algorithm-Generated \
  --hc_dir data/Hand-Crafted
```

Plot K=3 failure-cluster scatter figures:

```bash
python extract_of_error/plot_cluster_scatter.py \
  --alg_dir data/Algorithm-Generated \
  --hc_dir data/Hand-Crafted
```

## Notes for Reproducibility

- API credentials are read from environment variables or CLI flags. No private keys are required in source files.
- Dataset paths are always passed explicitly with `--directory_path`, `--alg_dir`, or `--hc_dir`.
- Cached summaries and run outputs are written under `outputs/`.
- This release does not include evaluation scripts or retrieval-augmented modules.

## Citation

If this repository helps your research, please cite the associated paper:

```bibtex
@misc{agentjury2026,
  title  = {AgentJury: Failure Attribution for Multi-Agent Systems},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Code release}
}
```

## License

The license will be specified by the repository owner before public release.
