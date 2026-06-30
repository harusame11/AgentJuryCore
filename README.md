# AgentJuryCore

Official implementation of **AgentJury**, a multi-agent failure attribution framework for localizing the responsible agent and step in collaborative agent traces.

AgentJury combines mixed-resolution probing, data-driven DAO-style expert arbitration, and top-k tolerant decoding. This release keeps the public inference pipeline and clustering utilities, while omitting evaluation scripts and retrieval-augmented components.

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

Prepare datasets as JSON directories. The expected fields are:

- Algorithm-Generated: `history`, `question`, `ground_truth`, `system_prompt`, `mistake_agent`, `mistake_step`, `mistake_reason`.
- Hand-Crafted: `history`, `question`, `mistake_agent`, `mistake_step`, `mistake_reason`; `ground_truth` is optional for `agentjury_hc_enhance_nogt`.

Example layout:

```text
data/
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
  --directory_path data/Algorithm-Generated \
  --max_tokens 1500
```

Hand-Crafted main method:

```bash
python Automated_FA/inference.py \
  --method agentjury_hc_enhance \
  --model ds-v3.2 \
  --directory_path data/Hand-Crafted \
  --is_handcrafted True \
  --max_tokens 1500
```

Hand-Crafted no-ground-truth variant:

```bash
python Automated_FA/inference.py \
  --method agentjury_hc_enhance_nogt \
  --model ds-v3.2 \
  --directory_path data/Hand-Crafted \
  --is_handcrafted True \
  --max_tokens 1500
```

Run logs are written to `outputs/` by default.

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
