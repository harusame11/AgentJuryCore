"""
plot_topic_keywords.py
======================
对 combined / alg_only / hc_only 三个子集分别重跑 BERTopic（min_cluster_size=15），
取每个 Topic 的 c-TF-IDF 关键词权重，绘制水平 bar chart 网格。

输出：
  bertopic_results/topic_keywords_combined.pdf/svg/png
  bertopic_results/topic_keywords_alg_only.pdf/svg/png
  bertopic_results/topic_keywords_hc_only.pdf/svg/png

用法：
    python3 plot_topic_keywords.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# ── 路径 ──────────────────────────────────────────────────────────────────────
DATA_PATH   = Path(__file__).parent / "extracted_reasons_all.csv"
OUT_DIR     = Path(__file__).parent / "bertopic_results"
EMBED_MODEL = "all-MiniLM-L6-v2"
FIELD       = "standardized"
MIN_CLUSTER = 15
TOP_N_WORDS = 10   # 每个主题展示的关键词数

# ── 人工可读的 Topic 标签（基于 combined 语义内容）────────────────────────────
# 如不确定可留空，脚本会自动用 BERTopic 生成的关键词拼接
TOPIC_LABEL_OVERRIDE: dict[tuple[str, int], str] = {
    # ("combined", 0): "Data Validation & Execution Errors",
    # ("combined", 1): "Agent Data Inaccuracy",
    # ("combined", 2): "Search & Retrieval Failures",
    # ("combined", 3): "Code & Navigation Errors",
}

PALETTE = [
    "#2563EB", "#16A34A", "#DC2626", "#D97706",
    "#7C3AED", "#0891B2", "#DB2777", "#65A30D",
]


# ── BERTopic 构建 ─────────────────────────────────────────────────────────────
def build_model(n: int) -> BERTopic:
    n_neighbors = max(3, min(10, n // 5))
    umap_model = UMAP(
        n_neighbors=n_neighbors, n_components=5,
        min_dist=0.0, metric="cosine", random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER, min_samples=1,
        metric="euclidean", cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=1,
    )
    return BERTopic(
        embedding_model=EMBED_MODEL,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=False,
    )


# ── 绘制单个数据集的关键词 bar chart（每个 Topic 独立保存）─────────────────────
def plot_keywords(model: BERTopic, dataset_name: str, n_texts: int, out_dir: Path):
    topic_info = model.get_topic_info()
    valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
    k = len(valid_topics)

    if k == 0:
        print(f"  [SKIP] No topics found for {dataset_name}")
        return

    topic_out_dir = out_dir / f"topic_keywords_{dataset_name}"
    topic_out_dir.mkdir(exist_ok=True)

    for idx, tid in enumerate(valid_topics):
        words_scores = model.get_topic(tid)
        if not words_scores:
            continue

        ws = sorted(words_scores[:TOP_N_WORDS], key=lambda x: x[1])
        words  = [w for w, _ in ws]
        scores = [s for _, s in ws]

        color = PALETTE[idx % len(PALETTE)]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(words, scores, color=color, alpha=0.82,
                       edgecolor="white", linewidth=0.5)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + max(scores) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va="center", ha="left",
                    fontsize=8, color="#333333")

        override_key = (dataset_name, tid)
        if override_key in TOPIC_LABEL_OVERRIDE:
            topic_title = TOPIC_LABEL_OVERRIDE[override_key]
        else:
            kw_str = ", ".join([w for w, _ in model.get_topic(tid)[:4]])
            topic_title = f"Topic {tid}  [{kw_str}]"

        n_topic = topic_info[topic_info["Topic"] == tid]["Count"].values[0]
        ax.set_title(f"{topic_title}  (n={n_topic})",
                     fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("c-TF-IDF Score", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#333333")
        ax.set_xlim(0, max(scores) * 1.18)
        fig.tight_layout()

        stem = topic_out_dir / f"topic_{tid:02d}"
        fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
        fig.savefig(str(stem) + ".svg", bbox_inches="tight")
        fig.savefig(str(stem) + ".png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"    Topic {tid} → {stem}.pdf / .svg / .png")

    print(f"  [DONE] {k} independent charts saved → {topic_out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=[FIELD]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows  |  field='{FIELD}'  |  min_cluster_size={MIN_CLUSTER}")

    subsets = {
        "combined": df,
        "alg_only": df[df["split"] == "Algorithm-Generated"].reset_index(drop=True),
        "hc_only":  df[df["split"] == "Hand-Crafted"].reset_index(drop=True),
    }

    for name, sub_df in subsets.items():
        texts = sub_df[FIELD].tolist()
        if len(texts) < MIN_CLUSTER:
            print(f"\n[SKIP] {name}: too few samples ({len(texts)})")
            continue

        print(f"\n{'='*55}")
        print(f"  {name}  (n={len(texts)})")
        print(f"{'='*55}")

        model = build_model(len(texts))
        topics, _ = model.fit_transform(texts)

        n_valid = len([t for t in topics if t != -1])
        n_noise = len([t for t in topics if t == -1])
        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
        print(f"  Topics: {len(valid_topics)}  |  Noise: {n_noise}  |  Assigned: {n_valid}")

        for tid in valid_topics:
            ws = model.get_topic(tid)[:6]
            kw = ", ".join(f"{w}({s:.3f})" for w, s in ws)
            cnt = topic_info[topic_info["Topic"] == tid]["Count"].values[0]
            print(f"    Topic {tid} (n={cnt}): {kw}")

        plot_keywords(model, name, len(texts), OUT_DIR)

    print("\nAll done.")


if __name__ == "__main__":
    main()
