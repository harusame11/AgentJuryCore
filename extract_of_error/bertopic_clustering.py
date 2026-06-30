"""
bertopic_clustering.py
======================
对 extracted_reasons_all.csv 中的 mistake_reason / standardized 字段
使用 BERTopic 进行主题聚类，产出三个结果：
  1. Combined  (HC + Alg 合并)
  2. HC only   (Hand-Crafted)
  3. Alg only  (Algorithm-Generated)

Usage:
  python3 bertopic_clustering.py                        # 默认用 standardized 字段
  python3 bertopic_clustering.py --field mistake_reason  # 使用原始标注字段
  python3 bertopic_clustering.py --min_cluster_size 3   # 调整最小簇大小
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # 无显示器环境
from pathlib import Path

# ── BERTopic 依赖链 ────────────────────────────────────────────────────────────
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


# ── 常量 ──────────────────────────────────────────────────────────────────────
DATA_PATH  = Path(__file__).parent / "extracted_reasons_all.csv"
OUT_DIR    = Path(__file__).parent / "bertopic_results"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 轻量级，适合短句语义编码


def make_bertopic_model(texts: list[str], min_cluster_size: int = 3) -> BERTopic:
    """
    构建针对小数据集优化的 BERTopic 模型。

    参数调整说明：
    - UMAP n_neighbors=5：小数据集（< 200）用小邻域，防止全局结构坍塌
    - UMAP n_components=5：降维到 5D 后再聚类（比直接在原始 384D 上聚类稳定）
    - HDBSCAN min_cluster_size：由外部传入，建议 2-4
    - HDBSCAN min_samples=1：最宽松的噪声判定，减少 -1 噪声类数量
    - CountVectorizer ngram_range=(1,2)：允许双词组合关键词（更具可读性）
    """
    n = len(texts)
    # 动态调整 n_neighbors（不超过样本数的 20%，最小 3）
    n_neighbors = max(3, min(10, n // 5))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    topic_model = BERTopic(
        embedding_model=EMBED_MODEL,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics="auto",     # 自动合并相似主题，降低碎片化
        calculate_probabilities=True,
        verbose=False,
    )
    return topic_model


def run_bertopic(
    texts: list[str],
    labels: list[str],          # 对应每条文本的原始 split 标签（用于输出分析）
    dataset_name: str,
    min_cluster_size: int,
    out_dir: Path,
    field_name: str,
):
    """
    对 texts 运行 BERTopic，保存结果到 out_dir/{dataset_name}_*.

    输出文件：
    - {dataset_name}_topics.csv       每条文本的主题分配
    - {dataset_name}_topic_info.txt   每个主题的关键词 + 样本数量
    - {dataset_name}_umap2d.png       2D UMAP 散点图（可视化）
    """
    print(f"\n{'='*60}")
    print(f"  Running BERTopic on: {dataset_name}  (n={len(texts)}, field={field_name})")
    print(f"  min_cluster_size={min_cluster_size}")
    print(f"{'='*60}")

    if len(texts) < 5:
        print(f"  [SKIP] Too few samples ({len(texts)}) for BERTopic.")
        return

    model = make_bertopic_model(texts, min_cluster_size)
    topics, probs = model.fit_transform(texts)

    # ── 结果汇总 ──────────────────────────────────────────────────────────────
    topic_info = model.get_topic_info()
    n_topics   = len(topic_info[topic_info["Topic"] != -1])
    n_noise    = sum(1 for t in topics if t == -1)

    print(f"\n  → Found {n_topics} topics  |  {n_noise} noise points (-1)")
    print(f"\n  Topic keywords:")
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            label = "[NOISE]"
        else:
            label = f"Topic {tid:2d}"
        words = row["Name"].replace(f"{tid}_", "").replace("_", " ")
        print(f"    {label} (n={row['Count']:3d}):  {words}")

    # ── 保存 per-sample 结果 ──────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "text":    texts,
        "split":   labels,
        "topic":   topics,
        "prob":    [max(p) if hasattr(p, "__len__") else p for p in probs],
    })
    # 拼接主题关键词列
    topic_name_map = {
        row["Topic"]: row["Name"].replace(f"{row['Topic']}_", "").replace("_", " ")
        for _, row in topic_info.iterrows()
    }
    out_df["topic_keywords"] = out_df["topic"].map(topic_name_map)
    topics_csv = out_dir / f"{dataset_name}_topics.csv"
    out_df.to_csv(topics_csv, index=False)
    print(f"\n  → Saved per-sample results: {topics_csv}")

    # ── 保存人类可读的 topic_info ──────────────────────────────────────────────
    info_path = out_dir / f"{dataset_name}_topic_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"BERTopic Results: {dataset_name}\n")
        f.write(f"Field: {field_name}  |  n={len(texts)}  |  min_cluster_size={min_cluster_size}\n")
        f.write(f"Topics found: {n_topics}  |  Noise points: {n_noise}\n")
        f.write("=" * 60 + "\n\n")

        for tid in sorted(set(topics)):
            subset = out_df[out_df["topic"] == tid]
            kw = topic_name_map.get(tid, "")
            if tid == -1:
                f.write(f"[NOISE] (n={len(subset)})\n")
            else:
                f.write(f"Topic {tid}  (n={len(subset)})  Keywords: {kw}\n")

            # 按 split 分类统计
            split_counts = subset["split"].value_counts().to_dict()
            f.write(f"  Split distribution: {split_counts}\n")
            f.write("  Samples:\n")
            for t in subset["text"].tolist():
                f.write(f"    - {t}\n")
            f.write("\n")

    print(f"  → Saved topic report:        {info_path}")

    # ── 2D UMAP 可视化 ─────────────────────────────────────────────────────────
    try:
        umap_2d = UMAP(
            n_neighbors=max(3, min(10, len(texts) // 5)),
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        embedder = SentenceTransformer(EMBED_MODEL)
        embeds   = embedder.encode(texts, show_progress_bar=False)
        coords   = umap_2d.fit_transform(embeds)

        fig, ax = plt.subplots(figsize=(9, 6))
        unique_topics = sorted(set(topics))
        cmap = plt.cm.get_cmap("tab10", max(len(unique_topics), 1))

        for i, tid in enumerate(unique_topics):
            mask = [t == tid for t in topics]
            xs   = coords[mask, 0]
            ys   = coords[mask, 1]
            lbl  = f"Noise (-1)" if tid == -1 else f"Topic {tid}: {topic_name_map.get(tid, '')[:30]}"
            col  = "lightgray" if tid == -1 else cmap(i)
            ax.scatter(xs, ys, label=lbl, color=col, alpha=0.75, s=60, edgecolors="k", linewidths=0.3)

        ax.set_title(f"BERTopic Clusters — {dataset_name}\n(field={field_name}, n={len(texts)})", fontsize=12)
        ax.legend(loc="best", fontsize=7, framealpha=0.7)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        plt.tight_layout()
        fig_path = out_dir / f"{dataset_name}_umap2d.png"
        plt.savefig(fig_path, dpi=300)
        plt.savefig(out_dir / f"{dataset_name}_umap2d.svg", bbox_inches="tight")
        plt.savefig(out_dir / f"{dataset_name}_umap2d.pdf", bbox_inches="tight")
        plt.close()
        print(f"  → Saved UMAP scatter:        {fig_path}")
    except Exception as e:
        print(f"  [WARN] Visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="BERTopic clustering for mistake_reason")
    parser.add_argument(
        "--field",
        choices=["standardized", "mistake_reason"],
        default="standardized",
        help="Text field to cluster. Default: standardized",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=3,
        help="HDBSCAN min_cluster_size. Lower = more fine-grained clusters. Default: 3",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DATA_PATH),
        help="Path to CSV file (default: extracted_reasons_all.csv)",
    )
    args = parser.parse_args()

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.data_path)
    print(f"\nLoaded {len(df)} rows from {args.data_path}")
    print(f"Using field: '{args.field}'")
    print(f"Split distribution:\n{df['split'].value_counts().to_string()}")

    # 去掉空值
    df = df.dropna(subset=[args.field]).reset_index(drop=True)
    print(f"After dropping NaN: {len(df)} rows")

    out_dir = Path(__file__).parent / "bertopic_results"
    out_dir.mkdir(exist_ok=True)

    # ── 三次聚类 ──────────────────────────────────────────────────────────────

    # 1. Combined (HC + Alg)
    run_bertopic(
        texts=df[args.field].tolist(),
        labels=df["split"].tolist(),
        dataset_name="combined",
        min_cluster_size=args.min_cluster_size,
        out_dir=out_dir,
        field_name=args.field,
    )

    # 2. Hand-Crafted only
    hc_df = df[df["split"] == "Hand-Crafted"].reset_index(drop=True)
    run_bertopic(
        texts=hc_df[args.field].tolist(),
        labels=hc_df["split"].tolist(),
        dataset_name="hc_only",
        min_cluster_size=max(2, args.min_cluster_size - 1),   # HC 数据量少，略放宽
        out_dir=out_dir,
        field_name=args.field,
    )

    # 3. Algorithm-Generated only
    alg_df = df[df["split"] == "Algorithm-Generated"].reset_index(drop=True)
    run_bertopic(
        texts=alg_df[args.field].tolist(),
        labels=alg_df["split"].tolist(),
        dataset_name="alg_only",
        min_cluster_size=args.min_cluster_size,
        out_dir=out_dir,
        field_name=args.field,
    )

    print(f"\n{'='*60}")
    print(f"  All results saved to: {out_dir}/")
    print(f"  Files per run:")
    print(f"    *_topics.csv      — per-sample topic assignment")
    print(f"    *_topic_info.txt  — readable topic report with samples")
    print(f"    *_umap2d.png      — 2D scatter visualization")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
