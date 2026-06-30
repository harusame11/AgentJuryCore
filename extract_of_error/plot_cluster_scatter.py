"""
plot_cluster_scatter.py
=======================
BGE-M3 嵌入 + KMeans(K=3) 语义聚类，UMAP 2D 降维可视化。
聚类在原始高维空间完成，2D 坐标仅用于展示。

输出格式：SVG（矢量，文字可编辑）+ EMF（via Inkscape）+ PNG（300dpi备份）
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from openai import OpenAI
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ── 字体设置（中文支持）─────────────────────────────────────────────────────────
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei',
                                'DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10

# ── 配置 ─────────────────────────────────────────────────────────────────────
SILICON_API_KEY  = os.environ.get("SILICON_API_KEY", "")
SILICON_BASE_URL = os.environ.get("SILICON_BASE_URL", "https://api.siliconflow.cn/v1")
EMBED_MODEL      = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "Who&When"
ALG_DIR = DEFAULT_DATA_ROOT / "Algorithm-Generated"
HC_DIR  = DEFAULT_DATA_ROOT / "Hand-Crafted"
HERE    = Path(__file__).parent

# 聚类标签（中英双语）
ALG_CLUSTER_LABELS = {
    0: "事实幻觉与推理错误\n(Factual Hallucination\n& Reasoning Error)",
    1: "实现缺陷与专家输出错误\n(Implementation Bug\n& Expert Output Error)",
    2: "检索遗漏与验证缺失\n(Retrieval & Verification\nOmission)",
}

HC_CLUSTER_LABELS = {
    0: "基于不完整证据的过早决策\n(Premature Decision on\nIncomplete Evidence)",
    1: "网络检索与内容提取失败\n(Web Retrieval &\nContent Extraction Failure)",
    2: "任务规划与指令错误\n(Task Planning &\nDirective Error)",
}

# 彩色方案（兼顾彩色印刷，同时用不同标记保证黑白可区分）
CLUSTER_COLORS  = ["#1a6faf", "#1a9641", "#b03a2e"]
CLUSTER_MARKERS = ["o", "s", "^"]


# ── EMF 导出 ──────────────────────────────────────────────────────────────────
def export_emf(svg_path: str):
    emf_path = svg_path.replace(".svg", ".emf")
    try:
        result = subprocess.run(
            ["inkscape", "--export-filename", emf_path, svg_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print(f"  [EMF] Saved → {os.path.basename(emf_path)}")
        else:
            print(f"  [EMF] Inkscape error: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  [EMF] Inkscape not found. Install via: sudo apt install inkscape")
    except Exception as e:
        print(f"  [EMF] Failed: {e}")


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_reasons(folder: Path, split_name: str) -> list[dict]:
    samples = []
    files = sorted(folder.glob("*.json"),
                   key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            reason = d.get("mistake_reason", "").strip()
            if reason:
                samples.append({"file": fpath.name, "split": split_name, "reason": reason})
        except Exception:
            pass
    return samples


# ── BGE-M3 嵌入（带 .npy 缓存）────────────────────────────────────────────────
def embed_bge_m3(client: OpenAI, texts: list[str],
                 cache_path: Path, batch_size: int = 32) -> np.ndarray:
    if cache_path.exists():
        print(f"  [缓存命中] {cache_path.name}")
        return np.load(cache_path)

    print(f"  调用 BGE-M3 API，共 {len(texts)} 条文本...")
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BGE-M3"):
        batch = texts[i: i + batch_size]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            vecs = [item.embedding for item in resp.data]
        except Exception as e:
            print(f"  批次 {i//batch_size} 失败: {e}，使用零向量代替")
            vecs = [[0.0] * 1024] * len(batch)
        all_vecs.extend(vecs)

    emb = np.array(all_vecs, dtype=np.float32)
    np.save(cache_path, emb)
    print(f"  已缓存 → {cache_path.name}")
    return emb


# ── 散点图绘制 ────────────────────────────────────────────────────────────────
def plot_scatter(embeddings: np.ndarray, labels: np.ndarray,
                 title_zh: str, title_en: str,
                 out_path: Path, cluster_name_map: dict,
                 colors: list, markers: list,
                 seed: int = 42, label_offsets: dict = None):
    """
    UMAP 2D 降维可视化。
    - 不同簇使用不同颜色 + 不同标记，兼顾黑白打印
    - 重心用星形标注
    - 文字保持可编辑（SVG/EMF 格式）
    """
    n = len(embeddings)
    n_neighbors = max(5, min(30, n // 4))
    umap2d = UMAP(n_components=2, n_neighbors=n_neighbors,
                  min_dist=0.0, spread=1.0, metric="cosine",
                  random_state=seed)
    coords = umap2d.fit_transform(embeddings)

    k = len(cluster_name_map)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # 各簇散点（不同颜色 + 不同标记）
    for cid in range(k):
        mask = labels == cid
        n_cid = mask.sum()
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[cid], marker=markers[cid],
            alpha=0.72, s=45,
            edgecolors="white", linewidths=0.5,
            label=f"簇{cid}  Cluster {cid} (n={n_cid})",
            zorder=3,
        )

    # 重心星标 + 文字标注
    for cid in range(k):
        mask = labels == cid
        if mask.sum() == 0:
            continue
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.scatter(cx, cy, c=colors[cid], s=180, marker="*",
                   edgecolors="black", linewidths=0.8, zorder=5)
        off = label_offsets[cid] if (label_offsets and cid in label_offsets) else (8, 6)
        ax.annotate(
            cluster_name_map[cid],
            (cx, cy),
            textcoords="offset points", xytext=off,
            fontsize=8.5, fontweight="bold", color=colors[cid],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      alpha=0.85, ec=colors[cid], linewidth=1.2),
        )

    # 坐标轴范围
    pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.10
    pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.10
    ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

    # 标签（中英双语）
    ax.set_xlabel("UMAP 第一维度  (UMAP Dimension 1)", fontsize=10.5)
    ax.set_ylabel("UMAP 第二维度  (UMAP Dimension 2)", fontsize=10.5)
    ax.set_title(f"{title_zh}\n{title_en}",
                 fontsize=10.5, fontweight="bold", pad=8)

    # 图例
    legend_patches = [
        mpatches.Patch(color=colors[cid],
                       label=f"簇{cid}  {cluster_name_map[cid].split(chr(10))[0]}")
        for cid in range(k)
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, loc="best",
              framealpha=0.88, edgecolor="#cccccc")

    ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#333333")
    fig.tight_layout()

    # 保存
    svg_path = str(out_path).replace(".pdf", ".svg")
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    fig.savefig(str(out_path), bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"[SVG] Saved → {os.path.basename(svg_path)}")
    export_emf(svg_path)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot K=3 failure-cluster scatter figures.")
    parser.add_argument("--alg_dir", default=str(ALG_DIR), help="Algorithm-Generated JSON directory.")
    parser.add_argument("--hc_dir", default=str(HC_DIR), help="Hand-Crafted JSON directory.")
    parser.add_argument("--api_key", default=SILICON_API_KEY, help="OpenAI-compatible API key. Defaults to SILICON_API_KEY.")
    parser.add_argument("--base_url", default=SILICON_BASE_URL, help="OpenAI-compatible API base URL.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set SILICON_API_KEY or pass --api_key.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    alg_samples = load_reasons(Path(args.alg_dir), "Algorithm-Generated")
    hc_samples  = load_reasons(Path(args.hc_dir),  "Hand-Crafted")
    all_samples = alg_samples + hc_samples
    print(f"已加载: ALG={len(alg_samples)}, HC={len(hc_samples)}, 合计={len(all_samples)}")

    print("\n[1/3] 算法生成子集嵌入:")
    alg_emb = embed_bge_m3(client, [s["reason"] for s in alg_samples],
                           HERE / "emb_cache_alg.npy")
    print("\n[2/3] 人工设计子集嵌入:")
    hc_emb  = embed_bge_m3(client, [s["reason"] for s in hc_samples],
                           HERE / "emb_cache_hc.npy")
    print("\n[3/3] 全集嵌入:")
    all_emb = embed_bge_m3(client, [s["reason"] for s in all_samples],
                           HERE / "emb_cache_all.npy")

    # 各子集独立 KMeans
    km_alg = KMeans(n_clusters=3, random_state=42, n_init=10).fit(alg_emb)
    km_hc  = KMeans(n_clusters=3, random_state=42, n_init=10).fit(hc_emb)
    km_all = KMeans(n_clusters=3, random_state=42, n_init=10).fit(all_emb)

    print("\n[绘图]")

    plot_scatter(
        alg_emb, km_alg.labels_,
        title_zh="故障模式聚类（算法生成子集，K=3）",
        title_en="K=3 Failure Clusters — Algorithm-Generated (BGE-M3 + UMAP 2D)",
        out_path=HERE / "cluster_scatter_alg.pdf",
        cluster_name_map=ALG_CLUSTER_LABELS,
        colors=CLUSTER_COLORS,
        markers=CLUSTER_MARKERS,
    )

    plot_scatter(
        hc_emb, km_hc.labels_,
        title_zh="故障模式聚类（人工设计子集，K=3）",
        title_en="K=3 Failure Clusters — Hand-Crafted (BGE-M3 + UMAP 2D)",
        out_path=HERE / "cluster_scatter_hc.pdf",
        cluster_name_map=HC_CLUSTER_LABELS,
        colors=CLUSTER_COLORS,
        markers=CLUSTER_MARKERS,
        label_offsets={0: (8, -22), 1: (8, 10), 2: (8, 6)},
    )

    plot_scatter(
        all_emb, km_all.labels_,
        title_zh="故障模式聚类（完整数据集，K=3）",
        title_en="K=3 Failure Clusters — Full Who&When Dataset (BGE-M3 + UMAP 2D)",
        out_path=HERE / "cluster_scatter_all.pdf",
        cluster_name_map=ALG_CLUSTER_LABELS,
        colors=CLUSTER_COLORS,
        markers=CLUSTER_MARKERS,
    )

    print("\n[完成] 所有图已保存（SVG / EMF / PDF / PNG）")


if __name__ == "__main__":
    main()
