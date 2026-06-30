"""
cluster_raw_reasons.py
======================
直接从 Who&When 原始 JSON 文件读取 mistake_reason，
用 SiliconFlow BGE-M3 嵌入后跑 KMeans(K=2,3,4,5)，
输出各 K 的 inertia 及聚类内容。

用法：
    python3 cluster_raw_reasons.py
"""

import os
import json
import csv
import argparse
from pathlib import Path
from openai import OpenAI
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np

# ── 配置 ─────────────────────────────────────────────────────────────────────
SILICON_API_KEY  = os.environ.get("SILICON_API_KEY", "")
SILICON_BASE_URL = os.environ.get("SILICON_BASE_URL", "https://api.siliconflow.cn/v1")
EMBED_MODEL      = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "Who&When"
ALG_DIR = DEFAULT_DATA_ROOT / "Algorithm-Generated"
HC_DIR  = DEFAULT_DATA_ROOT / "Hand-Crafted"
HERE    = Path(__file__).parent


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
        except Exception as e:
            print(f"  Skip {fpath.name}: {e}")
    return samples


# ── BGE-M3 嵌入 ───────────────────────────────────────────────────────────────
def embed_bge_m3(client: OpenAI, texts: list[str], batch_size: int = 32) -> np.ndarray:
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BGE-M3 embedding"):
        batch = texts[i: i + batch_size]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            vecs = [item.embedding for item in resp.data]
        except Exception as e:
            print(f"  Batch {i//batch_size} failed: {e}. Using zero vectors.")
            vecs = [[0.0] * 1024] * len(batch)
        all_vecs.extend(vecs)
    return np.array(all_vecs, dtype=np.float32)


# ── 聚类 + 输出 ───────────────────────────────────────────────────────────────
def cluster_and_report(samples: list[dict], embeddings: np.ndarray,
                       label: str, out_path: Path):
    lines = [f"=== Raw mistake_reason clustering: {label} ({len(samples)} samples) ===\n"]
    print(f"\n{'='*60}")
    print(f"Clustering: {label} ({len(samples)} samples)")
    print(f"{'='*60}")

    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        inertia_line = f"\nK={k} CLUSTERS  [KMeans inertia: {km.inertia_:.2f}]"
        print(inertia_line)
        lines.append(inertia_line)

        for cid in range(k):
            members = [samples[i] for i, l in enumerate(labels) if l == cid]
            block = f"\n--- Cluster {cid} ({len(members)} samples) ---"
            for m in members:
                block += f"\n  [{m['split']}] {m['reason']}"
            print(block)
            lines.append(block)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[DONE] Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cluster raw mistake_reason fields with BGE-M3 + KMeans.")
    parser.add_argument("--alg_dir", default=str(ALG_DIR), help="Algorithm-Generated JSON directory.")
    parser.add_argument("--hc_dir", default=str(HC_DIR), help="Hand-Crafted JSON directory.")
    parser.add_argument("--api_key", default=SILICON_API_KEY, help="OpenAI-compatible API key. Defaults to SILICON_API_KEY.")
    parser.add_argument("--base_url", default=SILICON_BASE_URL, help="OpenAI-compatible API base URL.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set SILICON_API_KEY or pass --api_key.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 加载
    alg_samples = load_reasons(Path(args.alg_dir), "Algorithm-Generated")
    hc_samples  = load_reasons(Path(args.hc_dir),  "Hand-Crafted")
    all_samples = alg_samples + hc_samples

    print(f"Loaded: ALG={len(alg_samples)}, HC={len(hc_samples)}, All={len(all_samples)}")

    # 嵌入（各自独立嵌入，避免混淆）
    print("\n[1/3] Embedding Algorithm-Generated...")
    alg_emb = embed_bge_m3(client, [s["reason"] for s in alg_samples])

    print("\n[2/3] Embedding Hand-Crafted...")
    hc_emb  = embed_bge_m3(client, [s["reason"] for s in hc_samples])

    print("\n[3/3] Embedding All...")
    all_emb = embed_bge_m3(client, [s["reason"] for s in all_samples])

    # 聚类 + 输出
    cluster_and_report(alg_samples, alg_emb, "Algorithm-Generated",
                       HERE / "cluster_raw_alg.txt")
    cluster_and_report(hc_samples,  hc_emb,  "Hand-Crafted",
                       HERE / "cluster_raw_hc.txt")
    cluster_and_report(all_samples, all_emb, "All (Who&When)",
                       HERE / "cluster_raw_all.txt")

    print("\n=== Inertia Summary ===")
    print("Check cluster_raw_alg.txt / cluster_raw_hc.txt / cluster_raw_all.txt")


if __name__ == "__main__":
    main()
