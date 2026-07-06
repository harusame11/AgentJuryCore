"""
extract_D.py
============
Pipeline for deriving DAO arbitration dimensions from Who&When dataset.

PHASE 1: LLM Feature Extraction
  - Read all mistake_reasons from both dataset splits
  - Use LLM to rewrite each into a standardized short phrase (<=15 words)
  - Save results to extracted_reasons.csv

PHASE 2: Embedding + Clustering
  - Embed standardized phrases using sentence-transformers
  - Run KMeans with k=2,3,4,5 to find natural clusters
  - Print cluster contents and centroids for dimension discovery

Usage:
  python extract_D.py --phase 1         # LLM extraction only
  python extract_D.py --phase 2         # Clustering only (needs phase 1 output)
  python extract_D.py --phase all       # Run both phases

API: Uses SiliconFlow OpenAI-compatible API (same as inference.py)
"""

import os
import json
import csv
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
DATA_ROOT = Path(os.environ.get("WHO_WHEN_ROOT", Path(__file__).resolve().parents[1] / "Who&When"))
DATASETS = {
    "Algorithm-Generated": DATA_ROOT / "Algorithm-Generated",
    "Hand-Crafted":        DATA_ROOT / "Hand-Crafted",
    #"mvp_test":            DATA_ROOT / "mvp_test_1_10example",
}
OUTPUT_CSV = Path(__file__).parent / "extracted_reasons_all.csv"
CLUSTER_OUTPUT = Path(__file__).parent / "cluster_results.txt"

# SiliconFlow API (OpenAI-compatible) — same config as inference.py
SILICON_API_KEY  = os.environ.get("SILICON_API_KEY", "")
SILICON_BASE_URL = os.environ.get("SILICON_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL_NAME       = os.environ.get("SILICON_MODEL", "Qwen/Qwen3.5-35B-A3B")


# ── PHASE 1: LLM Feature Extraction ─────────────────────────────────────────

def build_client():
    if not SILICON_API_KEY:
        raise ValueError("Missing API key. Set SILICON_API_KEY or pass --api_key.")
    return OpenAI(
        api_key=SILICON_API_KEY,
        base_url=SILICON_BASE_URL,
    )


def standardize_reason(client: OpenAI, mistake_reason: str, task_desc: str) -> str:
    """
    Use LLM to rewrite a mistake_reason into a standardized short phrase.
    The phrase should be a noun phrase describing the ERROR TYPE, not the
    specific instance.
    """
    prompt = (
        "You are an expert in multi-agent system failure analysis.\n"
        "Your task: Given a description of why an agent failed, "
        "rewrite it as a concise, GENERALIZED error type phrase (10-15 words max).\n"
        "IMPORTANT: Output must be ABSTRACT and task-agnostic. Do NOT mention specific "
        "data, task names. Focus on the CATEGORY of error.\n"
        f"{task_desc}\n\n"
        f"Original reason: {mistake_reason}\n\n"
        "Standardized error phrase (output ONLY the phrase, nothing else):"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You output concise error type phrases only."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=50,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error: {e}")
        return mistake_reason  # fallback: truncate original


def load_all_samples(datasets: dict) -> list[dict]:
    """Load all JSON files and extract relevant fields."""
    samples = []
    for split_name, folder in datasets.items():
        if not folder.exists():
            print(f"Warning: {folder} not found, skipping.")
            continue
        json_files = sorted(folder.glob("*.json"), key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
        for fpath in json_files:
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                reason = data.get("mistake_reason", "")
                if not reason:
                    continue
                samples.append({
                    "file":          fpath.name,
                    "split":         split_name,
                    "question_id":   data.get("question_ID", ""),
                    "question_desc": data.get("question", ""),
                    "mistake_agent": data.get("mistake_agent", ""),
                    "mistake_step":  data.get("mistake_step", ""),
                    "mistake_reason": reason,
                    "standardized":  "",   # filled in Phase 1
                })
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    return samples


# ── PHASE 2: Embedding + Clustering ─────────────────────────────────────────

def phase2_cluster():
    """Phase 2: Read CSV, embed, cluster, print dimension analysis."""
    print("=" * 60)
    print("PHASE 2: Embedding + Clustering")
    print("=" * 60)

    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Missing clustering dependencies. Install the optional analysis dependencies "
            "listed in requirements.txt before running phase 2."
        ) from exc

    # Load CSV
    if not OUTPUT_CSV.exists():
        print(f"Error: {OUTPUT_CSV} not found. Run Phase 1 first.")
        return

    samples = []
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["standardized"]:
                samples.append(row)

    print(f"Loaded {len(samples)} standardized reasons for clustering.\n")

    texts = [s["standardized"] for s in samples]

    # Embed
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding phrases...")
    embeddings = model.encode(texts, show_progress_bar=True)

    output_lines = []

    # Try K = 2, 3, 4, 5
    for k in [2, 3, 4, 5]:
        print(f"\n{'─'*50}")
        print(f"K = {k} clusters")
        print(f"{'─'*50}")

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        header = f"\n{'='*60}\nK={k} CLUSTERS\n{'='*60}"
        print(header)
        output_lines.append(header)

        for cluster_id in range(k):
            members = [samples[i] for i, l in enumerate(labels) if l == cluster_id]
            phrases = [m["standardized"] for m in members]

            block = (
                f"\n--- Cluster {cluster_id} ({len(members)} samples) ---\n"
                + "\n".join(f"  [{m['split']}] {p}" for m, p in zip(members, phrases))
            )
            print(block)
            output_lines.append(block)

        # Intra-cluster similarity (cohesion)
        inertia_line = f"\n  [KMeans inertia: {km.inertia_:.2f}]"
        print(inertia_line)
        output_lines.append(inertia_line)

    # Save cluster analysis
    with open(CLUSTER_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n\nPhase 2 complete. Full analysis saved to: {CLUSTER_OUTPUT}")
    print("\n>>> Compare clusters across K values to choose how many dimensions to use.")
    print(">>> Look for K where clusters are semantically coherent and distinct.")


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    global SILICON_API_KEY, SILICON_BASE_URL, MODEL_NAME

    parser = argparse.ArgumentParser(description="Extract DAO dimensions from Who&When dataset.")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all",
                        help="Which phase to run: 1=LLM extraction, 2=clustering, all=both")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the dataset directory containing JSON files. If provided, it overrides default DATASETS.")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                        help="Root directory containing Algorithm-Generated and Hand-Crafted splits.")
    parser.add_argument("--api_key", default=SILICON_API_KEY,
                        help="OpenAI-compatible API key. Defaults to SILICON_API_KEY.")
    parser.add_argument("--base_url", default=SILICON_BASE_URL,
                        help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="OpenAI-compatible chat model name.")
    args = parser.parse_args()

    SILICON_API_KEY = args.api_key
    SILICON_BASE_URL = args.base_url
    MODEL_NAME = args.model

    # Determine which datasets to use
    if args.data_path:
        datasets_to_use = {"CommandLine": Path(args.data_path)}
    else:
        data_root = Path(args.data_root)
        datasets_to_use = DATASETS
        datasets_to_use = {
            "Algorithm-Generated": data_root / "Algorithm-Generated",
            "Hand-Crafted": data_root / "Hand-Crafted",
        }

    if args.phase in ("1", "all"):
        client = build_client()
        # Pass the selected datasets to the extraction function
        print(f"Using data path: {list(datasets_to_use.values())}")
        samples = load_all_samples(datasets_to_use)
        
        print("=" * 60)
        print("PHASE 1: LLM Feature Extraction")
        print("=" * 60)
        print(f"Loaded {len(samples)} samples with mistake_reason.\n")

        # Resume from existing CSV if partially done
        existing = {}
        if OUTPUT_CSV.exists():
            with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing[row["file"] + "_" + row["split"]] = row["standardized"]
            print(f"Resuming: {len(existing)} already standardized.\n")

        results = []
        for s in tqdm(samples, desc="Standardizing reasons"):
            key = s["file"] + "_" + s["split"]
            if key in existing:
                s["standardized"] = existing[key]
            else:
                s["standardized"] = standardize_reason(
                    client, 
                    s["mistake_reason"], 
                    s["question_desc"]
                )
            results.append(s)

        # Save to CSV
        fieldnames = ["file", "split", "question_id", "question_desc", "mistake_agent",
                      "mistake_step", "mistake_reason", "standardized"]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nPhase 1 complete. Results saved to: {OUTPUT_CSV}")

    if args.phase in ("2", "all"):
        phase2_cluster()


if __name__ == "__main__":
    main()
