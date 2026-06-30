"""
sample_ablation_subset.py
=========================
从 Algorithm-Generated 数据集中以固定随机种子抽取消融子集，
并将所选文件复制到新目录，供所有消融实验共享同一子集。

用法：
    python3 sample_ablation_subset.py
        --src   "../Who&When/Algorithm-Generated"
        --dst   "../Who&When/ablation_subset"
        --n     40
        --seed  42

输出：
    ../Who&When/ablation_subset/  (含 N 个 JSON 文件)
    ablation_subset_manifest.txt  (记录所选文件列表，便于复现)
"""

import os
import random
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",  default="../Who&When/Algorithm-Generated")
    parser.add_argument("--dst",  default="../Who&When/ablation_subset")
    parser.add_argument("--n",    type=int, default=40, help="子集大小（默认40）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认42）")
    args = parser.parse_args()

    # ── 收集所有 JSON 文件并按文件名数字排序 ─────────────────────────────────
    all_files = sorted(
        [f for f in os.listdir(args.src) if f.endswith(".json")],
        key=lambda x: int(x.replace(".json", ""))
    )
    total = len(all_files)
    print(f"源目录: {args.src}")
    print(f"总文件数: {total}")

    if args.n > total:
        print(f"警告: 请求 {args.n} 个样本，但总共只有 {total} 个，将使用全部。")
        args.n = total

    # ── 固定种子采样 ──────────────────────────────────────────────────────────
    random.seed(args.seed)
    selected = sorted(random.sample(all_files, args.n),
                      key=lambda x: int(x.replace(".json", "")))

    print(f"\n随机种子: {args.seed}")
    print(f"采样数量: {args.n} / {total} ({args.n/total*100:.1f}%)")
    print(f"选中文件: {[f.replace('.json','') for f in selected]}")

    # ── 复制到目标目录 ────────────────────────────────────────────────────────
    os.makedirs(args.dst, exist_ok=True)
    for f in selected:
        shutil.copy2(os.path.join(args.src, f), os.path.join(args.dst, f))
    print(f"\n文件已复制到: {args.dst}")

    # ── 写清单文件（便于他人复现）────────────────────────────────────────────
    manifest_path = os.path.join(os.path.dirname(args.dst), "ablation_subset_manifest.txt")
    with open(manifest_path, "w") as mf:
        mf.write(f"seed={args.seed}\n")
        mf.write(f"n={args.n}\n")
        mf.write(f"src={args.src}\n")
        mf.write(f"files=\n")
        for f in selected:
            mf.write(f"  {f}\n")
    print(f"清单写入: {manifest_path}")


if __name__ == "__main__":
    main()
