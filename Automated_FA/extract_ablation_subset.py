"""
extract_ablation_subset.py
==========================
从完整实验日志（_merged_alg_enhance_1_126.txt）中，
提取消融子集（alg_ablation_subset）对应的 case 块，
输出为独立日志文件，可直接送入 evaluate_v3.py 评估。

Usage:
    python extract_ablation_subset.py \
        --log_file outputs/_merged_alg_enhance_1_126.txt \
        --subset_dir ../Who\&When/alg_ablation_subset \
        --output_file outputs/ablation_full_method_baseline.txt
"""

import re
import os
import argparse


def extract_ablation_subset(log_file: str, subset_dir: str, output_file: str):
    # 1. 读取消融子集 case 文件名
    subset_cases = set(
        f for f in os.listdir(subset_dir) if f.endswith(".json")
    )
    print(f"Ablation subset: {len(subset_cases)} cases")

    # 2. 读取完整日志
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 3. 按 case 块切分（兼容所有标签格式）
    pattern = r"(--- (?:\[\w+(?:-\w+)*\] )?Analyzing(?:\s+File)?: )"
    parts = re.split(pattern, content)
    # parts: [前导文本, 分隔符, case块, 分隔符, case块, ...]

    # 4. 重组为 {filename: full_block} 字典
    blocks = {}
    i = 1
    while i < len(parts) - 1:
        header = parts[i]       # e.g. "--- [ALG-Enhance] Analyzing: "
        body   = parts[i + 1]   # e.g. "1.json ---\n...(内容)..."
        # 提取文件名
        fname_match = re.match(r"([\w\-\.]+\.json)", body)
        if fname_match:
            fname = fname_match.group(1)
            blocks[fname] = header + body
        i += 2

    print(f"Total blocks found in log: {len(blocks)}")

    # 5. 筛选消融子集
    matched = {k: v for k, v in blocks.items() if k in subset_cases}
    missing = subset_cases - set(matched.keys())

    print(f"Matched: {len(matched)} / {len(subset_cases)}")
    if missing:
        print(f"Missing cases (not found in log): {sorted(missing, key=lambda x: int(x.split('.')[0]))}")

    # 6. 按 case 序号排序输出
    sorted_blocks = sorted(
        matched.items(),
        key=lambda x: int(x[0].split(".")[0])
    )

    # 7. 写出结果
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for fname, block in sorted_blocks:
            f.write(block)

    print(f"\nOutput written to: {output_file}")
    print(f"Cases included: {[x[0] for x in sorted_blocks]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ablation subset cases from a merged log file."
    )
    parser.add_argument("--log_file",    type=str, required=True,
                        help="Path to the full merged log (e.g. outputs/_merged_alg_enhance_1_126.txt)")
    parser.add_argument("--subset_dir",  type=str, required=True,
                        help="Path to the ablation subset dir (e.g. ../Who&When/alg_ablation_subset)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output path for the extracted log")
    args = parser.parse_args()

    extract_ablation_subset(args.log_file, args.subset_dir, args.output_file)
