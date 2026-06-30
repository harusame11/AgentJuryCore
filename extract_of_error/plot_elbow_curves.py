"""
plot_elbow_curves.py
====================
归一化肘部曲线（算法生成 / 人工设计）。
每条曲线以 K=2 的惯量为基准归一化，使两条线均从 1.0 出发，
便于跨数据集比较斜率变化。

输出格式：SVG（矢量，文字可编辑）+ EMF（via Inkscape）+ PNG（300dpi备份）
"""
import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 字体设置（中文支持）─────────────────────────────────────────────────────────
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei',
                                'DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10


# ── EMF 导出（via Inkscape）───────────────────────────────────────────────────
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


# ── 加载缓存嵌入 ──────────────────────────────────────────────────────────────
def load_emb(name):
    path = os.path.join(HERE, f"emb_cache_{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缓存不存在: {path}，请先运行 plot_cluster_scatter.py")
    return np.load(path)

print("加载缓存嵌入...")
emb_alg = load_emb("alg")
emb_hc  = load_emb("hc")

# ── 计算 K=2..5 的惯量 ────────────────────────────────────────────────────────
k_vals = [2, 3, 4, 5]

def compute_inertia(emb, label):
    inertias = []
    for k in k_vals:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(emb)
        inertias.append(km.inertia_)
        print(f"  [{label}] K={k}: inertia={km.inertia_:.2f}")
    return inertias

print("\n算法生成子集 (Algorithm-Generated):")
inertia_alg = compute_inertia(emb_alg, "ALG")
print("\n人工设计子集 (Hand-Crafted):")
inertia_hc  = compute_inertia(emb_hc,  "HC")

# ── K=2 归一化 ────────────────────────────────────────────────────────────────
def normalise(series):
    base = series[0]
    return [round(v / base, 4) for v in series]

norm_alg = normalise(inertia_alg)
norm_hc  = normalise(inertia_hc)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
# 黑白打印兼容：用不同线型 + 不同标记区分两条曲线
color_alg = "#1a6faf"   # 蓝色（彩色印刷）
color_hc  = "#b03a2e"   # 红色（彩色印刷）

fig, ax = plt.subplots(figsize=(7.5, 4.8))

ax.plot(k_vals, norm_alg,
        color=color_alg, marker="s", linewidth=2.0,
        markersize=7, linestyle="--",
        label="算法生成子集 (Algorithm-Generated, n=126)")
ax.plot(k_vals, norm_hc,
        color=color_hc, marker="^", linewidth=2.0,
        markersize=7, linestyle="-",
        label="人工设计子集 (Hand-Crafted, n=58)")

# K=3 参考线（标注放在竖线顶端，用 axes 坐标系固定 y=0.97，避免与数据点文字重叠）
ax.axvline(x=3, color="#555555", linestyle=":", linewidth=1.5, alpha=0.85)
ax.annotate("K = 3", xy=(3, 1), xycoords=("data", "axes fraction"),
            xytext=(5, -4), textcoords="offset points",
            fontsize=9, color="#555555", va="top")

# 标注各点相对K=2的下降百分比
def annotate_pct(xs, ys, color, offsets):
    for x, y, (ox, oy) in zip(xs, ys, offsets):
        pct = (1.0 - y) * 100
        lbl = "0%" if pct < 0.01 else f"−{pct:.1f}%"
        ax.annotate(lbl, (x, y),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=8, color=color, fontweight="bold")

# ALG: K=3 点向左偏移，避免与"K = 3"文字重叠；其余向右上
annotate_pct(k_vals, norm_alg, color_alg,
             [(6, 7), (-52, 7), (6, 7), (6, 7)])
# HC: K=5 点向右上方偏移，防止跑出图底边界
annotate_pct(k_vals, norm_hc,  color_hc,
             [(6, -16), (6, -16), (6, -16), (6, 7)])

# 坐标轴与标题（中英双语）
ax.set_xlabel("聚类数 K  (Number of Clusters K)", fontsize=11)
ax.set_ylabel("归一化惯量（相对于 K=2）\nNormalised Inertia (relative to K=2)", fontsize=10)
ax.set_title("肘部法则：失败原因聚类分析\n"
             "Elbow Method: Failure Reason Clustering",
             fontsize=11, fontweight="bold", pad=10)

ax.set_xticks(k_vals)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.legend(fontsize=9, loc="upper right", framealpha=0.90,
          edgecolor="#cccccc")
ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor("#333333")

fig.tight_layout()

# ── 保存 ──────────────────────────────────────────────────────────────────────
svg_path = os.path.join(HERE, "elbow_curves.svg")
fig.savefig(svg_path, bbox_inches="tight", format="svg")
fig.savefig(os.path.join(HERE, "elbow_curves.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "elbow_curves.png"), dpi=300, bbox_inches="tight")
print(f"\n[SVG] Saved → elbow_curves.svg")

export_emf(svg_path)
print("[DONE] elbow_curves 全部格式已保存")
