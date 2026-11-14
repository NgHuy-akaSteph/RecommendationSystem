"""Sinh hình cho chương kết quả từ các bảng trong results/final/.

Dùng script thay vì notebook (kế hoạch phase 4 dự kiến `02-results-figures.ipynb`): phase 1
đã sinh hình bằng `run_exploratory_data_analysis.py` theo đúng cách này, và một script chạy
lại được bằng một lệnh thì dễ tái lập hơn notebook có trạng thái ẩn. Hình ghi ra
results/figures/ để chương báo cáo nhúng trực tiếp.

Chạy: python -m src.eval.make_result_figures
"""

import matplotlib

matplotlib.use("Agg")  # không cần cửa sổ đồ hoạ, chỉ ghi file PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import paths  # noqa: E402

FIGURES_DIR = paths.RESULTS_DIR / "figures"
FINAL_DIR = paths.RESULTS_DIR / "final"
# Lọc cộng tác tô một màu, nội dung màu khác, baseline xám — người đọc thấy ngay nhóm nào
COLORS = {"CF": "#c0504d", "CB": "#4f81bd", "Baseline": "#999999"}


def _family(name: str) -> str:
    """Nhóm thuật toán để tô màu. Nổ với tên lạ thay vì mặc định về CB.

    `.map(names)` sinh NaN khi một thí nghiệm có trong bảng này mà thiếu ở bảng kia; mặc
    định về CB sẽ tô một thuật toán CF thành màu xanh mà không ai nhận ra.

    Nhận cả **nhãn hiển thị** ("Item-CF (RS)"), **khoá registry** ("rs-itemcf") lẫn
    **tên thí nghiệm** ("full_cb1-00-genres-binary-profile"): bảng rút gọn của
    `build_final_tables` chỉ chứa 8 mô hình benchmark, còn bảng bucket theo user có
    mọi thí nghiệp `full_` nên tên không tra được sẽ rơi về dạng tên thí nghiệm.
    """
    if not isinstance(name, str):
        raise ValueError(f"tên thuật toán không hợp lệ: {name!r} — thiếu trong full-summary.csv?")
    if name in ("Random", "MostPop", "random", "mostpop", "Random (sanity)"):
        return "Baseline"
    if name in ("ItemKNN-CF", "ALS", "Item-CF (RS)", "BPR (RS)",
                "cf-itemknn", "cf-als", "rs-itemcf", "rs-bpr"):
        return "CF"
    if name.startswith("CB-") or name.startswith("cb"):
        return "CB"
    # Tên thí nghiệm: bỏ tiền tố giao thức rồi đoán theo tiền tố họ mô hình
    bare = name
    for prefix in ("coldstart_", "full_"):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
    if bare.startswith("cb"):
        return "CB"
    if bare.startswith(("cf-", "rs-")):
        return "CF"
    if bare.startswith("baseline"):
        return "Baseline"
    raise ValueError(f"không rõ nhóm của thuật toán '{name}'; bổ sung vào _family()")


def _load(name: str) -> pd.DataFrame | None:
    path = FINAL_DIR / name
    return pd.read_csv(path) if path.exists() else None


def _bar(ax, df: pd.DataFrame, metric: str, title: str) -> None:
    df = df.sort_values(metric)
    colors = [COLORS[_family(n)] for n in df["Thuật toán"]]
    ax.barh(df["Thuật toán"], df[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(title)
    for y, v in enumerate(df[metric]):
        ax.text(v, y, f" {v:.4f}", va="center", fontsize=8)


def figure_accuracy(full: pd.DataFrame, cold: pd.DataFrame | None) -> None:
    """Hình chính: cùng thuật toán, hai giao thức, thứ hạng đảo ngược."""
    n = 2 if cold is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4.5))
    axes = np.atleast_1d(axes)
    _bar(axes[0], full, "NDCG@10", "Toàn bộ người dùng (158.774)")
    if cold is not None:
        _bar(axes[1], cold, "NDCG@10", "Cold-start item (10% phim mới)")
    fig.suptitle("NDCG@10 theo thuật toán — xanh: nội dung, đỏ: cộng tác, xám: baseline")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "results-accuracy.png", dpi=150)
    plt.close(fig)


def figure_metric_vs_k(full: pd.DataFrame) -> None:
    ks = [5, 10, 20, 50]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, ax in zip(("NDCG", "R"), axes):
        for _, row in full.iterrows():
            ax.plot(ks, [row[f"{metric}@{k}"] for k in ks], marker="o",
                    color=COLORS[_family(row["Thuật toán"])], alpha=0.85,
                    label=row["Thuật toán"])
        ax.set_xlabel("k")
        ax.set_ylabel(f"{metric}@k")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.set_title(f"{metric}@k theo độ dài danh sách")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "results-metric-vs-k.png", dpi=150)
    plt.close(fig)


def figure_user_buckets(buckets: pd.DataFrame, names: dict[str, str]) -> None:
    labels = ["5–30", "31–100", ">100"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    for _, row in buckets.iterrows():
        label = names.get(row["exp_name"], row["exp_name"])
        ax.plot(x, [row[lb] for lb in labels], marker="o",
                color=COLORS[_family(label)], alpha=0.85, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Số phim đã đánh giá trong train")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Cold-start người dùng: hiệu năng theo độ dài lịch sử")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "results-user-buckets.png", dpi=150)
    plt.close(fig)


def figure_accuracy_vs_novelty(full: pd.DataFrame, beyond: pd.DataFrame,
                               names: dict[str, str]) -> None:
    """Đánh đổi cốt lõi: chính xác cao thường đi kèm gợi ý phim phổ biến."""
    # Ghép theo `exp_name` (khoá thật), không theo nhãn: nhãn có thể trùng giữa hai
    # cấu hình của cùng một thuật toán và ghép theo nhãn sẽ nhân bản dòng.
    # `full` mang sẵn cột `coverage@10`/`gini@10` (gom từ `benchmark_tables`) nên phải
    # bỏ bản sao ở `beyond` trước khi ghép — nếu không, pandas đổi tên thành
    # `coverage@10_x` / `coverage@10_y` và các hàm sau tham chiếu tên cũ sẽ vỡ.
    # Nhãn hiển thị lấy từ `full` (mỗi exp_name có đúng một nhãn ở đó).
    drop = [c for c in beyond.columns if c in full.columns and c != "exp_name"]
    merged = full.merge(beyond.drop(columns=drop), on="exp_name", how="inner")
    if merged.empty:
        raise SystemExit("Không giao được bảng độ chính xác với bảng ngoài độ chính xác "
                         "(tên exp_name không khớp).")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(merged["NDCG@10"], merged["novelty@10"],
               s=merged["coverage@10"] * 2000 + 40,
               c=[COLORS[_family(n)] for n in merged["Thuật toán"]], alpha=0.7)
    for _, r in merged.iterrows():
        ax.annotate(r["Thuật toán"], (r["NDCG@10"], r["novelty@10"]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("NDCG@10 (độ chính xác)")
    ax.set_ylabel("Novelty@10 (càng cao càng gợi ý phim hiếm)")
    ax.set_title("Đánh đổi chính xác / mới lạ — kích thước chấm = coverage@10")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "results-accuracy-vs-novelty.png", dpi=150)
    plt.close(fig)


def figure_beyond_accuracy(beyond: pd.DataFrame, names: dict[str, str]) -> None:
    beyond = beyond.copy()
    beyond["Thuật toán"] = beyond["exp_name"].map(names).fillna(beyond["exp_name"])
    # Gini đặt ngay cạnh coverage: hai chỉ số bù nhau, đọc rời từng cái sẽ hiểu sai
    metrics = [m for m in ("coverage@10", "gini@10", "novelty@10", "ILD@10")
               if m in beyond.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.2))
    for metric, ax in zip(metrics, np.atleast_1d(axes)):
        # Gini ngược chiều các chỉ số khác: thấp mới tốt, nên sắp giảm dần để thanh
        # ngắn nhất (tốt nhất) vẫn nằm trên cùng như ba biểu đồ kia
        df = beyond.sort_values(metric, ascending=metric != "gini@10")
        ax.barh(df["Thuật toán"], df[metric],
                color=[COLORS[_family(n)] for n in df["Thuật toán"]])
        ax.set_title(metric + (" (thấp = trải đều hơn)" if metric == "gini@10" else ""))
    fig.suptitle("Chỉ số ngoài độ chính xác trên toàn bộ người dùng")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "results-beyond-accuracy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    full = _load("full-summary.csv")
    if full is None:
        raise SystemExit("Thiếu results/final/full-summary.csv. Chạy build_final_tables.py trước.")
    names = dict(zip(full["exp_name"], full["Thuật toán"]))

    written = []
    figure_accuracy(full, _load("coldstart-item.csv"))
    figure_metric_vs_k(full)
    written += ["results-accuracy.png", "results-metric-vs-k.png"]

    buckets = _load("coldstart-user-buckets.csv")
    if buckets is not None:
        figure_user_buckets(buckets, names)
        written.append("results-user-buckets.png")

    beyond = _load("beyond-accuracy-full.csv")
    if beyond is not None:
        figure_accuracy_vs_novelty(full, beyond, names)
        figure_beyond_accuracy(beyond, names)
        written += ["results-accuracy-vs-novelty.png", "results-beyond-accuracy.png"]

    print(f"Đã ghi {len(written)} hình vào {FIGURES_DIR}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
