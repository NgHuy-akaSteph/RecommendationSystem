"""Phân tích cold-start **người dùng**: metric theo số lượng lịch sử train của user.

Cold-start item (phim mới) đã có split riêng. Đây là nửa còn lại: người dùng mới, ít lịch
sử. Không cần split mới — chỉ cần chia tập đánh giá sẵn có theo |train_u| rồi đọc lại
metric per-user đã lưu.

Kỳ vọng cần kiểm chứng: lọc cộng tác suy giảm nhanh hơn lọc nội dung khi lịch sử ngắn, vì
CF phải suy ra sở thích từ hành vi trùng lặp với người khác, còn CB chỉ cần một vài phim
là dựng được hồ sơ nội dung. Nếu số liệu không cho thấy điều đó thì phải nói thẳng.

Chạy: python -m src.eval.user_bucket_analysis [--prefix full_] [--metric NDCG@10]
      [--protocol all|dev] [--output coldstart-user-buckets.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import paths
from src.data.build_interaction_matrices import load_dataset
from src.eval.collect_results_to_table import strip_protocol_prefix
from src.eval.statistical_tests import bootstrap_ci

# Biên chọn theo MIN_TRAIN=5 của split: nhóm đầu là "vừa đủ để đánh giá được"
BUCKETS = [(5, 30, "5–30"), (31, 100, "31–100"), (101, 10**9, ">100")]


def train_history_size(users: np.ndarray, split_dir: Path | None = None) -> np.ndarray:
    """Số tương tác train của từng user, theo thứ tự `users`."""
    train_csr = load_dataset(users="full", split_dir=split_dir).train.train_csr
    return np.diff(train_csr.indptr)[users]


def bucket_table(prefix: str = "full_", metric: str = "NDCG@10",
                 results_dir: Path | None = None, protocol: str = "all") -> pd.DataFrame:
    """Metric trung bình theo nhóm độ dài lịch sử, một dòng mỗi thí nghiệm.

    Args:
        protocol: ``"dev"`` chỉ lấy thí nghiệm không có tiền tố giao thức. Bắt buộc khi
            phân tích kết quả dev, vì kết quả dev **không có tiền tố** nên ``prefix=""``
            sẽ vơ cả ``full_*`` lẫn ``coldstart_*`` vào cùng một bảng — ba giao thức khác
            nhau nằm chung, và guard `eval_users` bên dưới sẽ báo lỗi một cách khó hiểu
            thay vì nói đúng nguyên nhân.
    """
    rdir = results_dir or paths.RESULTS_DIR
    files = sorted(rdir.glob(f"{prefix}*_peruser.npz"))
    if protocol == "dev":
        files = [p for p in files
                 if strip_protocol_prefix(p.name) == p.name]
    elif protocol != "all":
        raise ValueError(f"protocol phải là 'dev' hoặc 'all', nhận '{protocol}'")
    if not files:
        raise SystemExit(f"Không tìm thấy {prefix}*_peruser.npz (protocol={protocol}) trong {rdir}")

    split_dir = paths.PROCESSED_DIR / "coldstart" if prefix.startswith("coldstart") else None
    rows, sizes, ref_users = [], None, None
    for path in files:
        data = np.load(path)
        users, values = data["eval_users"], data[metric]
        # `sizes` chỉ tính từ file đầu tiên rồi áp cho mọi file sau. Nếu một file cũ có
        # cùng độ dài nhưng khác tập user thì bucket sẽ sai âm thầm — cùng lớp lỗi mà
        # statistical_tests.load_all đã chặn, nên chặn y hệt ở đây.
        if ref_users is None:
            ref_users, sizes = users, train_history_size(users, split_dir)
        elif not np.array_equal(users, ref_users):
            raise SystemExit(f"{path.name} dùng tập eval_users khác — không so bucket được")
        row = {"exp_name": path.name.removesuffix("_peruser.npz")}
        for lo, hi, label in BUCKETS:
            mask = (sizes >= lo) & (sizes <= hi)
            row[label] = float(values[mask].mean()) if mask.any() else np.nan
            row[f"n_{label}"] = int(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bucket_diffs(reference: str, others: list[str], metric: str = "NDCG@10",
                        results_dir: Path | None = None) -> pd.DataFrame:
    """Chênh lệch **ghép cặp** từng thí nghiệm so với `reference`, tách theo nhóm lịch sử.

    Trung bình nhóm trong `bucket_table` không đủ để kết luận khi hai mô hình sát nhau
    (vd. rating vs binary_pos ở nhóm 5–30 chỉ lệch 0,0026). Ở đây: bootstrap CI 95% của
    chênh lệch per-user và Wilcoxon cặp, tính trên cùng tập user. Dùng cho kết quả dev.
    """
    rdir = results_dir or paths.RESULTS_DIR
    ref = np.load(rdir / f"{reference}_peruser.npz")
    users = ref["eval_users"]
    sizes = train_history_size(users)
    rows = []
    for name in others:
        other = np.load(rdir / f"{name}_peruser.npz")
        if not np.array_equal(other["eval_users"], users):
            raise SystemExit(f"{name} dùng tập eval_users khác {reference}")
        # float64 trước khi vào scipy — xem chú thích ở statistical_tests.wilcoxon_against
        diff = other[metric].astype("float64") - ref[metric].astype("float64")
        for lo, hi, label in BUCKETS:
            d = diff[(sizes >= lo) & (sizes <= hi)]
            if len(d) == 0:     # nhóm rỗng: không có gì để kiểm định
                continue
            lo_ci, hi_ci = bootstrap_ci(d)
            # Cùng quy ước với statistical_tests.wilcoxon_against: bỏ cặp hoà khỏi Wilcoxon
            # nhưng báo tỷ lệ hoà riêng — với NDCG per-user, phần lớn cặp là hoà
            p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
            rows.append({"exp_name": name, "mốc": reference, "nhóm": label, "n": len(d),
                         "chênh_lệch": float(d.mean()), "ci_low": lo_ci, "ci_high": hi_ci,
                         "tỷ_lệ_tốt_hơn": float((d > 0).mean()),
                         "tỷ_lệ_kém_hơn": float((d < 0).mean()),
                         "tỷ_lệ_hoà": float((d == 0).mean()), "p_wilcoxon": p})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="full_")
    parser.add_argument("--metric", default="NDCG@10")
    parser.add_argument("--protocol", choices=["all", "dev"], default="all",
                        help="dev = chỉ thí nghiệm không có tiền tố giao thức")
    parser.add_argument("--output", default="coldstart-user-buckets.csv",
                        help="Tên file CSV trong results/")
    args = parser.parse_args()

    df = bucket_table(args.prefix, args.metric, protocol=args.protocol)
    labels = [b[2] for b in BUCKETS]
    show = df[["exp_name"] + labels].sort_values(labels[-1], ascending=False)

    print(f"=== {args.metric} theo số lịch sử train của user ===")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSố user mỗi nhóm:", {lb: int(df[f"n_{lb}"].iloc[0]) for lb in labels})
    print("\nTỷ lệ suy giảm từ nhóm >100 xuống nhóm 5–30 (càng nhỏ càng bền với user mới):")
    for _, r in show.iterrows():
        if r[labels[-1]] > 0:
            print(f"  {r['exp_name']:<45} {r[labels[0]] / r[labels[-1]]:.2f}x")

    out = paths.RESULTS_DIR / args.output
    df.to_csv(out, index=False)
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
