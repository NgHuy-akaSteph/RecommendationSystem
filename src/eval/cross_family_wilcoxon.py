"""Kiểm định Wilcoxon **chéo CF↔CB** trên giao thức `full`.

Vì sao cần: §5.1 của báo cáo kết luận "CF thắng CB 23,3%". Trước đây kết luận
đó chỉ được đỡ bằng **bootstrap CI không chạm nhau** — một lập luận về *trung bình*,
không trả lời được câu hỏi quan trọng hơn: *"theo từng user, CF có thắng CB không?"*.

Script này chạy Wilcoxon signed-rank **ghép cặp theo user** giữa mô hình CF tốt
nhất (ALS) và từng mô hình CB, trên đúng tập 158.774 user mà cả hai đã được chấm.

Điểm kỹ thuật: metric per-user đã lưu sẵn trong `results/full_*_peruser.npz` và
**cả 13 file dùng chung một mảng `eval_users`** (đã kiểm tra bằng `np.array_equal`).
Nhờ vậy ghép cặp hợp lệ mà **không cần chạy lại mô hình nào** — chi phí tính toán
chỉ là vài giây.

Không trộn với `wilcoxon_for_table` của `src/eval/benchmark_tables.py`: hàm đó cố
tình chỉ so **trong từng bảng** vì bảng CF và bảng CB trả lời hai câu hỏi khác
nhau. Bảng ở đây trả lời câu hỏi **thứ ba**: "hai *họ* thuật toán này có khác nhau
không, khi so trên cùng một tập user?" — và đây chính là phép so sánh mà kết luận
trung tâm của đồ án dựa vào.

Output:
    results/final/wilcoxon-cross-cf-cb.csv

Chạy:
    venv/bin/python -m src.eval.cross_family_wilcoxon
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import paths
from src.eval.benchmark_tables import CF_SPECS, CB_SPECS, Spec, result_name
from src.eval.statistical_tests import wilcoxon_against

# Mô hình CF làm mốc — mô hình CF tốt nhất (NDCG@10 = 0,137470 trên `full`).
REFERENCE_SPEC = next(s for s in CF_SPECS if s.label == "ALS")

# Metric per-user của giao thức `full` có tiền tố "full_".
FULL_PREFIX = "full_"


def _load_per_user(exp_name: str, metric: str = "NDCG@10",
                   results_dir=None) -> tuple[np.ndarray, np.ndarray]:
    """Nạp metric per-user + tập user của một thí nghiệm.

    Args:
        exp_name: tên thí nghiệm, ví dụ `full_cf-als-f064-a005-reg0p1`.
        metric: tên metric.
        results_dir: thư mục kết quả (mặc định `results/`).

    Returns:
        Cặp `(eval_users, values)`.

    Raises:
        SystemExit: nếu thiếu file metric per-user.
    """
    rdir = results_dir or paths.RESULTS_DIR
    path = rdir / f"{exp_name}_peruser.npz"
    if not path.exists():
        raise SystemExit(f"Thiếu {path.name} — chạy lại thí nghiệm trước.")
    data = np.load(path)
    if metric not in data:
        raise SystemExit(f"{path.name} không có '{metric}'.")
    return data["eval_users"], data[metric]


def _exp_name(spec: Spec) -> str:
    """Tên thí nghiệm ở giao thức `full` của một spec."""
    import yaml
    cfg = yaml.safe_load((paths.CONFIG_DIR / spec.config).read_text(encoding="utf-8"))
    return result_name(cfg["exp_name"], "full", None)


def build_cross_family_table(metric: str = "NDCG@10") -> pd.DataFrame:
    """Dựng bảng Wilcoxon ghép cặp giữa ALS (CF) và từng mô hình CB.

    Args:
        metric: metric per-user dùng để kiểm định.

    Returns:
        DataFrame có `mốc`, `so_với`, `Mô hình`, `họ`, `chênh_lệch_trung_bình`,
        `tỷ_lệ_user_tốt_hơn`, `tỷ_lệ_user_kém hơn`, `tỷ_lệ_hoà`,
        `rank_biserial`, `wilcoxon_stat`, `p_value`, `p_value_hiển_thị`,
        `có_ý_nghĩa_p<0.05`.

    Raises:
        SystemExit: nếu các mô hình không dùng chung tập `eval_users`.
    """
    per_user: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    families: dict[str, str] = {}
    ref_users = None

    specs = [REFERENCE_SPEC] + list(CB_SPECS)
    for spec in specs:
        exp = _exp_name(spec)
        users_arr, values = _load_per_user(exp, metric)
        if ref_users is None:
            ref_users = users_arr
        elif not np.array_equal(users_arr, ref_users):
            raise SystemExit(
                f"'{exp}' dùng tập eval_users khác — không ghép cặp được."
            )
        per_user[exp] = values
        labels[exp] = spec.label
        families[exp] = spec.table.upper()

    ref_exp = _exp_name(REFERENCE_SPEC)
    df = wilcoxon_against(per_user, ref_exp)
    df["Mô hình"] = df["so_với"].map(lambda e: labels.get(e, e))
    df["họ"] = df["so_với"].map(lambda e: families.get(e, ""))
    df.insert(0, "mốc", labels.get(ref_exp, ref_exp))
    df.insert(0, "metric", metric)
    df.insert(0, "protocol", "full")
    return df


def main() -> None:
    df = build_cross_family_table()
    out = paths.FINAL_DIR / "wilcoxon-cross-cf-cb.csv"
    df.to_csv(out, index=False)

    show = ["mốc", "Mô hình", "chênh_lệch_trung_bình", "tỷ_lệ_user_tốt_hơn",
            "tỷ_lệ_user_kém_hơn", "tỷ_lệ_hoà", "rank_biserial", "p_value_hiển_thị"]
    print(f"=== Wilcoxon chéo CF↔CB (giao thức `full`, NDCG@10) ===")
    print(df[show].to_string(index=False,
                             float_format=lambda x: f"{x:.4f}"))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
