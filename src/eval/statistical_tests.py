"""Kiểm định thống kê trên metric per-user: Wilcoxon signed-rank và bootstrap CI.

Vì sao cần: chênh lệch NDCG@10 giữa hai thuật toán là trung bình của 158.774 con số rất
nhiễu. Một khoảng cách 0,005 có thể là thật, cũng có thể là dao động mẫu. Hai công cụ trả
lời hai câu khác nhau:

    Wilcoxon signed-rank — "hai thuật toán có khác nhau không?" Ghép cặp **theo từng user**
        (cùng một người, hai thuật toán) nên triệt tiêu được biến thiên giữa các user, vốn
        lớn hơn nhiều so với biến thiên giữa các thuật toán. Không giả định phân phối
        chuẩn, điều này quan trọng vì NDCG per-user lệch phải rất mạnh và có nhiều số 0.

    Bootstrap CI — "trung bình nằm trong khoảng nào?" Lấy mẫu lại user có hoàn lại 1000
        lần, mỗi lần tính trung bình, rồi lấy phân vị 2,5 và 97,5.

Metric per-user đã lưu sẵn trong results/*_peruser.npz nên không mô hình nào phải chạy lại.

Chạy: python -m src.eval.statistical_tests [--prefix full_] [--metric NDCG@10]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import paths

N_BOOTSTRAP = 1000


def load_all(prefix: str, metric: str, results_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Đọc metric per-user của mọi thí nghiệm có tiền tố cho trước."""
    rdir = results_dir or paths.RESULTS_DIR
    out = {}
    ref_users = None
    for path in sorted(rdir.glob(f"{prefix}*_peruser.npz")):
        data = np.load(path)
        users = data["eval_users"]
        if ref_users is None:
            ref_users = users
        elif not np.array_equal(users, ref_users):
            raise SystemExit(f"{path.name} dùng tập user khác — không ghép cặp được")
        if metric not in data:
            raise SystemExit(f"{path.name} không có '{metric}'. Có: {sorted(data.files)}")
        out[path.name.removesuffix("_peruser.npz")] = data[metric]
    if not out:
        raise SystemExit(f"Không tìm thấy {prefix}*_peruser.npz trong {rdir}")
    return out


BOOT_CHUNK = 100   # số lần lấy mẫu lại xử lý cùng lúc


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 seed: int = paths.SEED) -> tuple[float, float]:
    """Khoảng tin cậy 95% của trung bình, lấy mẫu lại theo user.

    Chia lô và dùng int32 cho chỉ số. Dựng thẳng ma trận [n_boot, n_user] với int64 mặc
    định tốn 1,91 GB ở n_user = 158.774 (1,27 GB riêng cho chỉ số); `bootstrap_table` gọi
    hàm này 9 lần liên tiếp nên đó là rủi ro OOM thật chứ không phải lý thuyết. Chia lô
    100 đưa đỉnh xuống ~190 MB mà kết quả không đổi về mặt thống kê.

    Mọi thuật toán dùng chung một seed ⇒ các lần lấy mẫu lại là **ghép cặp**. Đó là chủ ý,
    nhưng kéo theo một hệ quả: hai khoảng tin cậy chồng lấn **không** đồng nghĩa "khác biệt
    không có ý nghĩa" — dùng `wilcoxon_against` cho câu hỏi đó.
    """
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype="float64")
    for start in range(0, n_boot, BOOT_CHUNK):
        chunk = min(BOOT_CHUNK, n_boot - start)
        idx = rng.integers(0, len(values), size=(chunk, len(values)), dtype=np.int32)
        means[start: start + chunk] = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_against(per_user: dict[str, np.ndarray], reference: str) -> pd.DataFrame:
    """So thuật toán `reference` với mọi thuật toán còn lại, ghép cặp theo user."""
    base = per_user[reference]
    rows = []
    for name, values in per_user.items():
        if name == reference:
            continue
        diff = base - values
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0     # scipy báo lỗi khi mọi hiệu bằng 0
        else:
            # float64 TRƯỚC khi vào scipy: với đầu vào float32, scipy tính và trả p ở float32,
            # nên p cỡ 1e-40 thành giá trị dưới chuẩn vô nghĩa (5.6e-45) hoặc 0 tuyệt đối
            stat, p = stats.wilcoxon(base.astype("float64"), values.astype("float64"),
                                     zero_method="wilcox")
        # Ép về float Python NGAY: metric per-user là float32 nên scipy trả p kiểu
        # np.float32, và khi đó `p < 1e-300` cho False vì numpy hạ 1e-300 xuống float32
        # (underflow về 0.0) rồi so 0.0 < 0.0. Ngưỡng nào nhỏ hơn ~1e-45 cũng dính bẫy này.
        stat, p = float(stat), float(p)
        wins, losses = float((diff > 0).mean()), float((diff < 0).mean())
        rows.append({
            "so_với": name,
            "chênh_lệch_trung_bình": float(diff.mean()),
            # Ba cột thay vì một: với NDCG per-user, phần lớn cặp là hoà (cả hai cùng
            # trượt). Chỉ nêu "thắng 32%" mà giấu "hoà 44%" sẽ khiến người đọc tưởng
            # thuật toán thua ở 68% số user.
            "tỷ_lệ_user_tốt_hơn": wins,
            "tỷ_lệ_user_kém_hơn": losses,
            "tỷ_lệ_hoà": 1.0 - wins - losses,
            # Hiệu ứng rank-biserial: độc lập cỡ mẫu, khác với p-value vốn luôn ≈ 0 ở
            # n = 158.774 nên không phân biệt được khác biệt lớn với khác biệt tí hon
            "rank_biserial": (wins - losses) / max(wins + losses, 1e-12),
            "wilcoxon_stat": stat,
            # p < 1e-300 là tràn số dưới của đuôi phân phối chuẩn tiệm cận, không phải 0 thật
            "p_value": p,
            "p_value_hiển_thị": "<1e-300" if p < 1e-300 else f"{p:.3g}",
            "có_ý_nghĩa_p<0.05": bool(p < 0.05),
        })
    return pd.DataFrame(rows).sort_values("chênh_lệch_trung_bình", ascending=False)


def bootstrap_table(per_user: dict[str, np.ndarray], metric: str) -> pd.DataFrame:
    rows = []
    for name, values in per_user.items():
        lo, hi = bootstrap_ci(values)
        rows.append({"exp_name": name, metric: float(values.mean()),
                     "ci95_thấp": lo, "ci95_cao": hi, "n_users": len(values)})
    return pd.DataFrame(rows).sort_values(metric, ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="full_")
    parser.add_argument("--metric", default="NDCG@10")
    parser.add_argument("--reference", default=None,
                        help="Thuật toán làm mốc so sánh (mặc định: thuật toán CB tốt nhất)")
    args = parser.parse_args()

    per_user = load_all(args.prefix, args.metric)
    boot = bootstrap_table(per_user, args.metric)
    print(f"=== Bootstrap CI 95% ({N_BOOTSTRAP} lần lấy mẫu lại, seed {paths.SEED}) ===")
    print(boot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Mốc mặc định là mô hình content tốt nhất: câu hỏi của đồ án là "CB đứng đâu"
    reference = args.reference or max(
        (n for n in per_user if "cb" in n), key=lambda n: per_user[n].mean(), default=None)
    if reference is None:
        reference = boot.iloc[0]["exp_name"]

    print(f"\n=== Wilcoxon signed-rank, mốc = {reference} ===")
    wil = wilcoxon_against(per_user, reference)
    print(wil.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    boot.to_csv(paths.RESULTS_DIR / "bootstrap-ci.csv", index=False)
    wil.insert(0, "mốc", reference)
    wil.to_csv(paths.RESULTS_DIR / "wilcoxon.csv", index=False)
    print(f"\nĐã lưu: bootstrap-ci.csv, wilcoxon.csv")


if __name__ == "__main__":
    main()
