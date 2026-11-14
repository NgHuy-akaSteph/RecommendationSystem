"""Chạy giao thức sampled LOO cho mọi thuật toán và so thứ hạng với full ranking.

Kết quả cần là **Kendall τ giữa hai bảng thứ hạng**, không phải bản thân các con số
HR/NDCG của giao thức lấy mẫu. τ thấp nghĩa là hai giao thức xếp hạng thuật toán khác
nhau, tức con số trong các bài báo dùng 100 negative không so sánh được với con số ở đây —
đúng nhận định của Krichene & Rendle (2020), và là lý do dự án chọn full ranking.

Chạy: python scripts/run_sampled_loo_comparison.py [--force]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paths                                             # noqa: E402
from src.eval.sampled_loo_evaluator import prepare, run_config    # noqa: E402

OUTPUT = paths.RESULTS_DIR / "sampled-vs-full.csv"


def run_all() -> pd.DataFrame:
    """Chạy sampled LOO cho mọi cấu hình best-*. Trả về bảng **thô**, chưa ghép giao thức."""
    context = prepare("dev")   # dataset + bộ ứng viên dùng chung cho mọi thuật toán
    rows = []
    for config in sorted(paths.CONFIG_DIR.glob("best-*.yaml")):
        print(f"--- sampled LOO: {config.stem} ---")
        rows.append(run_config(config, users="dev", context=context))
    return pd.DataFrame(rows)


def compare_protocols(sampled: pd.DataFrame) -> pd.DataFrame:
    """Ghép điểm của hai giao thức và xếp hạng thuật toán theo từng giao thức."""
    full = pd.read_csv(paths.RESULTS_DIR / "dev-summary.csv")[["exp_name", "NDCG@10"]]
    full = full.rename(columns={"NDCG@10": "NDCG@10_full"})
    merged = sampled.rename(columns={"NDCG@10": "NDCG@10_sampled",
                                     "HR@10": "HR@10_sampled"}).merge(full, on="exp_name")
    merged["hạng_full"] = merged["NDCG@10_full"].rank(ascending=False).astype(int)
    merged["hạng_sampled"] = merged["NDCG@10_sampled"].rank(ascending=False).astype(int)
    return merged.sort_values("hạng_full")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Cache là bảng **đã ghép**; nạp lại rồi ghép tiếp sẽ va tên cột và pandas đổi hậu tố
    # thành _x/_y, làm `merged["NDCG@10_full"]` nổ KeyError. Vì thế phải rẽ nhánh ở đây
    # chứ không rẽ bên trong run_all.
    if OUTPUT.exists() and not args.force:
        print(f"Đã có {OUTPUT.name}, dùng --force để chạy lại.")
        merged = pd.read_csv(OUTPUT)
    else:
        merged = compare_protocols(run_all())

    show = ["exp_name", "NDCG@10_full", "hạng_full", "NDCG@10_sampled", "HR@10_sampled",
            "hạng_sampled"]
    print("\n=== Thứ hạng thuật toán theo hai giao thức ===")
    print(merged[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    tau, p = stats.kendalltau(merged["hạng_full"], merged["hạng_sampled"])
    print(f"\nKendall τ giữa hai bảng thứ hạng: {tau:.3f} (p = {p:.4f})")
    print("τ = 1 nghĩa là hai giao thức xếp hạng giống hệt; càng thấp thì kết quả của"
          "\ngiao thức lấy mẫu càng không suy ra được kết quả xếp hạng đầy đủ.")

    merged["kendall_tau"] = tau
    merged["kendall_p"] = p
    merged.to_csv(OUTPUT, index=False)
    print(f"\nĐã lưu: {OUTPUT}")


if __name__ == "__main__":
    main()
