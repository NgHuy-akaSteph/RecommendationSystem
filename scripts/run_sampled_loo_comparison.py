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
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paths                                             # noqa: E402
from src.eval.sampled_loo_evaluator import prepare, run_config    # noqa: E402

OUTPUT = paths.RESULTS_DIR / "sampled-vs-full.csv"

# Bảng thứ hạng đối chiếu lấy từ metric gốc của từng thí nghiệm, KHÔNG phải từ
# bảng tổng hợp trung gian:
#
#   - `results/dev-summary.csv` là artifact trung gian (gitignore), thiếu cột
#     `split_id`/`catalog_id`/`seed`, và **không có** `cb4-04-late-bm25-w075`.
#   - `results/final/dev-summary.csv` có đủ cột bằng chứng protocol nhưng chỉ 9 dòng
#     và cũng thiếu `cb4-04-late-bm25-w075`.
#
# Nên nguồn chuẩn là `results/{exp_name}_metrics.csv` — mỗi file một dòng, đủ 12
# `exp_name` của `best-*.yaml`, và mang đầy đủ `split_id`/`catalog_id`/`seed`.
METRIC_TEMPLATE = "metrics.csv"


def _read_exp_name(config_path: Path) -> str:
    """Đọc `exp_name` trong file config."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return str(config["exp_name"])


def unique_best_configs(config_dir: Path | None = None) -> list[Path]:
    """Danh sách config `best-*.yaml` **không trùng `exp_name`**.

    Một số model có nhiều file config cùng trỏ tới một `exp_name` (vd
    `best-rs-itemcf.yaml` và `best-rs-itemcf-binary-pos.yaml` cùng là
    `rs-itemcf-k200-binary-pos`). Chạy cả hai sẽ sinh hai dòng trùng `exp_name`
    trong bảng kết quả → `rank()` tính sai → Kendall τ sai.

    Giữ **file đầu tiên** theo thứ tự `sorted()` (tức tên file đứng trước), các
    file còn lại bị bỏ và được log ra để không mất dấu vết.

    Returns:
        Danh sách path config, mỗi `exp_name` xuất hiện đúng một lần.
    """
    config_dir = config_dir or paths.CONFIG_DIR
    kept: list[Path] = []
    seen: dict[str, Path] = {}
    for path in sorted(config_dir.glob("best-*.yaml")):
        exp_name = _read_exp_name(path)
        if exp_name in seen:
            print(f"  bỏ qua {path.name}: trùng exp_name '{exp_name}' "
                  f"(đã giữ {seen[exp_name].name})")
            continue
        seen[exp_name] = path
        kept.append(path)
    return kept


def load_full_ranking(configs: list[Path]) -> pd.DataFrame:
    """Bảng thứ hạng của giao thức full-ranking cho các config đã chọn.

    Args:
        configs: danh sách config (đã dedup theo `exp_name`).

    Returns:
        DataFrame có cột `exp_name`, `config`, `NDCG@10_full`, `split_id`,
        `catalog_id`, `seed`.

    Raises:
        SystemExit: nếu thiếu file metric của một config, hoặc metric trùng `exp_name`.
    """
    rows = []
    for config_path in configs:
        exp_name = _read_exp_name(config_path)
        metrics_path = paths.RESULTS_DIR / f"{exp_name}_{METRIC_TEMPLATE}"
        if not metrics_path.exists():
            raise SystemExit(
                f"Thiếu file metric {metrics_path.name} cho config {config_path.name}. "
                f"Chạy lại sweep trước khi so sánh giao thức."
            )
        df = pd.read_csv(metrics_path)
        if df["exp_name"].nunique() != 1 or len(df) != 1:
            raise SystemExit(
                f"{metrics_path.name}: mong đợi đúng 1 dòng 1 exp_name, "
                f"thực tế {len(df)} dòng / {df['exp_name'].nunique()} exp_name."
            )
        row = df.iloc[0]
        rows.append({
            "exp_name": exp_name,
            "config": config_path.name,
            "NDCG@10_full": float(row["NDCG@10"]),
            "split_id": row.get("split_id"),
            "catalog_id": row.get("catalog_id"),
            "seed": row.get("seed"),
        })
    out = pd.DataFrame(rows)
    if out["exp_name"].duplicated().any():
        dupes = sorted(out.loc[out["exp_name"].duplicated(), "exp_name"].unique())
        raise SystemExit(f"Trùng exp_name sau khi ghép: {dupes}")
    return out


def run_all(configs: list[Path]) -> pd.DataFrame:
    """Chạy sampled LOO cho mọi config đã chọn. Trả về bảng **thô**, chưa ghép giao thức."""
    context = prepare("dev")   # dataset + bộ ứng viên dùng chung cho mọi thuật toán
    rows = []
    for config in configs:
        print(f"--- sampled LOO: {config.stem} ---")
        row = run_config(config, users="dev", context=context)
        row["config"] = config.name
        rows.append(row)
    return pd.DataFrame(rows)


def compare_protocols(sampled: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Ghép điểm của hai giao thức và xếp hạng thuật toán theo từng giao thức.

    Khoá ghép là cặp (`exp_name`, `config`) — không phải `exp_name` đơn thuần, vì
    `exp_name` có thể trùng giữa hai file config khác nhau.

    Raises:
        SystemExit: nếu bảng sampled có `exp_name` trùng, hoặc merge làm mất dòng.
    """
    # Dòng trùng `exp_name` ở phía sampled sẽ bị `inner join` **âm thầm loại bỏ**
    # nếu `config` khác — bảng ra vẫn đủ số dòng nên kiểm tra "thiếu dòng" không bắt
    # được. Vì thế kiểm tra trùng **trước** khi merge.
    if sampled["exp_name"].duplicated().any():
        dupes = sorted(sampled.loc[sampled["exp_name"].duplicated(), "exp_name"].unique())
        raise SystemExit(
            f"Bảng sampled có exp_name trùng: {dupes}. "
            f"Phải dedup trước (xem unique_best_configs)."
        )

    merged = sampled.rename(
        columns={"NDCG@10": "NDCG@10_sampled", "HR@10": "HR@10_sampled"}
    ).merge(full, on=["exp_name", "config"], how="inner")

    expected = len(full)
    if len(merged) != expected:
        raise SystemExit(
            f"Merge sai: {len(merged)} dòng, mong đợi {expected}. "
            f"Thiếu: {sorted(set(full['exp_name']) - set(merged['exp_name']))}"
        )
    if merged["exp_name"].duplicated().any():
        dupes = sorted(merged.loc[merged["exp_name"].duplicated(), "exp_name"].unique())
        raise SystemExit(f"Merge sinh dòng trùng exp_name: {dupes}")

    merged["hạng_full"] = merged["NDCG@10_full"].rank(ascending=False).astype(int)
    merged["hạng_sampled"] = merged["NDCG@10_sampled"].rank(ascending=False).astype(int)
    return merged.sort_values("hạng_full")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configs = unique_best_configs()
    print(f"Dùng {len(configs)} config best-*.yaml ({len(set(_read_exp_name(c) for c in configs))} "
          f"exp_name khác nhau).")

    # Cache là bảng **đã ghép**; nạp lại rồi ghép tiếp sẽ va tên cột và pandas đổi hậu tố
    # thành _x/_y, làm `merged["NDCG@10_full"]` nổ KeyError. Vì thế phải rẽ nhánh ở đây
    # chứ không rẽ bên trong run_all.
    if OUTPUT.exists() and not args.force:
        print(f"Đã có {OUTPUT.name}, dùng --force để chạy lại.")
        merged = pd.read_csv(OUTPUT)
    else:
        full = load_full_ranking(configs)
        merged = compare_protocols(run_all(configs), full)

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
