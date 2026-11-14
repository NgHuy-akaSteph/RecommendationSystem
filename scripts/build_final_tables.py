"""Gom kết quả benchmark của phase 03 vào `results/final/` cho chương báo cáo.

**Nguồn sự thật duy nhất là `src.eval.benchmark_tables`** — bảng sinh ở đó mang theo
`split_id` / `catalog_id` / `seed` / `model`, nên mỗi dòng tự chứng minh được nó chạy
trên cùng tập ứng viên với các dòng khác trong cùng bảng. Script này **không tính toán
metric mới**, chỉ rút gọn và đặt tên cho dễ đọc.

Ba điều cố ý làm khác bản cũ:

- **Không bao giờ ghi đè `results/final/benchmark-*.csv`.** Đó là sản phẩm của phase 03
  có bằng chứng protocol; bản ở đây là bản *rút gọn*, ghi đè là mất bằng chứng.
- **Không crash khi thiếu dữ liệu.** Bản cũ gọi `pd.read_csv(results/dev-summary.csv)`
  không bọc try → `FileNotFoundError` dừng cả lượt chạy. Ở đây phần nào thiếu thì bỏ qua
  và **báo rõ**, phần nào đủ thì ghi.
- **Nhãn không trùng lặp.** Bản cũ dùng `variant_label` để gắn hậu tố khi một thuật toán
  có nhiều cấu hình, kết quả là hai dòng đều tên "CB-1 TF-IDF (content thuần)" và các
  mô hình RS hiện tên thô (`rs-itemcf`, `rs-bpr`) lọt vào bảng dù không có nhãn.

Chạy: python scripts/build_final_tables.py
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paths                                              # noqa: E402
from src.eval import benchmark_tables as bt                         # noqa: E402
from src.eval.beyond_accuracy_metrics import compute_all            # noqa: E402
from src.eval.statistical_tests import bootstrap_table, load_all    # noqa: E402
from src.eval.user_bucket_analysis import bucket_table, paired_bucket_diffs  # noqa: E402

FINAL_DIR = paths.RESULTS_DIR / "final"

# Ba chế độ trọng số hồ sơ của CB-1: binary_pos (mốc), centered, rating
WEIGHT_MODE_EXPS = ["cb1-01-add-tags-tfidf", "cb1-04-centered-weights",
                    "cb1-04b-rating-weights"]

# File của phase 03 — script này chỉ đọc, tuyệt đối không ghi đè.
PROTECTED = ("benchmark-", "wilcoxon-")


def _guard_not_protected(name: str) -> None:
    if name.startswith(PROTECTED):
        raise SystemExit(
            f"'{name}' là sản phẩm của phase 03 (có split_id/catalog_id). "
            "Script này không được ghi đè. Dùng: python -m src.eval.benchmark_tables")


class _Missing(Exception):
    """Thiếu dữ liệu đầu vào — bỏ qua phần đó, không dừng cả lượt chại."""


def _try(fn, *args, **kwargs):
    """Gọi `fn` và chuẩn hoá mọi lỗi 'thiếu dữ liệu' thành `_Missing`.

    Các module phân tích ném cả `SystemExit` (có thông điệp) lẫn `FileNotFoundError`
    (khi file per-user của một cấu hình chưa từng chạy). Bản cũ chỉ bắt `SystemExit` nên
    vẫn crash với `FileNotFoundError` — đúng lỗi đang sửa ở đây.
    """
    try:
        return fn(*args, **kwargs)
    except (SystemExit, FileNotFoundError, KeyError, ValueError) as exc:
        raise _Missing(str(exc)) from exc


def _save(df: pd.DataFrame, name: str) -> Path | None:
    """Ghi CSV vào results/final/, bỏ qua (có báo) nếu không có dữ liệu."""
    _guard_not_protected(name)
    if df is None or df.empty:
        print(f"  BỎ QUA {name}: không có dữ liệu")
        return None
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_DIR / name, index=False)
    print(f"  {name}: {len(df)} dòng")
    return FINAL_DIR / name


def to_report_table(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột cho khớp phần báo cáo, **giữ nguyên** cột bằng chứng protocol."""
    out = df.rename(columns={"Mô hình": "Thuật toán"}).copy()
    # Cột "model" là khoá thô của registry; "Thuật toán" là nhãn đọc được. Giữ cả hai
    # để tra ngược được từ báo cáo về đúng lớp mô hình đã chạy.
    if "model" not in out.columns:
        out.insert(1, "model", pd.NA)
    front = ["Thuật toán", "model", "exp_name", "config"]
    cols = [c for c in front if c in out.columns] + \
           [c for c in out.columns if c not in front]
    return out[cols]


def protocol_tables(protocol: str) -> pd.DataFrame:
    """Ghép các bảng của một giao thức thành một bảng tổng hợp (CF + CB).

    Cột `split_id` / `catalog_id` được giữ nguyên để bảng rút gọn vẫn chứng minh được
    mọi dòng cùng giao thức.
    """
    if protocol == "coldstart":
        specs = bt.ALL_SPECS
    else:
        specs = bt.CF_SPECS + bt.CB_SPECS
    beyond_path = paths.beyond_accuracy_csv(protocol)
    beyond = pd.read_csv(beyond_path) if beyond_path.exists() else None
    df = bt.build_table(protocol, specs, beyond)
    return to_report_table(df)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ghi vào {FINAL_DIR}\n")

    print("Bảng theo giao thức:")
    for protocol, name in [("dev", "dev-summary.csv"),
                           ("full", "full-summary.csv"),
                           ("coldstart", "coldstart-item.csv")]:
        try:
            _save(_try(protocol_tables, protocol), name)
        except _Missing as exc:
            print(f"  BỎ QUA {name}: {exc}")

    print("\nCold-start người dùng:")
    try:
        _save(_try(bucket_table, "full_"), "coldstart-user-buckets.csv")
    except _Missing as exc:
        print(f"  BỎ QUA coldstart-user-buckets.csv: {exc}")

    try:
        # Kiểm chứng giả thuyết "hồ sơ rỗng" (C2): ba chế độ trọng số của CB-1 trên dev
        dev = _try(bucket_table, "", protocol="dev")
        _save(dev[dev["exp_name"].isin(WEIGHT_MODE_EXPS)], "weight-mode-buckets.csv")
        _save(_try(paired_bucket_diffs, WEIGHT_MODE_EXPS[0], WEIGHT_MODE_EXPS[1:]),
              "weight-mode-paired-tests.csv")
    except _Missing as exc:
        print(f"  BỎ QUA weight-mode-buckets.csv: {exc}")

    print("\nNgoài độ chính xác:")
    # Ghi thẳng vào results/final/ — bản ở gốc results/ là artifact trung gian (gitignore).
    for prefix, name in [("full_", "beyond-accuracy-full.csv"),
                         ("coldstart_", "beyond-accuracy-coldstart.csv")]:
        try:
            _save(_try(compute_all, prefix), name)
        except _Missing as exc:
            print(f"  BỎ QUA {name}: {exc}")

    print("\nKiểm định thống kê:")
    try:
        per_user = _try(load_all, "full_", "NDCG@10")
        _save(_try(bootstrap_table, per_user, "NDCG@10"), "bootstrap-ci.csv")
    except _Missing as exc:
        print(f"  BỎ QUA bootstrap-ci.csv: {exc}")

    # Wilcoxon **trong từng bảng** (CF / CB / cold-start) do `src.eval.benchmark_tables`
    # sinh và ghi thẳng vào results/final/ — vì nó đi kèm cột `split_id`/`catalog_id`.
    # Script này không ghi đè: so chéo CF↔CB là câu hỏi khác và cần bảng riêng, số config
    # đã quét hai bên lệch nhau (CF 48, CB 8).
    print("\nWilcoxon: sinh bởi `python -m src.eval.benchmark_tables --protocol "
          "{full,coldstart}` (không ghi đè ở đây).")

    sampled = paths.RESULTS_DIR / "sampled-vs-full.csv"
    if sampled.exists():
        shutil.copyfile(sampled, FINAL_DIR / sampled.name)
        print(f"\n  {sampled.name}: đã sao chép")

    print(f"\nXong. {len(list(FINAL_DIR.glob('*.csv')))} bảng trong {FINAL_DIR}")


if __name__ == "__main__":
    main()
