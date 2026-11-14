"""Gom kết quả từ results/*_metrics.csv thành bảng tổng hợp markdown/csv.

Bảng ablation của từng nhóm CB được **sinh bằng code** chứ không chép tay, để rerun
sau khi sửa tham số là bảng tự khớp lại.

Chạy:
    python -m src.eval.collect_results_to_table                       # bảng tổng hợp
    python -m src.eval.collect_results_to_table --ablation            # ghi ablation-cb*.csv
    python -m src.eval.collect_results_to_table --prefix cb1- --sort NDCG@10
"""

import argparse
from pathlib import Path

import pandas as pd

from src import paths

DISPLAY_COLS = ["exp_name", "model", "P@10", "R@10", "NDCG@10", "HR@10",
                "NDCG@20", "fit_s", "score_s", "n_eval_users", "params"]
# Bảng ablation giữ thứ tự bước (theo exp_name) chứ không sắp theo metric: người đọc
# cần thấy metric biến thiên thế nào qua từng bước tối ưu.
#
# Mỗi nhóm gồm **nhiều tiền tố** vì sweep CF không dùng một tiền tố chung: `cf-*`
# (ALS + ItemKNN) và `rs-*` (Item-CF, BPR — model cũ của repo CB) là hai họ riêng.
# Trước đây bảng CF chỉ lấy `cf-` (18 dòng) trong khi tài liệu tuyên bố "CF đã quét
# 51 config" — bằng chứng không khớp tuyên bố. Nay lấy cả hai họ (51 dòng).
ABLATION_GROUPS = {"cb1": "ablation-cb1.csv", "cb3": "ablation-cb3.csv",
                   "cb2": "ablation-cb2.csv", "cb4": "ablation-cb4.csv",
                   "cf": "ablation-cf.csv"}
# Tiền tố exp_name của từng nhóm ablation. Nhóm CF có hai tiền tố vì sweep CF gồm
# cả họ `cf-` (ALS, ItemKNN) lẫn họ `rs-` (Item-CF, BPR).
ABLATION_PREFIXES = {"cb1": ("cb1-",), "cb2": ("cb2-",), "cb3": ("cb3-",),
                     "cb4": ("cb4-",), "cf": ("cf-", "rs-")}
# Tiền tố do run_experiment_from_config.experiment_name gắn thêm khi đổi giao thức.
# Phải bóc ra trước khi đối chiếu với configs/, nếu không mọi kết quả full/cold-start đều
# bị guard chống kết quả mồ côi loại nhầm.
PROTOCOL_PREFIXES = ("coldstart_", "full_")


def strip_protocol_prefix(exp_name: str) -> str:
    """Bỏ tiền tố giao thức để lấy lại tên config gốc.

    Một lượt duy nhất qua danh sách, không lặp: `experiment_name` gắn tối đa một tiền tố
    split và một tiền tố users, theo đúng thứ tự này. Lặp vô hạn sẽ cắt nát một config
    tương lai lỡ đặt tên bắt đầu bằng ``full_``.
    """
    for prefix in PROTOCOL_PREFIXES:
        if exp_name.startswith(prefix):
            exp_name = exp_name[len(prefix):]
    return exp_name


def collect(results_dir: Path | None = None, sort_by: str | None = "NDCG@10",
            prefix: str | None = None, protocol: str = "all") -> pd.DataFrame:
    """Đọc mọi file *_metrics.csv, gộp thành một DataFrame, sắp theo sort_by.

    Args:
        protocol: ``"dev"`` chỉ lấy thí nghiệm không có tiền tố giao thức, ``"all"`` lấy hết.
            Mặc định ``"all"`` để giữ nguyên hành vi cũ, nhưng **người gọi nào chọn cấu hình
            tốt nhất thì bắt buộc dùng "dev"**: từ phase 4 trở đi results/ chứa cả
            ``full_*`` và ``coldstart_*``, mà điểm của giao thức full thường cao hơn dev,
            nên argmax không lọc sẽ chọn nhầm một dòng full rồi đi tìm
            ``configs/full_….yaml`` không tồn tại — im lặng giữ nguyên best-*.yaml cũ.
    """
    rdir = results_dir or paths.RESULTS_DIR
    csv_files = sorted(rdir.glob("*_metrics.csv"))
    if not csv_files:
        print(f"Không tìm thấy file *_metrics.csv trong {rdir}")
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # configs/ là nguồn sự thật về "sweep gồm những gì". Đổi tên một thí nghiệm sẽ để lại
    # file CSV cũ do **code cũ** sinh ra; gộp nó vào bảng là trộn số trước và sau khi sửa lỗi.
    known = {p.stem for p in paths.CONFIG_DIR.glob("*.yaml")}
    orphans = sorted(n for n in set(df["exp_name"]) if strip_protocol_prefix(n) not in known)
    if orphans:
        print(f"  CẢNH BÁO: bỏ {len(orphans)} kết quả không còn config tương ứng: {orphans}")
        df = df[[strip_protocol_prefix(n) in known for n in df["exp_name"]]]

    if protocol == "dev":
        df = df[[strip_protocol_prefix(n) == n for n in df["exp_name"]]]
    elif protocol != "all":
        raise ValueError(f"protocol phải là 'dev' hoặc 'all', nhận '{protocol}'")

    if prefix:
        df = df[df["exp_name"].str.startswith(prefix)]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    return df


def write_ablation_tables(results_dir: Path | None = None) -> dict[str, Path]:
    """Sinh results/ablation-cb1.csv và ablation-cb3.csv theo thứ tự bước ablation."""
    rdir = results_dir or paths.RESULTS_DIR
    written = {}
    for group, filename in ABLATION_GROUPS.items():
        prefixes = ABLATION_PREFIXES[group]
        df = collect(rdir, sort_by=None)
        df = df[df["exp_name"].str.startswith(prefixes)]
        if df.empty:
            print(f"  bỏ qua {filename}: chưa có kết quả nào cho {prefixes}")
            continue
        df = df.sort_values("exp_name")
        out = rdir / filename
        df.to_csv(out, index=False)
        written[group] = out
        print(f"  {filename}: {len(df)} thí nghiệm")
    return written


def to_markdown(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Xuất bảng markdown từ DataFrame, chọn cột nếu cần."""
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except (ImportError, ModuleNotFoundError):
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = [
            "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header, sep] + rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sort", default="NDCG@10", help="Cột sắp xếp")
    parser.add_argument("--prefix", default=None, help="Chỉ lấy exp_name bắt đầu bằng chuỗi này")
    parser.add_argument("--ablation", action="store_true", help="Ghi results/ablation-cb*.csv")
    parser.add_argument("--output", default=None, help="Lưu CSV tổng hợp (mặc định: stdout)")
    args = parser.parse_args()

    if args.ablation:
        write_ablation_tables()

    df = collect(sort_by=args.sort, prefix=args.prefix)
    if df.empty:
        return
    print(to_markdown(df, DISPLAY_COLS))

    if args.output:
        out = Path(args.output)
        df.to_csv(out, index=False)
        print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
