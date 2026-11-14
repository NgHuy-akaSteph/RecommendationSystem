"""Gom kết quả các mô hình thành 2 bảng CF / CB + 1 bảng cold-start item.

Mỗi dòng của bảng là **một cấu hình** trên **một giao thức**, kèm `split_id` và
`catalog_id` — hai cột này là bằng chứng rằng các dòng trong cùng một bảng thực sự
chạy trên cùng tập ứng viên và cùng cách chia. Không có chúng, "cùng giao thức" chỉ là
lời tuyên bố.

Bảng CF và bảng CB là **hai bảng riêng** chứ không phải một bảng xếp hạng chung: chúng
trả lời hai câu hỏi khác nhau. Wilcoxon chỉ chạy **trong từng bảng**: đối chiếu CF với CB
là câu hỏi khác và cần bảng riêng.

Sau khi quét đủ, số config đã chạy là **CF 51 / CB 93** (đếm từ `results/*_metrics.csv`),
nên kết luận "CF thắng CB" giờ đã **cân bằng về độ bao phủ sweep** chứ không còn là so
sánh thiên lệch.

Chạy: python -m src.eval.benchmark_tables --protocol full
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths

FINAL_DIR = paths.RESULTS_DIR / "final"

# (bảng, nhãn hiển thị, config, cột "model" trong metrics CSV)
# Item-CF lấy bản RS (`rs-itemcf` binary_pos) chứ không lấy `cf-itemknn` của CB: hai bản
# cho kết quả khớp tuyệt đối (0,129678) nên đưa cả hai vào bảng CF sẽ là một dòng lặp.
@dataclass(frozen=True)
class Spec:
    table: str
    label: str
    config: str
    model_key: str


CF_SPECS = [
    Spec("cf", "Random (sanity)", "baseline-random.yaml", "random"),
    Spec("cf", "MostPop", "baseline-mostpop.yaml", "mostpop"),
    Spec("cf", "Item-CF (RS)", "best-rs-itemcf-binary-pos.yaml", "rs-itemcf"),
    Spec("cf", "ALS", "best-cf-als.yaml", "cf-als"),
    Spec("cf", "BPR (RS)", "best-rs-bpr.yaml", "rs-bpr"),
]

CB_SPECS = [
    Spec("cb", "CB-1 TF-IDF", "best-cb1-tfidf.yaml", "cb1-tfidf"),
    Spec("cb", "CB-2 BM25", "best-cb2-bm25.yaml", "cb2-bm25"),
    Spec("cb", "CB-3 Tag Genome", "best-cb3-genome.yaml", "cb3-genome"),
    # CB-4 late-fusion (BM25 + Tag Genome), bản **không** dùng popularity prior.
    # Bản có `pop_prior > 0` đạt NDCG@10 cao hơn ở giao thức thường (0,1081 vs 0,1054)
    # nhưng **sụp** ở cold-start (0,0010) vì phim held-out có popularity = 0 bị đẩy xuống
    # đáy — nên bảng chính lấy bản content thuần, bản có prior chỉ để đối chiếu.
    Spec("cb", "CB-4 Hybrid", "best-cb4-hybrid-content.yaml", "cb4-hybrid"),
]

ALL_SPECS = CF_SPECS + CB_SPECS

METRIC_COLS = [f"{m}@{k}" for k in (5, 10, 20, 50) for m in ("P", "R", "NDCG", "HR")]
PROTO_COLS = ["split_id", "catalog_id", "seed", "n_eval_users"]


def result_name(base: str, users: str, split_dir: Path | None) -> str:
    """Khớp với src.eval.run_experiment_from_config.experiment_name."""
    prefix = (f"{split_dir.name}_" if split_dir else "") + ("full_" if users == "full" else "")
    return prefix + base


def protocol_prefix(users: str, split_dir: Path | None) -> str:
    return (f"{split_dir.name}_" if split_dir else "") + ("full_" if users == "full" else "")


def load_metrics(exp_name: str, results_dir: Path | None = None) -> pd.Series | None:
    rdir = results_dir or paths.RESULTS_DIR
    path = rdir / f"{exp_name}_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).iloc[0]


def load_per_user(exp_name: str, metric: str, results_dir: Path | None = None):
    rdir = results_dir or paths.RESULTS_DIR
    path = rdir / f"{exp_name}_peruser.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    if metric not in data:
        return None, None
    return data["eval_users"], data[metric]


def build_table(protocol: str, specs: list[Spec], beyond: pd.DataFrame | None = None,
                results_dir: Path | None = None) -> pd.DataFrame:
    """Một dòng mỗi mô hình trong `specs`, theo đúng giao thức `protocol`.

    protocol: "dev" | "full" | "coldstart".
    """
    rdir = results_dir or paths.RESULTS_DIR
    if protocol == "dev":
        users, split_dir = "dev", None
    elif protocol == "full":
        users, split_dir = "full", None
    elif protocol == "coldstart":
        users, split_dir = "dev", paths.PROCESSED_DIR / "coldstart"
    else:
        raise ValueError(f"protocol lạ: {protocol}")

    beyond_by_exp = {}
    if beyond is not None and not beyond.empty:
        beyond_by_exp = dict(zip(beyond["exp_name"], beyond["exp_name"]))

    rows = []
    for spec in specs:
        import yaml
        cfg = yaml.safe_load((paths.CONFIG_DIR / spec.config).read_text(encoding="utf-8"))
        exp = result_name(cfg["exp_name"], users, split_dir)
        metrics = load_metrics(exp, rdir)
        if metrics is None:
            raise SystemExit(f"Thiếu kết quả '{exp}'. Chạy scripts/run_shared_benchmark.py trước.")
        if metrics.get("model") != spec.model_key:
            raise SystemExit(
                f"'{exp}' có model='{metrics.get('model')}' nhưng bảng CF/CB ghi "
                f"'{spec.model_key}'. Cấu hình đã bị đổi sau khi chạy — phải chạy lại."
            )
        row = {"Mô hình": spec.label, "exp_name": exp, "config": spec.config}
        # Giữ luôn `model` — khoá thô của registry — để người đọc báo cáo tra ngược
        # được từ nhãn hiển thị về đúng lớp mô hình đã chạy (không chỉ từ "Mô hình").
        row["model"] = spec.model_key
        for col in METRIC_COLS:
            if col in metrics:
                row[col] = metrics[col]
        for col in PROTO_COLS:
            if col in metrics:
                row[col] = metrics[col]
        for col in ("fit_s", "score_s"):
            if col in metrics:
                row[col] = metrics[col]
        rows.append(row)

    df = pd.DataFrame(rows)
    if beyond_by_exp:
        cov_cols = [c for c in beyond.columns if c.startswith(("coverage@", "gini@"))]
        for col in cov_cols:
            df[col] = df["exp_name"].map(
                lambda e: beyond.loc[beyond["exp_name"] == e, col].iloc[0]
                if e in beyond_by_exp else np.nan)
    return df


def wilcoxon_for_table(protocol: str, specs: list[Spec], metric: str = "NDCG@10",
                       results_dir: Path | None = None) -> pd.DataFrame:
    """Kiểm định Wilcoxon ghép cặp theo user **giữa các mô hình cùng bảng**.

    Chỉ so trong cùng bảng: đối chiếu CF với CB là câu hỏi khác và cần bảng riêng.
    """
    from src.eval.statistical_tests import wilcoxon_against

    rdir = results_dir or paths.RESULTS_DIR
    if protocol == "dev":
        users, split_dir = "dev", None
    elif protocol == "full":
        users, split_dir = "full", None
    else:
        users, split_dir = "dev", paths.PROCESSED_DIR / "coldstart"

    per_user, names = {}, []
    ref_users = None
    for spec in specs:
        import yaml
        cfg = yaml.safe_load((paths.CONFIG_DIR / spec.config).read_text(encoding="utf-8"))
        exp = result_name(cfg["exp_name"], users, split_dir)
        users_arr, values = load_per_user(exp, metric, rdir)
        if users_arr is None:
            raise SystemExit(f"Thiếu {exp}_peruser.npz — chạy lại thí nghiệm trước.")
        # Ghép cặp Wilcoxon chỉ có nghĩa khi hai mô hình chấm trên **cùng** tập user.
        # Nếu lệch thì "so sánh" là so hai quần thể khác nhau và kết luận là vô nghĩa.
        if ref_users is None:
            ref_users = users_arr
        elif not np.array_equal(users_arr, ref_users):
            raise SystemExit(f"'{exp}' dùng tập eval_users khác — không ghép cặp được")
        per_user[exp] = values
        names.append((exp, spec.label))

    ref_exp = max(per_user, key=lambda n: float(per_user[n].mean()))
    df = wilcoxon_against(per_user, ref_exp)
    labels = dict(names)
    df["Mô hình"] = df["so_với"].map(lambda e: labels.get(e, e))
    df.insert(0, "mốc", labels.get(ref_exp, ref_exp))
    df.insert(0, "protocol", protocol)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="full", choices=["dev", "full", "coldstart"])
    parser.add_argument("--output-dir", default=str(FINAL_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beyond = None
    beyond_path = paths.beyond_accuracy_csv(args.protocol)
    if beyond_path.exists():
        beyond = pd.read_csv(beyond_path)

    tables = {}
    # Tên file: bảng `full` và `dev` dùng chung họ "cf"/"cb" nên cần hậu tố phân biệt
    # giao thức; bảng cold-start đã có "coldstart" trong tên nên không thêm hậu tố nữa.
    suffix = {"full": "", "dev": "-dev", "coldstart": ""}[args.protocol]
    if args.protocol == "coldstart":
        tables["coldstart"] = build_table(args.protocol, ALL_SPECS, beyond)
    else:
        tables["cf"] = build_table(args.protocol, CF_SPECS, beyond)
        tables["cb"] = build_table(args.protocol, CB_SPECS, beyond)

    for name, df in tables.items():
        df = df.sort_values("NDCG@10", ascending=False)
        path = out_dir / f"benchmark-{name}{suffix}.csv"
        df.to_csv(path, index=False)
        print(f"\n=== Bảng {name} (protocol={args.protocol}) ===")
        show = [c for c in ("Mô hình", "NDCG@10", "NDCG@20", "HR@10", "coverage@10",
                            "gini@10", "fit_s", "score_s") if c in df.columns]
        print(df[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"→ {path}")

    # Wilcoxon ghép cặp **trong từng bảng**: đối chiếu CF với CB là câu hỏi khác và
    # cần bảng riêng, vì số config đã quét hai bên lệch nhau (CF 51, CB 93).
    spec_groups = {"cf": CF_SPECS, "cb": CB_SPECS}
    if args.protocol == "coldstart":
        spec_groups = {"coldstart": ALL_SPECS}
    for name, specs in spec_groups.items():
        try:
            wil = wilcoxon_for_table(args.protocol, specs)
        except SystemExit as exc:
            print(f"  BỎ QUA wilcoxon-{name}: {exc}")
            continue
        path = out_dir / f"wilcoxon-{name}{suffix}.csv"
        wil.to_csv(path, index=False)
        print(f"\n=== Wilcoxon {name} (mốc = {wil['mốc'].iloc[0]}) ===")
        print(wil.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print(f"→ {path}")


if __name__ == "__main__":
    main()
