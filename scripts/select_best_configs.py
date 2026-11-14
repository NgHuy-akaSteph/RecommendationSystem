"""Chọn cấu hình tốt nhất mỗi thuật toán và sinh bảng tổng hợp dev.

Đầu ra:
    configs/best-{model}.yaml   bản sao config thắng cuộc của từng thuật toán
    results/dev-summary.csv     một dòng mỗi thuật toán, đầy đủ metric ở mọi k

Nhóm theo cột ``model`` chứ không theo tiền tố ``exp_name``: một vài thí nghiệm mang tên
``cb4-03-timedecay-*`` nhưng thực chất là biến thể của CB-1 (model ``cb1-tfidf``), gom
theo tên sẽ xếp nhầm nhóm.

Chạy: python scripts/select_best_configs.py [--metric NDCG@10]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paths                                    # noqa: E402
from src.eval.collect_results_to_table import collect    # noqa: E402

# Tên hiển thị trong báo cáo, theo đúng thứ tự trình bày
DISPLAY_NAMES = {
    "random": "Random",
    "mostpop": "MostPop",
    "cb1-tfidf": "CB-1 TF-IDF",
    "cb2-bm25": "CB-2 BM25",
    "cb3-genome": "CB-3 Tag Genome",
    "cb3-ridge": "CB-3 Ridge (biến thể)",
    "cb4-hybrid": "CB-4 Hybrid",
    "cf-itemknn": "ItemKNN-CF",
    "cf-als": "ALS",
}
METRIC_COLS = [f"{m}@{k}" for k in (5, 10, 20, 50) for m in ("P", "R", "NDCG", "HR")]


def select_best(metric: str = "NDCG@10") -> pd.DataFrame:
    # protocol="dev" bắt buộc: results/ còn chứa full_* và coldstart_*, mà điểm full
    # thường cao hơn dev nên argmax không lọc sẽ trỏ sang một exp_name không có file
    # config tương ứng — write_best_configs im lặng bỏ qua và giữ nguyên best-*.yaml cũ.
    df = collect(sort_by=None, protocol="dev")
    if df.empty:
        raise SystemExit("Chưa có kết quả nào trong results/")
    if "model" not in df.columns:
        raise SystemExit("Thiếu cột 'model' trong results/*_metrics.csv")

    if metric not in df.columns:
        raise SystemExit(f"Không có cột '{metric}'. Chọn một trong: "
                         f"{sorted(c for c in df.columns if '@' in c)}")
    best = df.loc[df.groupby("model")[metric].idxmax()].copy()
    best["Thuật toán"] = best["model"].map(lambda m: DISPLAY_NAMES.get(m, m))
    order = {m: i for i, m in enumerate(DISPLAY_NAMES)}
    return best.sort_values("model", key=lambda s: s.map(lambda m: order.get(m, 99)))


def write_content_only_hybrid(metric: str = "NDCG@10") -> Path | None:
    """Ghi thêm cấu hình hybrid **không dùng popularity prior**.

    Cấu hình CB-4 thắng ở dev có λ_pop > 0, nhưng chính prior đó khiến nó sụp ở cold-start
    (phim held-out có popularity bằng 0 nên bị đẩy xuống dưới mọi phim cũ): 0,0347 → 0,0010.
    Nếu chỉ sinh một best-cb4-hybrid.yaml thì `run_final_evaluations.py` không bao giờ chạy
    lại được con số cold-start thật, và người clone repo sẽ ra kết luận ngược.
    """
    df = collect(sort_by=None, protocol="dev")
    df = df[df["model"] == "cb4-hybrid"].copy()
    if df.empty:
        return None
    df["pop"] = df["params"].map(lambda p: json.loads(p).get("pop_prior", 0.0))
    pure = df[df["pop"] == 0.0]
    if pure.empty:
        print("  BỎ QUA best-cb4-hybrid-content.yaml: không có cấu hình λ_pop = 0")
        return None

    src = paths.CONFIG_DIR / f"{pure.loc[pure[metric].idxmax(), 'exp_name']}.yaml"
    dst = paths.CONFIG_DIR / "best-cb4-hybrid-content.yaml"
    shutil.copyfile(src, dst)
    print(f"  best-cb4-hybrid-content.yaml  <- {src.stem}")
    return dst


def write_best_configs(best: pd.DataFrame) -> list[Path]:
    written = []
    for _, row in best.iterrows():
        src = paths.CONFIG_DIR / f"{row['exp_name']}.yaml"
        if not src.exists():
            print(f"  BỎ QUA {row['model']}: không tìm thấy {src.name}")
            continue
        dst = paths.CONFIG_DIR / f"best-{row['model']}.yaml"
        shutil.copyfile(src, dst)
        written.append(dst)
        print(f"  best-{row['model']}.yaml  <- {row['exp_name']}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="NDCG@10", help="Chỉ số dùng để chọn best")
    args = parser.parse_args()

    best = select_best(args.metric)
    print(f"Cấu hình tốt nhất theo {args.metric}:")
    write_best_configs(best)
    write_content_only_hybrid(args.metric)

    cols = ["Thuật toán", "exp_name"] + [c for c in METRIC_COLS if c in best.columns] \
        + [c for c in ("fit_s", "score_s", "n_eval_users", "params") if c in best.columns]
    summary = best[cols]
    out = paths.RESULTS_DIR / "dev-summary.csv"
    summary.to_csv(out, index=False)

    show = ["Thuật toán", "P@10", "R@10", "NDCG@10", "HR@10", "NDCG@20"]
    print(f"\n{summary[show].to_string(index=False, float_format=lambda x: f'{x:.4f}')}")
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
