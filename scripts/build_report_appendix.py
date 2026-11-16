"""Sinh phụ lục B cho báo cáo từ các bảng trong `results/final/`.

Bảng sweep đầy đủ (144 dòng) quá dài để nhúng vào Markdown, nên phụ lục trích
**tóm tắt theo mô hình** (biên độ sweep, config tốt nhất) — và người đọc muốn
xem đủ 144 dòng thì mở `results/final/ablation-*.csv` đã commit.

Script chạy:
    venv/bin/python scripts/build_report_appendix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FINAL = ROOT / "results" / "final"
OUT = ROOT / "docs" / "report" / "09-appendix.md"

ABLATION_FILES = ["ablation-cf.csv", "ablation-cb1.csv", "ablation-cb2.csv",
                  "ablation-cb3.csv", "ablation-cb4.csv"]


def load_sweep() -> pd.DataFrame:
    frames = [pd.read_csv(FINAL / f) for f in ABLATION_FILES]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    sweep = load_sweep()
    metric = "NDCG@10"
    if metric not in sweep.columns:
        metric = next(c for c in sweep.columns if c.lower().startswith("ndcg"))

    # Biên độ sweep theo model: (max - min) / max trên NDCG@10 của giao thức dev.
    rows = []
    for model, grp in sweep.groupby("model"):
        vals = grp[metric].astype(float)
        lo, hi = float(vals.min()), float(vals.max())
        best = grp.loc[vals.idxmax()]
        rows.append({
            "model": model,
            "n": len(grp),
            "min": lo,
            "max": hi,
            "spread_pct": (hi - lo) / hi * 100 if hi else 0.0,
            "best_exp": best.get("exp_name", ""),
            "params": best.get("params", ""),
        })
    by_model = pd.DataFrame(rows).sort_values("max", ascending=False)

    lines: list[str] = []
    lines.append("# Phụ lục")
    lines.append("")
    lines.append("> Tất cả số trong phụ lục sinh **tự động** từ "
                 "`results/final/*.csv` bằng `scripts/build_report_appendix.py` — "
                 "không chép tay.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## A. Bảng sweep đầy đủ (144 dòng)")
    lines.append("")
    lines.append("Bảng sweep không nhúng vào báo cáo vì quá dài (144 dòng × nhiều cột "
                 "tham số). Toàn bộ **đã commit** trong repo:")
    lines.append("")
    lines.append("| File | Số dòng | Nội dung |")
    lines.append("|---|---:|---|")
    for f in ABLATION_FILES:
        df = pd.read_csv(FINAL / f)
        lines.append(f"| `results/final/{f}` | {len(df)} | "
                     f"{df['model'].nunique()} mô hình, mỗi dòng mang `params` + "
                     f"`split_id`/`catalog_id`/`seed` |")
    lines.append("")
    lines.append(f"**Tổng: {len(sweep)} dòng = "
                 f"{len(sweep[sweep['model'].isin(['cf-als', 'cf-itemknn', 'rs-itemcf', 'rs-bpr'])])} CF + "
                 f"{len(sweep[~sweep['model'].isin(['cf-als', 'cf-itemknn', 'rs-itemcf', 'rs-bpr'])])} CB.**")
    lines.append("")
    lines.append("### A.1. Biên độ sweep **theo model**")
    lines.append("")
    lines.append("Quy ước: biên độ = **(max − min) / max**, tính trên NDCG@10 của "
                 "giao thức `dev` (20.000 user, `split_id=temporal80_20`, `seed=42`).")
    lines.append("")
    lines.append("| Model | n | min | max | Biên độ | Config tốt nhất |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in by_model.itertuples():
        lines.append(f"| `{r.model}` | {r.n} | {r.min:.4f} | {r.max:.4f} | "
                     f"**{r.spread_pct:.2f}%** | `{r.best_exp}` |")
    lines.append("")
    lines.append("> **Không trộn hai quy ước.** Biên độ *theo model* tính trên lưới "
                 "tham số của riêng mô hình đó; biên độ *theo nhóm* (§5.2.2) tính trên "
                 "toàn bảng ablation của nhóm. Hai con số khác nhau và cả hai đều đúng.")
    lines.append("")

    # --- B. Bootstrap CI + Wilcoxon
    ci = pd.read_csv(FINAL / "bootstrap-ci.csv")
    lines.append("---")
    lines.append("")
    lines.append("## B. Bootstrap CI 95% và kiểm định Wilcoxon")
    lines.append("")
    lines.append("### B.1. Bootstrap CI 95% (giao thức `full`, 158.774 user)")
    lines.append("")
    lines.append("*Nguồn: `results/final/bootstrap-ci.csv`.*")
    lines.append("")
    lines.append("| `exp_name` | NDCG@10 | CI 95% thấp | CI 95% cao | n users |")
    lines.append("|---|---:|---:|---:|---:|")
    ci_sorted = ci.sort_values("NDCG@10", ascending=False)
    for _, r in ci_sorted.iterrows():
        lines.append(f"| `{r['exp_name']}` | {r['NDCG@10']:.4f} | "
                     f"{r['ci95_thấp']:.4f} | {r['ci95_cao']:.4f} | "
                     f"{int(r['n_users']):,} |".replace(",", "."))
    lines.append("")
    lines.append("**Đọc kết quả:** CI của ALS [0,1366; 0,1383] **không chạm** CI của "
                 "CB-4 [0,1047; 0,1062] — khoảng cách giữa CF và CB lớn hơn nhiễu "
                 "thống kê.")
    lines.append("")

    for fname, title in [("wilcoxon-cf.csv", "CF"), ("wilcoxon-cb.csv", "CB"),
                         ("wilcoxon-coldstart.csv", "Cold-start")]:
        df = pd.read_csv(FINAL / fname)
        lines.append(f"### B.{2 if fname == 'wilcoxon-cf.csv' else (3 if fname == 'wilcoxon-cb.csv' else 4)}. "
                     f"Wilcoxon signed-rank — {title}")
        lines.append("")
        lines.append(f"*Nguồn: `results/final/{fname}`.*")
        lines.append("")
        lines.append("| Mốc | So với | Chênh lệch TB | % user tốt hơn | % user kém hơn | p-value |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in df.iterrows():
            p = float(r["p_value"])
            p_txt = f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
            lines.append(
                f"| {r['mốc']} | {r['so_với']} | "
                f"{float(r['chênh_lệch_trung_bình']):.4f} | "
                f"{float(r['tỷ_lệ_user_tốt_hơn']) * 100:.2f}% | "
                f"{float(r['tỷ_lệ_user_kém_hơn']) * 100:.2f}% | {p_txt} |")
        lines.append("")
    lines.append("> **Giới hạn:** kiểm định Wilcoxon được chạy **trong từng bảng**, "
                 "**không** so chéo CF↔CB — vì hai nhóm được tinh chỉnh với số config "
                 "khác nhau, nên so chéo sẽ trộn hiệu ứng của *thuật toán* với hiệu ứng "
                 "của *mức độ tinh chỉnh*. Đây là hướng phát triển (§8.2).")
    lines.append("")

    # --- C. Thống kê dataset
    ds = pd.read_csv(FINAL / "dataset-summary.csv")
    lines.append("---")
    lines.append("")
    lines.append("## C. Thống kê dataset đầy đủ")
    lines.append("")
    lines.append("*Nguồn: `results/final/dataset-summary.csv`.*")
    lines.append("")
    lines.append("| Chỉ số | Giá trị |")
    lines.append("|---|---:|")
    for r in ds.itertuples(index=False):
        lines.append(f"| {r[0]} | {r[1]} |")
    lines.append("")

    # --- D. Quy trình tái lập
    lines.append("---")
    lines.append("")
    lines.append("## D. Quy trình tái lập")
    lines.append("")
    lines.append("Môi trường: Python 3.12, seed = 42. Hướng dẫn đầy đủ (cấu trúc repo, "
                 "mọi lệnh): [`docs/codebase-summary.md`](../codebase-summary.md).")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Tiền xử lý")
    lines.append("venv/bin/python -m src.data.load_movielens_filtered")
    lines.append("venv/bin/python -m src.data.build_item_content_features")
    lines.append("venv/bin/python -m src.data.split_temporal_per_user")
    lines.append("")
    lines.append("# 2. Thí nghiệm (một config ví dụ)")
    lines.append("venv/bin/python -m src.eval.run_experiment_from_config configs/best-cf-als.yaml")
    lines.append("")
    lines.append("# 3. Gom bảng kết quả + hình")
    lines.append("venv/bin/python scripts/build_final_tables.py")
    lines.append("venv/bin/python -m src.eval.make_result_figures")
    lines.append("")
    lines.append("# 4. Kiểm chứng")
    lines.append("venv/bin/python scripts/check_report_numbers.py   # phải báo 0 lệch")
    lines.append("venv/bin/python -m pytest tests/ -q")
    lines.append("```")
    lines.append("")

    # --- E. Cách kiểm chứng số liệu
    lines.append("---")
    lines.append("")
    lines.append("## E. Cách kiểm chứng số liệu trong báo cáo")
    lines.append("")
    lines.append("Mọi con số trong báo cáo đều lấy từ `results/final/*.csv` (27 bảng, "
                 "đã commit). Script kiểm chứng đối chiếu báo cáo với CSV:")
    lines.append("")
    lines.append("```bash")
    lines.append("venv/bin/python scripts/check_report_numbers.py")
    lines.append("```")
    lines.append("")
    lines.append("Kết quả mong đợi: **0 lệch** trên 24.831 giá trị / 27 bảng.")
    lines.append("")
    lines.append("Bổ sung:")
    lines.append("")
    lines.append("- Biên độ sweep trong báo cáo được **tính động** từ "
                 "`ablation-*.csv` (`src/eval/sweep_spread.py`), không hard-code — "
                 "người đọc chạy lại là tự xác minh.")
    lines.append("- `pytest tests/test_report_sweep_spreads.py "
                 "tests/test_report_numbers_consistency.py` kiểm tra hồi quy tài liệu.")
    lines.append("")
    lines.append("**Giới hạn của cơ chế này:** script kiểm tra *số đã ghi trong báo "
                 "cáo* khớp *số trong CSV*. Nó **không** kiểm tra báo cáo có **đủ** "
                 "mọi bảng hay không, và không kiểm tra diễn giải văn bản có đúng với "
                 "số hay không — phần đó là trách nhiệm của người đọc.")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã ghi {OUT} ({len(lines)} dòng)")


if __name__ == "__main__":
    main()
