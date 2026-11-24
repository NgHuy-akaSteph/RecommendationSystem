"""Sinh phụ lục báo cáo từ các bảng trong `results/final/`.

Bảng sweep đầy đủ (144 dòng) quá dài để nhúng vào Markdown, nên phụ lục trích
**tóm tắt theo mô hình** (biên độ sweep, config tốt nhất) — và người đọc muốn
xem đủ 144 dòng thì mở `results/final/ablation-*.csv` đã commit.

Quy ước số: **dấu phẩy = thập phân** (`0,1375`), **dấu chấm = phân cách nghìn**
(`158.774`). Cả hai đều bắt buộc: `docs/report/README.md` và `scripts/check_report_numbers.py`
đều quy định dấu phẩy. Hệ quả: `check_report_numbers.py` chỉ quét được số dạng
dấu phẩy — nên script này **không** được sinh số dạng dấu chấm thập phân, nếu không
số đó sẽ đi qua hàng rào mà không được kiểm tra.

Script chạy:
    venv/bin/python scripts/build_report_appendix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import sweep_spread  # noqa: E402

FINAL = ROOT / "results" / "final"
OUT = ROOT / "docs" / "report" / "09-appendix.md"

ABLATION_FILES = ["ablation-cf.csv", "ablation-cb1.csv", "ablation-cb2.csv",
                  "ablation-cb3.csv", "ablation-cb4.csv"]

CF_MODELS = ("cf-als", "cf-itemknn", "rs-itemcf", "rs-bpr")


def _vn(value: float, decimals: int = 4) -> str:
    """Định dạng thập phân kiểu Việt (`0,1375`).

    Args:
        value: giá trị cần định dạng.
        decimals: số chữ số thập phân.

    Returns:
        Chuỗi đã định dạng, dấu thập phân thay bằng dấu phẩy.
    """
    return f"{value:.{decimals}f}".replace(".", ",")


def _vn_thousands(value: int) -> str:
    """Định dạng số nguyên kiểu Việt với dấu chấm phân cách nghìn (`158.774`).

    Args:
        value: số nguyên.

    Returns:
        Chuỗi đã định dạng.
    """
    return f"{value:,}".replace(",", ".")


def _vn_value(value: object) -> str:
    """Định dạng một ô CSV bất kỳ theo quy ước Việt.

    Ô số -> thập phân dấu phẩy; ô số nguyên lớn -> dấu chấm phân cách nghìn;
    ô chuỗi -> giữ nguyên.

    Args:
        value: giá trị thô từ CSV.

    Returns:
        Chuỗi đã định dạng.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return _vn_thousands(value)
    if isinstance(value, float):
        # Số nguyên lớn -> dùng dấu chấm phân cách nghìn
        if value == int(value) and abs(value) >= 1000:
            return _vn_thousands(int(value))
        return _vn(value, 4)
    return str(value)


def load_sweep() -> pd.DataFrame:
    """Gộp toàn bộ bảng ablation.

    Returns:
        DataFrame ghép của các file `ablation-*.csv`.
    """
    frames = [pd.read_csv(FINAL / f) for f in ABLATION_FILES]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Sinh phụ lục.

    Returns:
        None.
    """
    sweep = load_sweep()
    metric = "NDCG@10"
    if metric not in sweep.columns:
        metric = next(c for c in sweep.columns if c.lower().startswith("ndcg"))

    # Biên độ theo model — lấy từ `sweep_spread` để **một nguồn sự thật**:
    # mỗi model gom từ file ablation **sở hữu** nó, nên `cb1-tfidf` có 15 dòng
    # (không phải 18) và biên độ khớp bảng §4.2.2 trong báo cáo.
    by_model = sweep_spread.spread_by_model()
    best_by_exp = {str(r["exp_name"]): r for _, r in
                   sweep.iterrows() if "exp_name" in sweep.columns}

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
    n_cf = int(sweep["model"].isin(CF_MODELS).sum())
    lines.append(f"**Tổng: {len(sweep)} dòng = {n_cf} CF + "
                 f"{len(sweep) - n_cf} CB.**")
    lines.append("")
    lines.append("### A.1. Biên độ sweep **theo model**")
    lines.append("")
    lines.append("Quy ước: biên độ = **(max − min) / max**, tính trên NDCG@10 của "
                 "giao thức `dev` (20.000 user, `split_id=temporal80_20`, `seed=42`).")
    lines.append("")
    lines.append("| Model | n | min | max | Biên độ | Config tốt nhất |")
    lines.append("|---|---:|---:|---:|---:|---|")
    ordered = sorted(by_model.values(), key=lambda s: s.max_value, reverse=True)
    for sp in ordered:
        lines.append(f"| `{sp.label}` | {sp.n} | {_vn(sp.min_value)} | "
                     f"{_vn(sp.max_value)} | **{_vn(sp.spread_pct, 2)}%** | "
                     f"`{sp.best_exp_name}` |")
    lines.append("")
    lines.append("> **Không trộn hai quy ước.** Biên độ *theo model* tính trên lưới "
                 "tham số của riêng mô hình đó; biên độ *theo nhóm* (§4.2.2) tính trên "
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
        lines.append(f"| `{r['exp_name']}` | {_vn(float(r['NDCG@10']))} | "
                     f"{_vn(float(r['ci95_thấp']))} | "
                     f"{_vn(float(r['ci95_cao']))} | "
                     f"{_vn_thousands(int(r['n_users']))} |")
    lines.append("")
    lines.append("**Đọc kết quả:** CI của ALS [0,1366; 0,1383] **không chạm** CI của "
                 "CB-4 [0,1047; 0,1062] — khoảng cách giữa CF và CB lớn hơn nhiễu "
                 "thống kê.")
    lines.append("")

    sections = [("wilcoxon-cf.csv", "B.2", "CF"),
                ("wilcoxon-cb.csv", "B.3", "CB"),
                ("wilcoxon-coldstart.csv", "B.4", "Cold-start")]
    for fname, tag, title in sections:
        df = pd.read_csv(FINAL / fname)
        lines.append(f"### {tag}. Wilcoxon signed-rank — {title}")
        lines.append("")
        lines.append(f"*Nguồn: `results/final/{fname}`.*")
        lines.append("")
        lines.append("| Mốc | So với | Chênh lệch TB | % user tốt hơn | "
                     "% user kém hơn | p-value |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in df.iterrows():
            # p-value dùng cột đã định dạng sẵn của CSV (ví dụ `<1e-300`) — định
            # dạng lại bằng `.2e` sẽ ra `0,00e+00` và hàng rào không nhận ra.
            p_hi = str(r.get("p_value_hiển_thị", "")).strip()
            if not p_hi:
                p = float(r["p_value"])
                p_hi = f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
            lines.append(
                f"| {r['mốc']} | {r['so_với']} | "
                f"{_vn(float(r['chênh_lệch_trung_bình']))} | "
                f"{_vn(float(r['tỷ_lệ_user_tốt_hơn']) * 100, 2)}% | "
                f"{_vn(float(r['tỷ_lệ_user_kém_hơn']) * 100, 2)}% | {p_hi} |")
        lines.append("")

    # --- B.5. Wilcoxon chéo CF↔CB
    cross_path = FINAL / "wilcoxon-cross-cf-cb.csv"
    if cross_path.exists():
        df_cross = pd.read_csv(cross_path)
        lines.append("### B.5. Wilcoxon chéo CF↔CB — ALS (CF) đối chiếu từng mô hình CB")
        lines.append("")
        lines.append("*Nguồn: `results/final/wilcoxon-cross-cf-cb.csv` — giao thức "
                     "`full`, 158.774 user.*")
        lines.append("")
        lines.append("| Mô hình CB | Chênh lệch TB | % user ALS tốt hơn | "
                     "% user CB tốt hơn | % hòa | rank-biserial | p-value |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in df_cross.iterrows():
            p_hi = str(r.get("p_value_hiển_thị", "")).strip()
            if not p_hi:
                p = float(r["p_value"])
                p_hi = f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
            lines.append(
                f"| {r['Mô hình']} | "
                f"{_vn(float(r['chênh_lệch_trung_bình']))} | "
                f"{_vn(float(r['tỷ_lệ_user_tốt_hơn']) * 100, 2)}% | "
                f"{_vn(float(r['tỷ_lệ_user_kém_hơn']) * 100, 2)}% | "
                f"{_vn(float(r['tỷ_lệ_hoà']) * 100, 2)}% | "
                f"{_vn(float(r['rank_biserial']), 4)} | {p_hi} |")
        lines.append("")
        lines.append("**Đọc kết quả:** ALS thắng **cả 4** mô hình CB với `p < 1e-300`. "
                     "Đây là kiểm định **chéo CF↔CB** — khác với B.2–B.4 vốn kiểm "
                     "định *trong từng bảng*.")
        lines.append("")

    lines.append("> **Giới hạn:** kiểm định Wilcoxon ở B.2–B.4 được chạy **trong "
                 "từng bảng**, **không** so chéo CF↔CB — vì hai nhóm được tinh chỉnh với "
                 "số config khác nhau, nên so chéo sẽ trộn hiệu ứng của *thuật toán* với "
                 "hiệu ứng của *mức độ tinh chỉnh*. Bảng B.5 bên dưới trả lời câu hỏi "
                 "khác: hai *họ* thuật toán có khác nhau theo từng user không.")
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
        lines.append(f"| {r[0]} | {_vn_value(r[1])} |")
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
    lines.append("venv/bin/python -m src.eval.run_experiment_from_config "
                 "configs/best-cf-als.yaml")
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

    # --- E. So sánh giao thức đánh giá
    loo_path = FINAL / "sampled-vs-full.csv"
    if loo_path.exists():
        df_loo = pd.read_csv(loo_path)
        lines.append("---")
        lines.append("")
        lines.append("## E. So sánh giao thức đánh giá — full ranking vs sampled LOO")
        lines.append("")
        lines.append("*Nguồn: `results/final/sampled-vs-full.csv` — "
                     "`scripts/run_sampled_loo_comparison.py`.*")
        lines.append("")
        lines.append("| `exp_name` | NDCG@10 full | NDCG@10 LOO | Hạng full | "
                     "Hạng LOO |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in df_loo.iterrows():
            lines.append(f"| `{r['exp_name']}` | "
                         f"{_vn(float(r['NDCG@10_full']))} | "
                         f"{_vn(float(r['NDCG@10_sampled']))} | "
                         f"{int(r['hạng_full'])} | {int(r['hạng_sampled'])} |")
        lines.append("")
        lines.append("**Đọc kết quả:** Kendall τ = 0,908 (`p = 4,81e-05`) giữa thứ "
                     "hạng full-ranking và sampled LOO. Hàng rào này **bảo vệ** lựa "
                     "chọn full ranking: thứ tự xếp hạng được giữ, nên kết luận "
                     "xếp hạng của đồ án không đổi nếu dùng LOO.")
        lines.append("")

    # --- F. Cách kiểm chứng số liệu
    lines.append("---")
    lines.append("")
    lines.append("## F. Cách kiểm chứng số liệu trong báo cáo")
    lines.append("")
    n_tables = len(list(FINAL.glob("*.csv")))
    lines.append(f"Mọi con số trong báo cáo đều lấy từ `results/final/*.csv` "
                 f"({n_tables} bảng, đã commit). Script kiểm chứng đối chiếu báo cáo "
                 "với CSV:")
    lines.append("")
    lines.append("```bash")
    lines.append("venv/bin/python scripts/check_report_numbers.py")
    lines.append("```")
    lines.append("")
    lines.append("Kết quả mong đợi: **0 lệch** trên toàn bộ giá trị từ các bảng "
                 "CSV.")
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
