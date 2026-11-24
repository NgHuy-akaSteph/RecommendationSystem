#!/usr/bin/env python
"""Đối chiếu số liệu trong báo cáo với `results/final/`.

Báo cáo được viết tay (Markdown) nên dễ trôi số. Script này thay phần "đọc tay"
bằng một kiểm tra máy: mọi số trong báo cáo phải **truy được về** một giá trị đo
được trong `results/final/*.csv` (hoặc một đại lượng tính được từ đó).

Cách kiểm:
  1. Nạp mọi CSV trong `results/final/`, gom **mọi** ô số.
  2. Định dạng mỗi giá trị theo kiểu Việt ở nhiều độ chính xác (2–6 chữ số thập phân)
     → tập giá trị được phép. Kèm dạng **phần trăm** (giá trị × 100) vì báo cáo
     hay viết coverage là `14,94%` trong khi CSV lưu `0,149392`.
  3. Quét các file báo cáo, tìm mọi token dạng `0,1375`, `23,3%`, `8,77e-201`.
  4. Token nào không nằm trong tập được phép → **lệch** → `exit 1`.

Đây là kiểm tra **whitelist**: số trong báo cáo phải khớp, không yêu cầu mọi số trong
CSV đều xuất hiện trong báo cáo.

Chạy: python scripts/check_report_numbers.py [--verbose]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paths                                   # noqa: E402
from src.eval import sweep_spread                       # noqa: E402

REPORT_DIR = paths.ROOT / "docs" / "report"

# File báo cáo được kiểm tra — toàn bộ các chương trong `docs/report/`.
# `README.md` bị loại có chủ đích: đó là mục lục, không chứa số liệu.
REPORT_FILES = [
    REPORT_DIR / "00-abstract.md",
    REPORT_DIR / "01-introduction.md",
    REPORT_DIR / "02-literature-review.md",
    REPORT_DIR / "03-dataset-and-methodology.md",
    REPORT_DIR / "04-results-and-analysis.md",
    REPORT_DIR / "05-discussion-and-conclusion.md",
    REPORT_DIR / "09-appendix.md",
]

# Độ chính xác dùng để định dạng giá trị CSV (số thập phân)
PRECISIONS = (1, 2, 3, 4, 5, 6)

# File **dữ liệu gốc** của MovieLens 25M — số dòng các file này được nêu ở §3.1.1
# nhưng không nằm trong `results/final/`. Checker đọc trực tiếp để đối chiếu
# (không hard-code) — file nằm trong `data/` (gitignore) nên nếu thiếu thì bỏ qua.
#
# Lưu ý: đếm dòng bằng số byte, trừ dòng header — chính xác với CSV không có
# dấu nháy kép nhúng newline.
RAW_MOVIELENS_FILES = {
    "ratings.csv": "Số dòng ratings.csv (hành vi xem phim)",
    "movies.csv": "Số dòng movies.csv (siêu dữ liệu phim)",
    "tags.csv": "Số dòng tags.csv (tag người dùng tự gán)",
    "genome-scores.csv": "Số dòng genome-scores.csv (điểm liên quan Tag Genome)",
    "genome-tags.csv": "Số dòng genome-tags.csv (tên tag trong genome)",
}

# Mẫu token số kiểu Việt: `0,1375`, `23,3`, `−0,6` (dấu trừ Unicode hoặc ASCII)
#
# `(?![\d,])` ở cuối chặn việc cắt đôi một số dài (`1,234` không thành `1,2` + `34`),
# nhưng phải cho phép chữ `e` phía sau — nếu không thì p-value `1,93e-63` sẽ bị
# tách thành `1,93` và `63`, và `1,93` sẽ bị báo lệch trong khi p-value đã khớp.
NUMBER_RE = re.compile(r"(?<![\d,])([−\-]?\d+,\d{1,6})(?![\d,])")

# Mẫu p-value ký hiệu khoa học kiểu Việt: `8,77e-201`
SCIENTIFIC_RE = re.compile(r"(?<![\d,])(\d+,\d{2}e[+-]?\d{1,4})(?![\d])")

# Dấu trừ dùng nhiều kiểu trong văn bản — chuẩn hoá về ASCII
MINUS_CHARS = ("−", "–", "—")


def _normalize_sign(text: str) -> str:
    """Chuẩn hoá dấu trừ Unicode về ASCII và bỏ khoảng trắng."""
    for ch in MINUS_CHARS:
        text = text.replace(ch, "-")
    return text.replace(" ", "")


def _split_sign(token: str) -> tuple[str, str]:
    """Tách dấu (−1, '+') khỏi phần trị tuyệt đối của token."""
    token = _normalize_sign(token)
    if token.startswith("-"):
        return "-", token[1:]
    if token.startswith("+"):
        return "+", token[1:]
    return "", token


# Mẫu số dùng **dấu chấm** — báo cáo dùng song song hai quy ước:
#   • dấu phẩy = thập phân   → `0,1375`  (quét bởi NUMBER_RE)
#   • dấu chấm  = phân cách nghìn → `158.774`
#
# Lý do cần mẫu này: `NUMBER_RE` chỉ khớp `\d+,\d{1,6}` nên **bỏ sót toàn bộ**
# số viết bằng dấu chấm — trong đó có những số quan trọng (số dòng dataset gốc,
# số phim, hạng trong demo). Trước khi có mẫu này, checker báo "0 lệch" cho cả
# những chỗ sai — hàng rào không trong suốt.
#
# Ràng buộc: phải có ít nhất một chữ số trước dấu chấm, để không khớp tham chiếu
# mục như `§5.2`.
DOT_NUMBER_RE = re.compile(
    r"(?<![\d,.])([−\-]?\d{1,3}(?:\.\d{3})+(?![\d,.])|[−\-]?\d+\.\d{1,6}(?![\d,.]))"
)


def _dot_variants(token: str) -> set[str]:
    """Các cách viết tương đương của một token dùng dấu chấm.

    Với `158.774` có **hai** cách hiểu: phân cách nghìn (= 158774) hoặc thập phân
    (= 158.774). Với `0.1375` chỉ có một cách hiểu hợp lý (thập phân). Script thử
    cả hai và chấp nhận nếu **một** cách khớp — vì mục tiêu là phát hiện số sai,
    không phải phán xét quy ước trình bày.

    Args:
        token: token thô bắt được từ `DOT_NUMBER_RE`.

    Returns:
        Tập các biến thể có thể khớp với giá trị trong `results/final/`.
    """
    normalized = _normalize_sign(token)
    sign, magnitude = _split_sign(normalized)
    variants: set[str] = set()

    if "." in magnitude:
        # Cách hiểu 1 — thập phân: "0.1375" -> 0,1375
        try:
            variants.add(_format_vn(float(magnitude), 6))
        except ValueError:
            pass
        # Cách hiểu 2 — phân cách nghìn: "13.816" -> 13816
        variants.add(magnitude.replace(".", ""))
        variants.add(magnitude.replace(".", ","))
        # Giữ nguyên token gốc (dạng dấu chấm) để khớp với EXEMPT
        variants.add(magnitude)
    else:
        variants.add(magnitude)

    # Giữ dấu trong biến thể có dấu
    if sign == "-":
        variants = {f"-{v}" for v in variants} | variants
    return variants

# Số trong báo cáo không phải metric — cần nêu rõ lý do để được miễn trừ.
#
# Mỗi mục là (giá trị, lý do). Chỉ miễn trừ đúng giá trị đó, không miễn trừ cả lớp —
# nếu miễn trừ quá rộng thì script hết tác dụng.
EXEMPT: dict[str, str] = {
    "64,128": "tập giá trị tham số {64,128}, không phải metric",
    "0,87": "viết tắt '~0,87' cho 0,8674 (tỷ lệ hoà)",
    "76,7": "tỷ lệ ô genome bị xoá khi threshold áp sau z-score (§4.4.3) — "
            "thống kê đặc thù tiền xử lý, không phải metric đánh giá",
    "34,6": "số cũ bị loại ('4.587 user / ×34,6') được dẫn lại để chứng minh "
            "không tái lập được (§7.3) — không phải kết quả hiện hành",
    "0,142889": "điểm BM25 của một phim cold-start cụ thể, đọc trực tiếp từ demo "
                "Streamlit tab 3 (§6.6.4) — không phải metric tổng hợp trong CSV",
    "4.587": "số cũ bị loại ('4.587 user') được dẫn lại để chứng minh không tái "
             "lập được (§7.3) — không phải kết quả hiện hành",
    # EXEMPT ở dạng **dấu chấm** vì `_dot_variants` sinh ra cả hai dạng.
    "13816": "số phim còn lại sau lọc catalog (13.816) — đọc từ tên catalog_id",
}


def _format_vn(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _raw_movielens_counts() -> dict[str, str]:
    """Số dòng thực tế của các file MovieLens gốc.

    Đọc trực tiếp từ `data/raw/ml-25m/` thay vì hard-code — nếu dataset đổi,
    checker báo lệch thay vì im lặng chấp nhận con số cũ.

    Returns:
        Dict `số dòng (dạng chuỗi nguyên)` -> mô tả. Rỗng nếu thiếu dữ liệu.
    """
    raw_dir = paths.ROOT / "data" / "raw" / "ml-25m"
    if not raw_dir.exists():
        return {}
    counts: dict[str, str] = {}
    for name, reason in RAW_MOVIELENS_FILES.items():
        path = raw_dir / name
        if not path.exists():
            continue
        with path.open("rb") as fh:
            n_lines = sum(1 for _ in fh) - 1  # trừ dòng header
        counts[str(n_lines)] = reason
    return counts


def _derived_values() -> dict[str, str]:
    """Các đại lượng **tính được** từ CSV nhưng không xuất hiện trực tiếp trong ô nào.

    Script tự tính để không phải hard-code con số — cùng nguyên tắc với
    `tests/test_report_sweep_spreads.py`.
    """
    derived: dict[str, str] = {}

    # Biên độ sweep (theo model và theo nhóm) — công thức (max−min)/max
    for sp in sweep_spread.spread_by_model().values():
        derived[sweep_spread.format_vn(sp.spread_pct, 2)] = "biên độ sweep (theo model)"
    for sp in sweep_spread.spread_by_group().values():
        derived[sweep_spread.format_vn(sp.spread_pct, 2)] = "biên độ sweep (theo nhóm)"

    # Biên độ lát cắt reg × iterations của ALS
    derived[sweep_spread.format_vn(sweep_spread.als_reg_iterations_spread(), 2)] = \
        "biên độ lát cắt reg×iterations của ALS"

    # Biên độ ALS theo mẫu số trung bình — báo cáo nêu để nêu rõ quy ước
    als_rows = pd.read_csv(paths.FINAL_DIR / "ablation-cf.csv")
    als_rows = als_rows[als_rows["model"] == "cf-als"]
    als_mean = (float(als_rows["NDCG@10"].max()) - float(als_rows["NDCG@10"].min())) \
        / float(als_rows["NDCG@10"].mean()) * 100
    derived[sweep_spread.format_vn(als_mean, 2)] = "biên độ ALS theo mẫu số trung bình"

    # Chênh lệch ALS − CB-4 ở hai mẫu số (đã nêu rõ trong báo cáo)
    als = 0.13747012615203857
    cb4 = 0.1054347977042198
    derived[sweep_spread.format_vn((als - cb4) / als * 100, 1)] = "chênh ALS−CB4 (mẫu số ALS)"
    derived[sweep_spread.format_vn((als - cb4) / cb4 * 100, 1)] = "chênh ALS−CB4 (mẫu số CB4)"

    # Lệch tương đối dev ↔ full của từng model (§5.2.3)
    dev = pd.read_csv(paths.FINAL_DIR / "benchmark-cf-dev.csv")
    full = pd.read_csv(paths.FINAL_DIR / "benchmark-cf.csv")
    for name in ("ALS", "Item-CF (RS)", "BPR (RS)", "MostPop"):
        d = float(dev.loc[dev["Mô hình"] == name, "NDCG@10"].iloc[0])
        f = float(full.loc[full["Mô hình"] == name, "NDCG@10"].iloc[0])
        derived[sweep_spread.format_vn((f - d) / d * 100, 1)] = f"lệch dev↔full ({name})"
    dev_cb = pd.read_csv(paths.FINAL_DIR / "benchmark-cb-dev.csv")
    full_cb = pd.read_csv(paths.FINAL_DIR / "benchmark-cb.csv")
    for name in ("CB-2 BM25", "CB-1 TF-IDF", "CB-3 Tag Genome"):
        d = float(dev_cb.loc[dev_cb["Mô hình"] == name, "NDCG@10"].iloc[0])
        f = float(full_cb.loc[full_cb["Mô hình"] == name, "NDCG@10"].iloc[0])
        derived[sweep_spread.format_vn((f - d) / d * 100, 1)] = f"lệch dev↔full ({name})"

    # Chênh lệch giữa các model CB trong bảng phân tầng (đối chiếu "chỉ genres")
    strata = pd.read_csv(paths.FINAL_DIR / "tag-leakage-strata.csv")
    cb1 = strata[strata["Mô hình"].str.startswith("CB-1")].iloc[0]
    genres = strata[strata["Mô hình"].str.contains("genres")].iloc[0]
    for col in ("Trước kỷ nguyên tag (< 2006)", "Giao thời (2006–2012)",
                "Tag đã dày (>= 2012)"):
        diff = float(cb1[col]) - float(genres[col])
        derived[sweep_spread.format_vn(diff, 4)] = "chênh CB-1 − chỉ genres"

    # Chênh lệch CF − CB theo nhóm user (bảng §5.5)
    buckets = pd.read_csv(paths.FINAL_DIR / "coldstart-user-buckets.csv").set_index("exp_name")
    als_b = buckets.loc["full_cf-als-f064-a005-reg0p1"]
    cb4_b = buckets.loc["full_cb4-04-late-bm25-w075"]
    for col in ("5–30", "31–100", ">100"):
        derived[sweep_spread.format_vn(float(als_b[col]) - float(cb4_b[col]), 4)] = \
            "chênh ALS − CB-4 theo nhóm user"

    # Chênh lệch chính xác của mô hình so với mốc CB-4 (bảng Wilcoxon cold-start)
    wc = pd.read_csv(paths.FINAL_DIR / "wilcoxon-coldstart.csv")
    for _, row in wc.iterrows():
        derived[sweep_spread.format_vn(float(row["chênh_lệch_trung_bình"]), 6)] = \
            f"chênh lệch TB Wilcoxon ({row['Mô hình']})"

    # Số dòng file MovieLens gốc — đọc trực tiếp, không hard-code
    for count, reason in _raw_movielens_counts().items():
        derived[count] = reason

    # Số lượng item held-out của giao thức cold-start.
    #
    # Đọc từ `paths.ROOT / "data" / "processed"` chứ **không** dùng
    # `paths.PROCESSED_DIR` — conftest monkeypatch biến `PROCESSED_DIR` sang
    # `tmp_path`, nên dùng `PROCESSED_DIR` sẽ không tìm thấy file và im lặng
    # bỏ qua số này (hàng rào mất tác dụng trong test).
    heldout = (paths.ROOT / "data" / "processed" / "coldstart"
               / "heldout_items.npy")
    if heldout.exists():
        import numpy as np
        n_heldout = int(np.load(heldout).shape[0])
        derived[str(n_heldout)] = "số phim held-out của giao thức cold-start"

    # Bảng so sánh các cách chia tập (§3.3.3) — tái lập được, đọc từ CSV
    split_cmp = paths.FINAL_DIR / "split-comparison.csv"
    if split_cmp.exists():
        df_split = pd.read_csv(split_cmp)
        for col in df_split.columns:
            if df_split[col].dtype.kind not in "if":
                continue
            for raw in df_split[col].dropna():
                for decimals in PRECISIONS:
                    derived[sweep_spread.format_vn(float(raw), decimals)] = \
                        "số user theo cách chia tập (§3.3.3)"

    # Số hạng của phim cold-start trong demo (§4.11.4) — đọc từ CSV
    demo = paths.FINAL_DIR / "demo-coldstart-comparison.csv"
    if demo.exists():
        df_demo = pd.read_csv(demo)
        for col in df_demo.columns:
            if df_demo[col].dtype.kind not in "if":
                continue
            for raw in df_demo[col].dropna():
                for decimals in PRECISIONS:
                    derived[sweep_spread.format_vn(float(raw), decimals)] = \
                        "số liệu demo cold-start"

    # Kết quả so sánh giao thức full-ranking vs sampled LOO (§3.9.3)
    loo = paths.FINAL_DIR / "sampled-vs-full.csv"
    if loo.exists():
        df_loo = pd.read_csv(loo)
        for col in df_loo.columns:
            if df_loo[col].dtype.kind not in "if":
                continue
            for raw in df_loo[col].dropna():
                for decimals in PRECISIONS:
                    derived[sweep_spread.format_vn(float(raw), decimals)] = \
                        "số liệu so sánh giao thức full vs sampled LOO"

    return derived


def _collect_allowed(final_dir: Path) -> set[str]:
    """Tập giá trị số được phép xuất hiện trong báo cáo.

    Bao gồm:
      - giá trị thô ở nhiều độ chính xác (1–6 chữ số thập phân);
      - **dạng âm** (`-0,022823` cho ô `−0,022823`);
      - **dạng phần trăm**: 0,149392 → `14,94` (báo cáo hay viết coverage là `14,94%`).
    """
    allowed: set[str] = set()
    for csv_path in sorted(final_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, dtype=str)
        for col in df.columns:
            for raw in df[col].dropna():
                text = _normalize_sign(str(raw))
                # Bỏ ký hiệu phần trăm — "1,0988%" chỉ là 1.0988
                text = text.rstrip("%")
                try:
                    value = float(text.replace(",", ""))
                except ValueError:
                    continue
                for decimals in PRECISIONS:
                    allowed.add(_format_vn(value, decimals))
                    allowed.add(_format_vn(abs(value), decimals))
                    # Dạng phần trăm: 0,149392 → "14,94"
                    allowed.add(_format_vn(value * 100, decimals))
                    allowed.add(_format_vn(abs(value) * 100, decimals))
                # CSV có thể ghi số lớn bằng dấu phân cách nghìn: "162,540" = 162540.
                # Báo cáo cũng dùng đúng dạng này, nên phải cho phép giá trị gốc.
                allowed.add(text)
    return allowed


def _strip_code_spans(text: str) -> str:
    """Bỏ nội dung trong `` ` `` — đó là literal config, không phải số liệu."""
    return re.sub(r"`[^`]*`", " ", text)


def _mask_scientific(text: str) -> str:
    """Che p-value ký hiệu khoa học trước khi quét số thập phân.

    `1,93e-63` gồm hai phần: `1,93` là mantissa, `63` là số mũ. Nếu không che đi,
    mẫu số thập phân sẽ bắt riêng `1,93` và báo lệch, dù p-value đã khớp. P-value
    được kiểm riêng bằng `SCIENTIFIC_RE`.
    """
    return SCIENTIFIC_RE.sub(lambda m: " " * len(m.group(0)), text)


# Tham chiếu mục dạng `§4.10` — **không phải số liệu**. Phải che trước khi quét
# số dấu chấm, vì `4.10` trông y hệt một số thập phân.
SECTION_REF_MASK_RE = re.compile(r"§\d+(?:\.\d+)+")

# Số trong **tiêu đề mục** (`## 4.10 ...`) cũng không phải số liệu — đó là số thứ
# tự mục, không phải giá trị đo được.
HEADING_MASK_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def _mask_section_refs(text: str) -> str:
    """Che tham chiếu mục `§x.y` và số trong tiêu đề mục.

    Args:
        text: nội dung markdown.

    Returns:
        Nội dung đã che tham chiếu mục và số trong heading.
    """
    text = SECTION_REF_MASK_RE.sub(lambda m: " " * len(m.group(0)), text)
    return HEADING_MASK_RE.sub(lambda m: " " * len(m.group(0)), text)


def _scientific_variants(token: str) -> set[str]:
    """Các cách viết khác của một p-value ký hiệu khoa học.

    `8,77e-201` trong báo cáo là giá trị đã làm tròn của `8.767218e-201`. Không thể
    suy ra trực tiếp từ CSV bằng cách so chuỗi, nên p-value được đối chiếu bằng
    cách kiểm tra giá trị **thực** có làm tròn khớp hay không.
    """
    return {token}


def check(allowed: set[str], derived: dict[str, str],
          verbose: bool = False) -> list[tuple[str, str, str]]:
    """Quét báo cáo, trả về danh sách lệch `(file, token, lý do)`."""
    problems: list[tuple[str, str, str]] = []
    p_values = _collect_p_values()

    for md_path in REPORT_FILES:
        if not md_path.exists():
            problems.append((md_path.name, "-", "file báo cáo không tồn tại"))
            continue
        text = _strip_code_spans(md_path.read_text(encoding="utf-8"))
        seen: set[str] = set()

        for match in SCIENTIFIC_RE.finditer(text):
            token = match.group(1)
            if token in seen:
                continue
            seen.add(token)
            if _matches_p_value(token, p_values):
                continue
            problems.append((md_path.name, token, "p-value không khớp results/final/"))

        # Che p-value trước khi quét số thập phân — mantissa của p-value không phải
        # số liệu riêng (xem _mask_scientific).
        text = _mask_scientific(text)

        # Che tham chiếu mục `§4.10` — không phải số liệu (xem _mask_section_refs).
        dot_text = _mask_section_refs(text)

        for match in NUMBER_RE.finditer(text):
            token = match.group(1)
            if token in seen:
                continue
            seen.add(token)
            sign, magnitude = _split_sign(token)
            # Đối chiếu cả bản có dấu lẫn bản tuyệt đối: "−0,022823" phải khớp
            # ô "−0,022823" trong CSV, còn "0,022823" (thiếu dấu) vẫn được chấp nhận
            # vì CSV lưu số âm có dấu.
            if sign in ("", "-") and magnitude in allowed:
                continue
            if magnitude in allowed or magnitude in derived or magnitude in EXEMPT:
                continue
            problems.append((md_path.name, token, "không truy được về results/final/"))

        # Số dùng dấu chấm — `158.774` (nghìn) hoặc `0.1375` (thập phân).
        # Trước khi có vòng này, toàn bộ số dạng này đi qua checker mà không
        # hề được kiểm tra.
        for match in DOT_NUMBER_RE.finditer(dot_text):
            token = match.group(1)
            if token in seen:
                continue
            seen.add(token)
            if any(v in allowed or v in derived or v in EXEMPT
                   for v in _dot_variants(token)):
                continue
            problems.append((md_path.name, token, "không truy được về results/final/"))
    return problems


def _collect_p_values() -> list[float]:
    """Gom mọi p-value trong `results/final/` để đối chiếu p-value ghi trong báo cáo."""
    values: list[float] = []
    for csv_path in sorted(paths.FINAL_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        for col in df.columns:
            if "p_value" not in col and "wilcoxon" not in col:
                continue
            for raw in df[col].dropna():
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    continue
    return values


def _matches_p_value(token: str, p_values: list[float]) -> bool:
    """Token kiểu `8,77e-201` có khớp p-value nào trong CSV (sau khi làm tròn) không."""
    normalized = token.replace(",", ".")
    try:
        value = float(normalized)
    except ValueError:
        return False
    for p in p_values:
        if p == 0:
            continue
        # Làm tròn theo 3 chữ số có nghĩa như CSV đã làm (`p_value_hiển_thị`)
        for decimals in (2, 3):
            if f"{p:.{decimals}e}".replace("e", "e") == normalized:
                return True
            if _format_vn(p * 100, decimals) == normalized:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true",
                        help="in chi tiết số đã đối chiếu và số miễn trừ")
    args = parser.parse_args()

    allowed = _collect_allowed(paths.FINAL_DIR)
    derived = _derived_values()
    problems = check(allowed, derived, verbose=args.verbose)

    if problems:
        print(f"✗ {len(problems)} số trong báo cáo KHÔNG khớp results/final/:")
        for filename, token, reason in problems:
            print(f"    {filename}: '{token}' — {reason}")
        print("\nSửa: lấy số từ results/final/*.csv, hoặc nếu là đại lượng suy ra "
              "thì thêm vào _derived_values() kèm lý do.")
        sys.exit(1)

    n_files = len([p for p in REPORT_FILES if p.exists()])
    n_csv = len(list(paths.FINAL_DIR.glob("*.csv")))
    print(f"✓ 0 lệch — đã đối chiếu {n_files} file báo cáo "
          f"với {len(allowed)} giá trị từ {n_csv} bảng CSV.")
    if args.verbose:
        print(f"  giá trị được phép: {len(allowed)}")
        print(f"  đại lượng suy ra: {len(derived)}")
        for value, reason in sorted(derived.items()):
            print(f"    {value} — {reason}")
        print("  miễn trừ:")
        for value, reason in sorted(EXEMPT.items()):
            print(f"    {value} — {reason}")


if __name__ == "__main__":
    main()
