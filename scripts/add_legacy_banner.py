"""Gắn banner 'tài liệu legacy' đầu các file docs/*.md mô tả protocol cũ.

Phase 04 xoá `src/evaluation/`, `src/recommenders/`, `src/data_loader.py` và
`notebooks/`. Các file docs còn lại (guide CF, ALS, BPR, HR@K...) vẫn mô tả **đường
chạy cũ** và tham chiếu tới file đã xoá — người đọc sẽ tưởng code còn tồn tại.

Banner nói rõ: tài liệu này legacy, và đường chạy hiện hành ở đâu.

Chạy: python scripts/add_legacy_banner.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"

BANNER = """> ⚠️ **Tài liệu legacy — mô tả protocol CŨ (split global 70/30).**
>
> Tài liệu này giải thích thuật toán theo cách làm của `RecommendationSystem` trước
> khi hợp nhất với `recommendation-cb`. Nó **không phải** đường chạy sinh ra số liệu
> báo cáo, và các file code nó tham chiếu (`src/evaluation/`, `src/recommenders/`,
> `src/data_loader.py`, `notebooks/`) **đã bị xoá** ở Phase 04.
>
> **Đường chạy hiện hành:** xem [`README.md`](../README.md) và
> [`docs/data-processing-comparison-across-projects.md`](./data-processing-comparison-across-projects.md)
> (mục 7.1 liệt kê 4 lỗi đánh giá của protocol cũ).
>
> Giá trị còn lại của tài liệu này: phần **giải thích thuật toán** (công thức, trực
> giác, cài đặt) vẫn đúng và hữu ích — chỉ phần *đường chạy* và *số liệu* là cũ.

"""

# Các file docs mô tả protocol cũ — không đụng `data-processing-comparison` (vốn đã
# so sánh cả hai bên) và `project-changelog` (vốn là nhật ký lịch sử).
TARGETS = [
    "als-recommender-explained.md",
    "bpr-implementation-complete-report.md",
    "bpr-implementation-summary.md",
    "bpr-quick-reference.md",
    "cf-implementation-guide.md",
    "cf-quickstart.md",
    "hr-at-k-usage.md",
    "theoretical-foundations-cf-methods.md",
]

MARKER = "Tài liệu legacy — mô tả protocol CŨ"


def main() -> None:
    missing = []
    for name in TARGETS:
        path = DOCS_DIR / name
        if not path.exists():
            missing.append(name)
            continue
        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            print(f"  BỎ QUA {name}: đã có banner")
            continue
        # Banner đặt **sau** dòng tiêu đề H1 (dòng đầu) để cấu trúc tài liệu giữ nguyên.
        lines = text.split("\n")
        if lines and lines[0].startswith("# "):
            head, rest = lines[0], "\n".join(lines[1:])
            new = f"{head}\n\n{BANNER}{rest.lstrip(chr(10))}"
        else:
            new = BANNER + text
        path.write_text(new, encoding="utf-8")
        print(f"  {name}: +banner")
    if missing:
        print(f"KHÔNG tìm thấy: {', '.join(missing)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
