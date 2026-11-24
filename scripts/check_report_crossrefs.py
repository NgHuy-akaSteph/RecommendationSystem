"""Kiểm tra cross-reference nội bộ của báo cáo (`§x.y`).

`scripts/check_report_numbers.py` kiểm tra **số liệu** khớp CSV. Script này kiểm tra
phần còn lại: mọi tham chiếu `§x.y[.z]` trong báo cáo có trỏ tới một mục **thực sự
tồn tại** không.

Vì sao cần: sót một cross-reference là lỗi *âm* — checker số liệu không bắt được,
nhưng người đọc bấm vào tham chiếu thấy mục không tồn tại thì mất niềm tin vào
toàn bộ báo cáo. Sau khi gộp 8 chương xuống 5 chương, ~90 tham chiếu bị đánh số lại
— đây là hàng rào cho thao tác đó.

Chạy:
    venv/bin/python scripts/check_report_crossrefs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "report"

# File báo cáo được kiểm tra (README.md là mục lục — cũng nên kiểm tra)
REPORT_FILES = sorted(
    p for p in REPORT.glob("*.md") if p.name != "README.md"
)

# Mục con viết dạng bold: `**5.3.1 Tiêu đề**`
H2_RE = re.compile(r"^##\s+(\d+)\.(\d+)\s")
H3_RE = re.compile(r"^###\s+(\d+)\.(\d+)\.(\d+)\s")
BOLD_RE = re.compile(r"^\*\*(\d+)\.(\d+)\.(\d+)\s")

REF_RE = re.compile(r"§(\d+)\.(\d+)(?:\.(\d+))?")


def collect_sections() -> set[tuple[int, int, int]]:
    """Thu thập mọi mục tồn tại trong báo cáo.

    Returns:
        Tập khóa `(ch, mục, mục con)`; mục con dùng 0 cho mục lớn.
    """
    sections: set[tuple[int, int, int]] = set()
    for path in REPORT_FILES:
        for line in path.read_text(encoding="utf-8").split("\n"):
            m2 = H2_RE.match(line)
            if m2:
                sections.add((int(m2.group(1)), int(m2.group(2)), 0))
                continue
            m3 = H3_RE.match(line)
            if m3:
                sections.add((int(m3.group(1)), int(m3.group(2)),
                              int(m3.group(3))))
                continue
            mb = BOLD_RE.match(line)
            if mb:
                sections.add((int(mb.group(1)), int(mb.group(2)),
                              int(mb.group(3))))
    return sections


def main() -> int:
    """Kiểm tra và in kết quả.

    Returns:
        0 nếu không có tham chiếu hỏng, 1 nếu có.
    """
    sections = collect_sections()
    problems: list[tuple[str, str]] = []

    for path in REPORT_FILES + [REPORT / "README.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in REF_RE.finditer(text):
            ch, sec, sub = int(match.group(1)), int(match.group(2)), match.group(3)
            key = (ch, sec, int(sub) if sub else 0)
            if key not in sections:
                problems.append((path.name, match.group(0)))

    if problems:
        print(f"✗ {len(problems)} cross-reference hỏng:")
        for fname, ref in problems:
            print(f"    {fname}: {ref}")
        return 1

    print(f"✓ 0 cross-reference hỏng — {len(sections)} mục trong "
          f"{len(REPORT_FILES) + 1} file báo cáo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
