#!/usr/bin/env python3
"""Đổi cấu trúc báo cáo từ 8 chương sang 5 chương — tự động đánh số lại.

Bản gốc có 8 chương + abstract + phụ lục. Yêu cầu mới: gộp về 5 chương.
Script này gộp file, đánh số lại heading và đổi mọi cross-reference.

Vì sao viết script thay vì sửa tay: báo cáo có ~50 mục con và ~90 cross-reference.
Sửa tay dễ sót — và sót một cross-reference là lỗi *âm*, checker không bắt được
(checker chỉ kiểm tra số liệu, không kiểm tra con trỏ nội bộ).

Chạy:
    venv/bin/python scripts/restructure_report_to_5_chapters.py --dry-run
    venv/bin/python scripts/restructure_report_to_5_chapters.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "report"

# File mới <- (các file cũ, tiêu đề chương mới)
CHAPTER_MAP = [
    ("01-introduction.md", ["01-introduction.md"],
     "Chương 1 — Giới thiệu"),
    ("02-literature-review.md", ["02-literature-review.md"],
     "Chương 2 — Tổng quan và cơ sở lý thuyết"),
    ("03-dataset-and-methodology.md",
     ["03-dataset-and-preprocessing.md", "04-methodology.md"],
     "Chương 3 — Dữ liệu, tiền xử lý và phương pháp"),
    ("04-results-and-analysis.md",
     ["05-experimental-results.md", "06-leakage-analysis-and-discussion.md"],
     "Chương 4 — Kết quả thực nghiệm và phân tích"),
    ("05-discussion-and-conclusion.md",
     ["07-discussion.md", "08-conclusion.md"],
     "Chương 5 — Thảo luận và kết luận"),
]

# File bị xoá sau khi gộp
OBSOLETE = ["03-dataset-and-preprocessing.md", "04-methodology.md",
            "05-experimental-results.md",
            "06-leakage-analysis-and-discussion.md", "07-discussion.md",
            "08-conclusion.md"]

# Chương cũ -> chương mới (cho tham chiếu dang "Chương N")
CHAPTER_MAP_NUM = {1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 5, 8: 5}

SECTION_REF_RE = re.compile(r"§(\d+)\.(\d+)(?:\.(\d+))?")
# Tham chiếu cấp chương dạng `§5` — không có phần thập phân
CHAPTER_SECTION_REF_RE = re.compile(r"§(\d+)(?![\d.])")
CHAPTER_REF_RE = re.compile(r"\bChương\s+(\d+)(?:\.(\d+))?")
H2_RE = re.compile(r"^##\s+(\d+)\.(\d+)\s*(.*)$")
H3_RE = re.compile(r"^###\s+(\d+)\.(\d+)\.(\d+)\s*(.*)$")
# Mục con viết dạng bold: `**7.3.1 Tiêu đề**` (chỉ trong Chương 7 cũ)
BOLD_SUB_RE = re.compile(r"^\*\*(\d+)\.(\d+)\.(\d+)\s+(.+?)\*\*\s*$")


def split_body(path: Path) -> tuple[str, str]:
    """Tách tiêu đề chương (dòng `# ...`) khỏi phần thân.

    Args:
        path: file markdown.

    Returns:
        Cặp `(phần thân, tiêu đề chương)`.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).strip("\n"), lines[0]
    return "\n".join(lines).strip("\n"), ""


def build_section_map() -> dict[tuple[int, int, int], tuple[int, int, int]]:
    """Tính ánh xạ mục cũ -> mục mới.

    Mục lớn (`## x.y`) của mỗi chương mới được đánh số **liên tục** theo thứ tự
    xuất hiện. Mục con (`### x.y.z` hoặc `**x.y.z ...**`) được đánh số lại
    **liên tục trong mỗi mục lớn** — cần thiết vì khi gộp hai chương cũ, hai mục con
    cùng số thứ tự có thể trùng nhau.

    Returns:
        Dict `(ch, mục, mục con)` cũ -> `(ch, mục, mục con)` mới.
        Mục con dùng 0 khi mục không có mục con.
    """
    mapping: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for idx, (_new_name, old_names, _title) in enumerate(CHAPTER_MAP, start=1):
        major = 0
        for old_name in old_names:
            path = REPORT / old_name
            if not path.exists():
                continue
            sub = 0
            for line in path.read_text(encoding="utf-8").split("\n"):
                m2 = H2_RE.match(line)
                if m2:
                    major += 1
                    sub = 0
                    mapping[(int(m2.group(1)), int(m2.group(2)), 0)] = \
                        (idx, major, 0)
                    continue
                m3 = H3_RE.match(line)
                if m3:
                    sub += 1
                    mapping[(int(m3.group(1)), int(m3.group(2)),
                             int(m3.group(3)))] = (idx, major, sub)
                    continue
                mb = BOLD_SUB_RE.match(line)
                if mb:
                    sub += 1
                    mapping[(int(mb.group(1)), int(mb.group(2)),
                             int(mb.group(3)))] = (idx, major, sub)
    return mapping


def renumber_headings(text: str, new_ch: int) -> str:
    """Đánh số lại heading trong một chương mới.

    Giữ nguyên định dạng gốc: `###` vẫn là `###`, mục con viết dạng bold vẫn
    giữ dạng bold — chỉ đổi **số** ở đầu.

    Args:
        text: phần thân của chương (đã gộp các file cũ, có dấu `---` ngăn cách).
        new_ch: số chương mới.

    Returns:
        Văn bản đã đánh số lại heading `##` / `###` / mục con bold.
    """
    out = []
    major = 0
    sub = 0
    for line in text.split("\n"):
        m2 = H2_RE.match(line)
        if m2:
            major += 1
            sub = 0
            out.append(f"## {new_ch}.{major} {m2.group(3)}".rstrip())
            continue
        m3 = H3_RE.match(line)
        if m3:
            sub += 1
            out.append(f"### {new_ch}.{major}.{sub} {m3.group(4)}".rstrip())
            continue
        mb = BOLD_SUB_RE.match(line)
        if mb:
            sub += 1
            out.append(f"**{new_ch}.{major}.{sub} {mb.group(4)}**")
            continue
        out.append(line)
    return "\n".join(out)


def rewrite_refs(text: str, mapping: dict) -> str:
    """Đổi mọi cross-reference `§x.y[.z]` theo ánh xạ.

    Args:
        text: nội dung markdown.
        mapping: ánh xạ từ `build_section_map`.

    Returns:
        Nội dung đã đổi tham chiếu.
    """
    def repl(m: re.Match) -> str:
        ch, sec, sub = int(m.group(1)), int(m.group(2)), m.group(3)
        key = (ch, sec, int(sub) if sub else 0)
        target = mapping.get(key)
        if target is None:
            return m.group(0)
        nch, nsec, nsub = target
        return f"§{nch}.{nsec}.{nsub}" if sub else f"§{nch}.{nsec}"

    return SECTION_REF_RE.sub(repl, text)


def rewrite_chapter_refs(text: str, mapping: dict) -> str:
    """Đổi tham chiếu `Chương N` và `Chương N.x` theo ánh xạ.

    Args:
        text: nội dung markdown.
        mapping: ánh xạ mục từ `build_section_map`.

    Returns:
        Nội dung đã đổi.
    """
    def repl(m: re.Match) -> str:
        old_ch = int(m.group(1))
        old_sec = m.group(2)
        new_ch = CHAPTER_MAP_NUM.get(old_ch, old_ch)
        if old_sec is None:
            return f"Chương {new_ch}"
        key = (old_ch, int(old_sec), 0)
        target = mapping.get(key)
        if target is None:
            return f"Chương {new_ch}.{old_sec}"
        return f"Chương {target[0]}.{target[1]}"

    return CHAPTER_REF_RE.sub(repl, text)


def rewrite_chapter_section_refs(text: str) -> str:
    """Đổi tham chiếu cấp chương `§5` sang chương mới.

    Args:
        text: nội dung markdown.

    Returns:
        Nội dung đã đổi.
    """
    return CHAPTER_SECTION_REF_RE.sub(
        lambda m: f"§{CHAPTER_MAP_NUM.get(int(m.group(1)), int(m.group(1)))}",
        text)


def main() -> None:
    """Chạy restructure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="chỉ in ra kế hoạch, không ghi file")
    args = parser.parse_args()

    mapping = build_section_map()

    print("=== Ánh xạ mục cũ -> mới ===")
    for (och, osec, osub), (nch, nsec, nsub) in sorted(mapping.items()):
        if osec == 0:
            continue
        old = f"{och}.{osec}" + (f".{osub}" if osub else "")
        new = f"{nch}.{nsec}" + (f".{nsub}" if nsub else "")
        flag = "" if old == new else "  <- đổi"
        print(f"  §{old} -> §{new}{flag}")

    if args.dry_run:
        print("\n(dry-run — không ghi file)")
        return

    # --- 1. Gộp file, đánh số lại heading
    for idx, (new_name, old_names, title) in enumerate(CHAPTER_MAP, start=1):
        chunks = []
        for old_name in old_names:
            path = REPORT / old_name
            if not path.exists():
                continue
            body, _ = split_body(path)
            chunks.append(body)
        merged_body = "\n\n---\n\n".join(chunks)
        merged = renumber_headings(merged_body, idx)
        out_path = REPORT / new_name
        out_path.write_text(f"# {title}\n\n{merged}\n", encoding="utf-8")
        print(f"  ghi {new_name} ({len(merged.splitlines())} dòng)")

    # --- 2. Đổi cross-reference trong TẤT CẢ file báo cáo
    #
    # Chỉ áp dụng cho **phần thân**; dòng tiêu đề chương (`# Chương N — ...`)
    # được viết lại ở bước 1 sau, để không bị đổi số chương sai.
    for path in sorted(REPORT.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        head, sep, body = text.partition("\n")
        if sep:
            new = rewrite_chapter_section_refs(
                rewrite_refs(rewrite_chapter_refs(body, mapping), mapping))
            if new != body:
                path.write_text(head + sep + new, encoding="utf-8")
                print(f"  cập nhật {path.name}")

    # --- 3. Xoá file cũ đã gộp
    for old_name in OBSOLETE:
        path = REPORT / old_name
        if path.exists():
            path.unlink()
            print(f"  xoá {old_name}")

    # --- 4. Viết lại tiêu đề chương (sau cross-ref để không bị đổi sai)
    for idx, (new_name, _old_names, title) in enumerate(CHAPTER_MAP, start=1):
        path = REPORT / new_name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        if lines and lines[0].startswith("# "):
            lines[0] = f"# {title}"
            path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  tiêu đề {new_name}: {title}")

    print("\nXong.")


if __name__ == "__main__":
    main()
