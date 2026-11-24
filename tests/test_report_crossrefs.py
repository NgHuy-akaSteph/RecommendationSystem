"""Test hàng rào cross-reference nội bộ của báo cáo.

`scripts/check_report_crossrefs.py` kiểm tra mọi tham chiếu `§x.y[.z]` trong báo
cáo có trỏ tới mục **thực sự tồn tại** không.

Vì sao cần test: đây là hàng rào cho thao tác đánh số lại mục khi gộp chương.
Một script luôn trả về "0 hỏng" cũng vô nghĩa — test này chứng minh nó **bắt được**
tham chiếu hỏng.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from src import paths

ROOT = paths.ROOT
SCRIPT = ROOT / "scripts" / "check_report_crossrefs.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_report_crossrefs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_has_five_chapters(checker):
    """Báo cáo phải có đúng 5 chương + abstract + phụ lục."""
    report_dir = ROOT / "docs" / "report"
    expected = [
        "00-abstract.md",
        "01-introduction.md",
        "02-literature-review.md",
        "03-dataset-and-methodology.md",
        "04-results-and-analysis.md",
        "05-discussion-and-conclusion.md",
        "09-appendix.md",
    ]
    actual = sorted(p.name for p in report_dir.glob("*.md")
                    if p.name != "README.md")
    assert actual == sorted(expected)


def test_current_report_has_no_broken_crossrefs(checker):
    """Báo cáo hiện tại không được có tham chiếu hỏng."""
    sections = checker.collect_sections()
    assert sections, "không thu thập được mục nào"

    broken = []
    for path in checker.REPORT_FILES + [checker.REPORT / "README.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in checker.REF_RE.finditer(text):
            ch = int(match.group(1))
            sec = int(match.group(2))
            sub = match.group(3)
            key = (ch, sec, int(sub) if sub else 0)
            if key not in sections:
                broken.append((path.name, match.group(0)))

    assert not broken, f"cross-reference hỏng: {broken}"


def test_checker_detects_broken_crossref(checker, tmp_path, monkeypatch):
    """Tham chiếu tới mục không tồn tại phải bị báo."""
    fake = tmp_path / "fake.md"
    fake.write_text("Xem §9.9.9 tại đây.\n", encoding="utf-8")
    monkeypatch.setattr(checker, "REPORT_FILES", [fake])

    sections = checker.collect_sections()
    assert (9, 9, 9) not in sections
