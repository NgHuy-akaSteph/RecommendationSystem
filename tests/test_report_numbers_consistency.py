"""Test script đối chiếu số liệu báo cáo ↔ `results/final/`.

`scripts/check_report_numbers.py` là hàng rào chống **trôi số** khi viết báo cáo
Markdown bằng tay. Script này kiểm chứng rằng hàng rào đó thật sự bắt được lỗi —
nếu không, một script luôn trả về "0 lệch" cũng vô nghĩa.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from src import paths

ROOT = paths.ROOT
SCRIPT = ROOT / "scripts" / "check_report_numbers.py"
CH3 = ROOT / "docs" / "report" / "03-dataset-and-methodology.md"
CH4 = ROOT / "docs" / "report" / "04-results-and-analysis.md"
CH5 = ROOT / "docs" / "report" / "05-discussion-and-conclusion.md"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_report_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_files_exist():
    """Ba chương đã gộp phải tồn tại ở `docs/report/`."""
    assert CH3.exists()
    assert CH4.exists()
    assert CH5.exists()


def test_current_reports_have_zero_mismatches(checker):
    """Báo cáo hiện tại phải sạch — 0 lệch so với `results/final/`."""
    allowed = checker._collect_allowed(paths.FINAL_DIR)
    derived = checker._derived_values()
    problems = checker.check(allowed, derived)
    assert not problems, f"số trong báo cáo không khớp results/final/: {problems}"


def test_checker_catches_injected_wrong_number(checker, tmp_path, monkeypatch):
    """Sửa một số trong báo cáo → checker phải bắt được.

    Đây là phép thử quan trọng nhất: nếu checker không bắt được lỗi thì "0 lệch"
    không mang ý nghĩa gì.
    """
    fake = tmp_path / "05-experimental-results.md"
    fake.write_text(
        "# Bản giả\n\n| ALS | 0,1375 |\n| Sai | 0,9999 |\n", encoding="utf-8"
    )
    monkeypatch.setattr(checker, "REPORT_FILES", [fake])

    allowed = checker._collect_allowed(paths.FINAL_DIR)
    derived = checker._derived_values()
    problems = checker.check(allowed, derived)
    assert problems, "checker không bắt được số sai 0,9999"
    assert any(token == "0,9999" for _, token, _ in problems)


def test_checker_accepts_numbers_present_in_csv(checker, tmp_path, monkeypatch):
    """Số có thật trong CSV phải được chấp nhận (không báo lệch nhầm)."""
    fake = tmp_path / "05-experimental-results.md"
    fake.write_text(
        "# Bản giả\n\n| ALS | 0,1375 |\n| CI | 0,136620 |\n", encoding="utf-8"
    )
    monkeypatch.setattr(checker, "REPORT_FILES", [fake])

    allowed = checker._collect_allowed(paths.FINAL_DIR)
    derived = checker._derived_values()
    assert not checker.check(allowed, derived)


def test_checker_accepts_negative_numbers(checker, tmp_path, monkeypatch):
    """Số âm trong báo cáo phải khớp ô số âm trong CSV."""
    fake = tmp_path / "06-leakage-analysis-and-discussion.md"
    fake.write_text(
        "# Bản giả\n\n| centered | −0,022823 |\n", encoding="utf-8"
    )
    monkeypatch.setattr(checker, "REPORT_FILES", [fake])

    allowed = checker._collect_allowed(paths.FINAL_DIR)
    derived = checker._derived_values()
    assert not checker.check(allowed, derived)


def test_checker_accepts_percent_form(checker, tmp_path, monkeypatch):
    """Coverage ghi dạng phần trăm (`14,94%`) phải khớp giá trị 0,149392 trong CSV."""
    fake = tmp_path / "06-leakage-analysis-and-discussion.md"
    fake.write_text(
        "# Bản giả\n\n| ALS | 14,94% |\n", encoding="utf-8"
    )
    monkeypatch.setattr(checker, "REPORT_FILES", [fake])

    allowed = checker._collect_allowed(paths.FINAL_DIR)
    derived = checker._derived_values()
    assert not checker.check(allowed, derived)


def test_checker_accepts_p_values(checker, tmp_path, monkeypatch):
    """p-value dạng ký hiệu khoa học phải được đối chiếu theo giá trị thực."""
    fake = tmp_path / "05-experimental-results.md"
    fake.write_text(
        "# Bản giả\n\n| CB-4 vs ALS | p = 8,77e-201 |\n", encoding="utf-8"
    )
    monkeypatch.setattr(checker, "REPORT_FILES", [fake])

    allowed = checker._collect_allowed(paths.FINAL_DIR)
    derived = checker._derived_values()
    assert not checker.check(allowed, derived)


def test_checker_catches_wrong_p_value(checker, tmp_path, monkeypatch):
    """p-value không tồn tại trong CSV phải bị bắt."""
    fake = tmp_path / "05-experimental-results.md"
    fake.write_text(
        "# Bản giả\n\n| Kết quả | p = 1,23e-99 |\n", encoding="utf-8"
    )
    monkeypatch.setattr(checker, "REPORT_FILES", [fake])

    allowed = checker._collect_allowed(paths.FINAL_DIR)
    derived = checker._derived_values()
    problems = checker.check(allowed, derived)
    assert problems, "checker không bắt được p-value sai"
    assert any("1,23e-99" in token for _, token, _ in problems)


def test_checker_reports_missing_report_file(checker, tmp_path, monkeypatch):
    """Thiếu file báo cáo phải bị báo, không bị bỏ qua im lặng."""
    monkeypatch.setattr(checker, "REPORT_FILES", [tmp_path / "khong-ton-tai.md"])
    allowed = checker._collect_allowed(paths.FINAL_DIR)
    derived = checker._derived_values()
    problems = checker.check(allowed, derived)
    assert problems
    assert any("không tồn tại" in reason for _, _, reason in problems)
