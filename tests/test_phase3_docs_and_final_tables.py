"""Test chống hồi quy cho hai lỗi tài liệu + script vừa sửa.

Hai lỗi đã xảy ra thật và không test nào bắt được:

    - Tài liệu ghi "ALS không nhạy với tham số, biên độ 0,40%" — con số đó chỉ đúng cho
      lát cắt `reg × iterations`. Toàn lưới (có thêm `α`, `factors`) biên độ 18,65%.
      Nếu tài liệu lại quay về tuyên bố cũ thì người đọc tưởng ALS đã hội tụ.
    - `scripts/build_final_tables.py` crash khi thiếu `results/dev-summary.csv` và ghi
      đè `results/final/benchmark-*.csv` — mất bằng chứng protocol (`split_id`,
      `catalog_id`) mà phase 03 đã tạo ra.
"""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import paths

ROOT = paths.ROOT
README = ROOT / "README.md"
CHANGELOG = ROOT / "docs" / "project-changelog.md"
PHASE3 = (ROOT / "plans" / "261113-2106-unify-recsys-and-recommendation-cb"
          / "phase-03-run-shared-benchmark-tables.md")
BUILD_SCRIPT = ROOT / "scripts" / "build_final_tables.py"

DOCS = [README, CHANGELOG, PHASE3]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── tài liệu: không được quay lại tuyên bố cũ ───────────────────────────

@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_do_not_claim_als_is_insensitive(path):
    """Cấm câu tuyên bố ALS không nhạy với tham số.

    Biên độ 0,40% chỉ đúng cho lát cắt `reg × iterations`; toàn lưới là 18,65%.
    Nếu tài liệu ghi "ALS không nhạy" mà không kèm lát cắt thì đọc sai.
    """
    text = _text(path)
    for match in re.finditer(r"[^\n]*0[,.]4\s*%[^\n]*", text):
        line = match.group(0)
        # Được phép nếu dòng nói rõ đó là lát cắt hẹp, không phải toàn lưới
        if re.search(r"lát cắt|cut|reg\s*[×x]\s*iterations|3 config|0[,.]29\s*%",
                     line, re.IGNORECASE):
            continue
        # Được phép nếu dòng đang **phủ nhận** tuyên bố cũ (viết để đính chính)
        if re.search(r"trong\s+18\s+config|top-?18|8[,.]85\s*%", line):
            continue
        if re.search(r"cũng sai|sai\b|đính chính|không đúng|chỉ đúng cho|chỉ là biên độ",
                     line, re.IGNORECASE):
            continue
        pytest.fail(f"{path.name}: tuyên bố biên độ 0,4% không nói rõ lát cắt:\n  {line.strip()}")


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_state_full_als_spread(path):
    """Mọi tài liệu phải nêu biên độ toàn lưới ALS 18,65%."""
    assert "18,65%" in _text(path), f"{path.name}: thiếu biên độ toàn lưới ALS 18,65%"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_state_real_sweep_counts(path):
    """Số config thực tế đã chạy phải là CF 51 / CB 85 (sweep đã chạy đủ).

    Trước đây là CF 49 / CB 8 — cây tài liệu cảnh báo "so sánh chưa cân bằng vì CF
    được quét nhiều hơn". Sau khi sweep CB chạy đủ 85 config thì câu cảnh báo đó không
    còn đúng, và tài liệu phải nói số mới.
    """
    text = _text(path)
    assert re.search(r"CF\s*51\s*/\s*CB\s*85|CF\s*51", text), \
        f"{path.name}: thiếu số config thực tế CF 51 / CB 85"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_do_not_claim_sweep_is_unbalanced(path):
    """Cấm tuyên bố còn sót 'CB 85 config chưa chạy' — sweep đã chạy đủ.

    Chỉ cho phép nếu dòng đó nằm trong phần **lịch sử/đính chính** (ghi rõ là trạng
    thái trước đây), vì changelog cần giữ lại câu cảnh báo cũ để người đọc hiểu vì
    sao kết luận đã đổi.
    """
    text = _text(path)
    for match in re.finditer(r"[^\n]*(?:85\s*config\s*CB\s*còn\s*lại|CB\s*chỉ\s*mới\s*chạy|chưa\s*chạy)[^\n]*", text):
        line = match.group(0)
        if re.search(r"trước|đã chạy đủ|cũ|đính chính|lịch sử|thiên lệch|hồi quy|đảo|"
                     r"tạm dừng|trạng thái tại thời điểm",
                     line, re.IGNORECASE):
            continue
        # Dòng mô tả **số liệu cũ** (CF 49 / CB 8) nằm trong phần lịch sử đính chính:
        # nó là bằng chứng vì sao kết luận đã đổi, không phải tuyên bố về hiện trạng.
        # Dấu hiệu nhận biết: nhắc con số 49 (số config CF của thời điểm đó).
        if re.search(r"\b49\b", line):
            continue
        pytest.fail(f"{path.name}: còn tuyên bố sweep CB chưa chạy đủ:\n  {line.strip()}")


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_do_not_claim_the_old_cf18_cb103_balance(path):
    """'CF 18 vs CB 103' là số của repo CB, không phải của repo này."""
    text = _text(path)
    for match in re.finditer(r"[^\n]*(?:CF\s*18|18\s*vs\s*103|CB\s*103)[^\n]*", text):
        line = match.group(0)
        # Được phép nếu dòng ghi rõ đó là con số của repo CB bị đảo
        if re.search(r"đảo|repo CB|ngược|sai|đính chính|số của CB", line, re.IGNORECASE):
            continue
        pytest.fail(f"{path.name}: còn tuyên bố CF 18 / CB 103 mà không đính chính:\n  {line.strip()}")


# ── script build_final_tables ───────────────────────────────────────────

def test_build_script_never_overwrites_phase3_benchmark_tables():
    """Guard bên trong script: file `benchmark-*`/`wilcoxon-*` của phase 03 là bất khả xâm phạm."""
    text = _text(BUILD_SCRIPT)
    assert "PROTECTED" in text
    assert "_guard_not_protected" in text


def test_build_script_does_not_read_dev_summary_from_disk():
    """Bản cũ đọc thẳng `results/dev-summary.csv` → FileNotFoundError khi file thiếu."""
    text = _text(BUILD_SCRIPT)
    assert 'paths.RESULTS_DIR / "dev-summary.csv"' not in text
    assert "pd.read_csv(paths.RESULTS_DIR" not in text


def test_build_script_catches_missing_data_instead_of_crashing():
    """Mọi lỗi 'thiếu dữ liệu' phải thành `_Missing` chứ không phải exception thô."""
    text = _text(BUILD_SCRIPT)
    assert "FileNotFoundError" in text
    assert "_Missing" in text


@pytest.mark.slow
def test_build_script_runs_end_to_end_without_crashing(tmp_path):
    """Chạy thật script trong thư mục làm việc tạm, xác nhận không crash và không
    ghi đè `benchmark-*.csv` của phase 03."""
    backup = {}
    final_dir = paths.RESULTS_DIR / "final"
    protected = sorted(final_dir.glob("benchmark-*.csv"))
    for p in protected:
        backup[p] = p.read_bytes()

    proc = subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for p in protected:
        assert p.read_bytes() == backup[p], f"{p.name} đã bị ghi đè bởi build_final_tables"
