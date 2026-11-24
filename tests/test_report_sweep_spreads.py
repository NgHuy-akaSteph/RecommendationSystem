"""Test biên độ sweep khớp với `results/final/ablation-*.csv`.

Đây là test hồi quy cho lỗi tài liệu đã xảy ra thật: `notes/report-structure.md` §5.2
gộp hai quy ước biên độ — "CF 17,94%" (biên độ riêng lưới ALS, mẫu số max) đứng cạnh
"CB-1 96,06%" (biên độ **toàn nhóm**). Người đọc tưởng CF sweep biến thiên 17,94%
trong khi thực tế toàn nhóm CF là 73,65%.

Test ở đây **tính động** từ CSV và so với những gì tài liệu viết — thêm config mới
mà quên cập nhật khung báo cáo thì test đỏ.
"""

import re

import pandas as pd
import pytest

from src import paths
from src.eval import sweep_spread

REPORT_DIR = paths.ROOT / "docs" / "report"
OUTLINE = paths.ROOT / "notes" / "report-structure.md"
CH4 = REPORT_DIR / "04-results-and-analysis.md"

# Tài liệu được phép nêu biên độ sweep
SPREAD_DOCS = [OUTLINE, CH4]

TOLERANCE = 0.01  # điểm %


def _text(path) -> str:
    return path.read_text(encoding="utf-8")


def _percent_strings(text: str) -> set[str]:
    """Thu thập mọi chuỗi phần trăm dạng `12,34%` trong văn bản (bỏ ký hiệu `%`)."""
    return {
        m.group(0).replace(" ", "").rstrip("%")
        for m in re.finditer(r"\d+,\d{2}\s*%", text)
    }


@pytest.fixture(scope="module")
def by_model():
    return sweep_spread.spread_by_model()


@pytest.fixture(scope="module")
def by_group():
    return sweep_spread.spread_by_group()


def test_spread_convention_is_max_denominator():
    """Quy ước biên độ phải là `(max − min) / max` — không phải mẫu số trung bình.

    Lý do: repo đã từng có hai quy ước cùng tồn tại (17,94% mẫu số max vs 18,65%
    mẫu số trung bình) khiến hai tài liệu nói khác nhau. Chốt một quy ước duy nhất.
    """
    values = pd.Series([0.113470, 0.138270])
    assert sweep_spread.spread_of(values) == pytest.approx(17.936, abs=0.01)
    # Mẫu số trung bình sẽ ra 18,65% — phải KHÁC, để test bắt được nhầm lẫn
    assert sweep_spread.spread_of(values) != pytest.approx(18.648, abs=0.01)


def test_als_spread_is_17_94_percent(by_model):
    """Biên độ riêng lưới ALS (15 config) là 17,94%."""
    assert "cf-als" in by_model
    als = by_model["cf-als"]
    assert als.n == 15
    assert als.spread_pct == pytest.approx(17.936, abs=TOLERANCE)
    assert sweep_spread.format_vn(als.spread_pct, 2) == "17,94"


def test_cf_group_spread_is_73_65_percent(by_group):
    """Biên độ **toàn nhóm CF** (51 config) là 73,65% — khác hẳn lưới ALS."""
    assert "CF" in by_group
    cf = by_group["CF"]
    assert cf.n == 51
    assert cf.spread_pct == pytest.approx(73.65, abs=TOLERANCE)
    assert sweep_spread.format_vn(cf.spread_pct, 2) == "73,65"


def test_model_and_group_spreads_differ(by_model, by_group):
    """Bảng theo model và theo nhóm phải cho số khác nhau — cấm trộn.

    Đây là bản chất của lỗi đã xảy ra: tài liệu dùng "CF 17,94%" (theo model)
    để mô tả biên độ của nhóm CF (73,65%).
    """
    assert by_model["cf-als"].spread_pct != by_group["CF"].spread_pct


def test_cb4_group_contains_time_decay_rows():
    """`ablation-cb4.csv` có 3 dòng `model=cb1-tfidf` (họ time-decay).

    Đây là cái bẫy đã ghi ở khung báo cáo: biên độ nhóm CB-4 vẫn đúng vì 32/35 dòng
    là `cb4-hybrid`, nhưng bảng theo model phải nói rõ.
    """
    df = pd.read_csv(paths.FINAL_DIR / "ablation-cb4.csv")
    assert len(df) == 35
    assert (df["model"] == "cb1-tfidf").sum() == 3
    assert (df["model"] == "cb4-hybrid").sum() == 32


def test_als_reg_iterations_slice_still_0_40_percent():
    """Lát cắt `reg` × `iterations` vẫn là 0,40% — số cũ chỉ đúng cho lát cắt này."""
    assert sweep_spread.als_reg_iterations_spread() == pytest.approx(0.399, abs=0.01)


@pytest.mark.parametrize("path", SPREAD_DOCS, ids=lambda p: p.name)
def test_outline_states_both_spread_conventions(path, by_model, by_group):
    """Tài liệu phải nêu **cả hai** bảng biên độ, không trộn.

    Ngưỡng ± 0,01 điểm % để tránh fail vô lý khi làm tròn.
    """
    text = _text(path)
    present = _percent_strings(text)

    # Bảng theo model: 9 model, mỗi model một biên độ
    expected_model = {
        sweep_spread.format_vn(sp.spread_pct, 2) for sp in by_model.values()
    }
    # Bảng theo nhóm: CF, CB-1…CB-4
    expected_group = {
        sweep_spread.format_vn(sp.spread_pct, 2) for sp in by_group.values()
    }
    expected = expected_model | expected_group
    missing = expected - present
    assert not missing, (
        f"{path.name}: thiếu biên độ sweep {sorted(missing)} "
        f"(tính từ results/final/ablation-*.csv)"
    )


@pytest.mark.parametrize("path", SPREAD_DOCS, ids=lambda p: p.name)
def test_outline_states_spread_formula_and_protocol(path):
    """Mọi biên độ sweep phải kèm công thức và protocol `dev` (20.000 user)."""
    text = _text(path)
    assert re.search(r"\(max\s*[−–-]\s*min\)\s*/\s*max", text), \
        f"{path.name}: thiếu công thức biên độ (max−min)/max"
    assert "dev" in text, f"{path.name}: thiếu nhãn protocol dev"
    assert re.search(r"20\.?000|20,000", text), \
        f"{path.name}: thiếu số user của protocol dev"


@pytest.mark.parametrize("path", SPREAD_DOCS, ids=lambda p: p.name)
def test_outline_does_not_mix_model_and_group_spread_labels(path):
    """Cấm việc gắn nhãn nhóm CF với biên độ của lưới ALS (17,94%).

    Đây chính là lỗi đã xảy ra: "CF 17,94%" — 17,94% là biên độ của `cf-als`,
    không phải của nhóm CF (73,65%).
    """
    text = _text(path)
    for match in re.finditer(r"[^\n]*\bCF\b[^\n]*17,94\s*%[^\n]*", text):
        line = match.group(0)
        # Được phép nếu dòng đang **phủ nhận** nhầm lẫn (đính chính lịch sử)
        if re.search(r"không được trộn|phủ nhận|đính chính|sai|là biên độ của",
                     line, re.IGNORECASE):
            continue
        pytest.fail(
            f"{path.name}: gắn biên độ lưới ALS (17,94%) cho nhóm CF:\n  {line.strip()}"
        )
