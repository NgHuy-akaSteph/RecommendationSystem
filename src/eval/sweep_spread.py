"""Tính biên độ sweep từ `results/final/ablation-*.csv`.

Quy ước duy nhất của repo: **biên độ = `(max − min) / max`** trên NDCG@10 của giao
thức `dev`. Hàm này là nguồn sự thật cho mọi bảng biên độ viết trong báo cáo và
được test `tests/test_report_sweep_spreads.py` dùng lại — không chép tay số.

Tách theo **model** (mỗi model là một lưới tham số riêng) và theo **nhóm**
(toàn bảng ablation của nhóm). Hai cách này cho hai con số khác nhau và
**không được trộn** — ví dụ biên độ lưới ALS là 17,94% còn toàn nhóm CF là 73,65%.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import paths

# Nhóm → file ablation tương ứng trong `results/final/`
GROUP_FILES: dict[str, str] = {
    "CF": "ablation-cf.csv",
    "CB-1": "ablation-cb1.csv",
    "CB-2": "ablation-cb2.csv",
    "CB-3": "ablation-cb3.csv",
    "CB-4": "ablation-cb4.csv",
}

# Model → file **sở hữu** nó. Bảng biên độ theo model chỉ gom dòng của file sở hữu.
#
# Lý do: `ablation-cb4.csv` chứa 3 dòng `model=cb1-tfidf` (họ time-decay `cb4-03-*`
# đặt trên nền CB-1, **không phải** hybrid). Nếu gom theo model trên toàn bộ file thì
# `cb1-tfidf` sẽ có 18 dòng thay vì 15 và biên độ lệch — hai bảng trong báo cáo
# (theo model / theo nhóm) phải dùng cùng một cách chia.
MODEL_OWNER: dict[str, str] = {
    "cf-als": "CF",
    "cf-itemknn": "CF",
    "rs-bpr": "CF",
    "rs-itemcf": "CF",
    "cb1-tfidf": "CB-1",
    "cb2-bm25": "CB-2",
    "cb3-genome": "CB-3",
    "cb3-ridge": "CB-3",
    "cb4-hybrid": "CB-4",
}

METRIC = "NDCG@10"


@dataclass(frozen=True)
class SweepSpread:
    """Biên độ sweep của một lưới config.

    Attributes:
        label: nhãn hiển thị (tên model hoặc tên nhóm).
        n: số config trong lưới.
        min_value: NDCG@10 nhỏ nhất.
        max_value: NDCG@10 lớn nhất.
        spread_pct: biên độ theo công thức `(max − min) / max` (%).
        best_exp_name: tên config đạt NDCG@10 cao nhất trong lưới.
    """

    label: str
    n: int
    min_value: float
    max_value: float
    spread_pct: float
    best_exp_name: str = ""


def spread_of(values: pd.Series) -> float:
    """Biên độ `(max − min) / max` theo phần trăm.

    Args:
        values: chuỗi giá trị metric của một lưới config.

    Returns:
        Biên độ tính bằng phần trăm. Trả về 0.0 nếu mọi giá trị bằng 0
        (tránh chia cho 0).
    """
    hi = float(values.max())
    lo = float(values.min())
    if hi <= 0:
        return 0.0
    return (hi - lo) / hi * 100.0


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def spread_by_model(final_dir: Path | None = None) -> dict[str, SweepSpread]:
    """Biên độ theo **model** — mỗi model là một lưới tham số riêng.

    Mỗi model được gom từ **file ablation sở hữu nó** (xem `MODEL_OWNER`), không
    gom trên toàn bộ file. Lý do: `ablation-cb4.csv` chứa 3 dòng `model=cb1-tfidf`
    (họ time-decay `cb4-03-*` đặt trên nền CB-1) — nếu gom theo model trên toàn bộ
    file thì `cb1-tfidf` có 18 dòng thay vì 15 và biên độ lệch khỏi bảng §5.2.
    """
    final_dir = final_dir or paths.FINAL_DIR
    out: dict[str, SweepSpread] = {}
    for group, name in GROUP_FILES.items():
        path = final_dir / name
        if not path.exists():
            continue
        df = _load(path)
        for model, grp in df.groupby("model"):
            # Model thuộc nhóm khác thì bỏ qua — dòng của nó đã được tính ở file sở hữu
            if MODEL_OWNER.get(str(model)) != group:
                continue
            # Config tốt nhất = dòng có NDCG@10 cao nhất trong lưới sở hữu
            best_row = grp.loc[grp[METRIC].idxmax()]
            out[str(model)] = SweepSpread(
                label=str(model),
                n=len(grp),
                min_value=float(grp[METRIC].min()),
                max_value=float(grp[METRIC].max()),
                spread_pct=spread_of(grp[METRIC]),
                best_exp_name=str(best_row.get("exp_name", "")),
            )
    return out


def spread_by_group(final_dir: Path | None = None) -> dict[str, SweepSpread]:
    """Biên độ theo **nhóm** — toàn bảng ablation của nhóm (CF, CB-1…CB-4)."""
    final_dir = final_dir or paths.FINAL_DIR
    out: dict[str, SweepSpread] = {}
    for group, name in GROUP_FILES.items():
        path = final_dir / name
        if not path.exists():
            continue
        df = _load(path)
        best_row = df.loc[df[METRIC].idxmax()]
        out[group] = SweepSpread(
            label=group,
            n=len(df),
            min_value=float(df[METRIC].min()),
            max_value=float(df[METRIC].max()),
            spread_pct=spread_of(df[METRIC]),
            best_exp_name=str(best_row.get("exp_name", "")),
        )
    return out


def als_sensitivity_table(final_dir: Path | None = None) -> pd.DataFrame:
    """Bảng độ nhạy của ALS theo `factors` × `alpha` (từ `ablation-cf.csv`).

    Dùng để minh hoạ: ALS nhạy mạnh với `α` nhưng gần như không nhạy với
    `reg`/`iterations` trong lát cắt cố định `factors=64, α=5`.
    """
    final_dir = final_dir or paths.FINAL_DIR
    df = _load(final_dir / "ablation-cf.csv")
    als = df[df["model"] == "cf-als"].copy()
    als["_params"] = als["params"].apply(json.loads)
    als["factors"] = als["_params"].apply(lambda p: p.get("factors"))
    als["alpha"] = als["_params"].apply(lambda p: p.get("alpha"))
    return als.pivot_table(
        index="factors", columns="alpha", values=METRIC, aggfunc="max"
    ).reset_index()


def als_reg_iterations_spread(final_dir: Path | None = None) -> float:
    """Biên độ lát cắt `reg` × `iterations` (cố định `factors=64, α=5`).

    Đây là lát cắt mà tài liệu cũ gọi nhầm là "biên độ ALS" — chỉ đúng 0,40%.
    """
    final_dir = final_dir or paths.FINAL_DIR
    df = _load(final_dir / "ablation-cf.csv")
    als = df[df["model"] == "cf-als"].copy()
    als["_params"] = als["params"].apply(json.loads)
    mask = als["_params"].apply(
        lambda p: p.get("factors") == 64 and p.get("alpha") == 5.0
    )
    return spread_of(als.loc[mask, METRIC])


def format_vn(value: float, decimals: int = 4) -> str:
    """Định dạng số kiểu Việt (dấu phẩy thập phân) — quy ước của báo cáo."""
    return f"{value:.{decimals}f}".replace(".", ",")
