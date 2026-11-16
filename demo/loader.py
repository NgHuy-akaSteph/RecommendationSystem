"""Nạp artifact + metadata một lần, dùng lại cho mọi lần tương tác của demo.

`@st.cache_resource` giữ đối tượng trong bộ nhớ tiến trình: nạp artifact mất ~1,6s
và giữ 120 MB, nạp lại mỗi lần bấm nút thì demo sẽ giật.

Lưu ý: cache_resource (không phải cache_data) vì đối tượng trả về có vật thể
không serialize được (scipy CSR, numpy arrays) — cache_data sẽ cố pickle và lỗi.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.recommender_service import RecommenderService  # noqa: E402

ARTIFACTS_DIR = ROOT / "demo" / "artifacts"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_KEYS = ["cf-als", "cb2-bm25"]

MODEL_LABELS = {
    "cf-als": "ALS (Collaborative Filtering)",
    "cb2-bm25": "BM25 (Content-based)",
}


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def get_service() -> RecommenderService:
    """Service gợi ý dùng chung cho toàn bộ phiên demo."""
    if not ARTIFACTS_DIR.exists():
        raise FileNotFoundError(
            "Chưa có artifact. Chạy: venv/bin/python scripts/save_demo_artifacts.py"
        )
    return RecommenderService(ARTIFACTS_DIR, MODEL_KEYS, PROCESSED_DIR)
