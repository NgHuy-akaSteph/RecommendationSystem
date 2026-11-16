"""Demo Streamlit — hệ thống gợi ý trên MovieLens 25M.

Ba tab, mỗi tab một câu hỏi:

    1. **Gợi ý cho người dùng** — CF và CB trả lời khác nhau, cả hai đều hợp lý.
    2. **Phim tương tự** — content-based và **giải thích được** (tag chung).
    3. **Cold-start** — phim mới chưa có tương tác: CF không có gì để học, CB vẫn
       gợi ý được. Đây là luận điểm trung tâm của đồ án.

Chạy:
    venv/bin/python scripts/save_demo_artifacts.py     # một lần, sinh artifact
    venv/bin/streamlit run demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run demo/app.py` đặt thư mục *cwd* (repo root) vào sys.path, không đặt
# thư mục chứa file đang chạy — nên `import demo.tabs` sẽ hỏng. Chèn repo root vào
# sys.path **trước** mọi import trong package `demo`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.tabs import coldstart_tab, similar_tab, user_tab  # noqa: E402

st.set_page_config(
    page_title="Demo Recommendation System — MovieLens 25M",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Hệ thống gợi ý phim — MovieLens 25M")

with st.sidebar:
    st.markdown(
        "**Cơ sở dữ liệu:** MovieLens 25M · 13.816 phim (lọc theo Tag Genome) · "
        "158.774 người dùng.\n\n"
        "**Mô hình:** ALS (Collaborative Filtering) và BM25 (Content-based), "
        "cùng một giao thức đánh giá full-ranking."
    )
    st.caption(
        "Số liệu trong báo cáo lấy từ `results/final/` — 27 bảng CSV đã commit. "
        "Demo này dùng lại đúng các mô hình đã chạy, không phải bản riêng."
    )

tab_user, tab_similar, tab_cold = st.tabs(
    ["1️⃣ Gợi ý cho người dùng", "2️⃣ Phim tương tự", "3️⃣ Cold-start (phim mới)"]
)

with tab_user:
    user_tab.render()
with tab_similar:
    similar_tab.render()
with tab_cold:
    coldstart_tab.render()
