"""Tab 2 — Phim tương tự, kèm lý do giải thích bằng Tag Genome.

Điểm cộng của content-based: **giải thích được**. Mỗi gợi ý đi kèm các tag
genome chung giữa phim nguồn và phim gợi ý, nên người xem thấy ngay *vì sao* phim
này được gợi ý thay vì phải tin mũ đen.
"""
from __future__ import annotations

import streamlit as st

from demo.loader import get_service


def render() -> None:
    service = get_service()

    st.markdown(
        "Chọn một phim, hệ thống tìm các phim tương tự theo **nội dung** (cosine trên "
        "1.128 chiều Tag Genome) và chỉ ra các tag chung — đó là lý do cụ thể cho gợi ý."
    )

    query = st.text_input("Tìm phim theo tên", placeholder="Ví dụ: matrix, star wars, toy story")

    if not query:
        st.info("Nhập tên phim để xem gợi ý tương tự.")
        return

    hits = service.search_movies(query, limit=100)
    if not hits:
        st.warning(f"Không tìm thấy phim nào chứa “{query}”.")
        return

    titles = [f"{h['title']} ({h['movie_id']})" for h in hits]
    selected = st.selectbox("Chọn phim", titles)
    chosen = hits[titles.index(selected)]

    st.markdown(f"**Nguồn:** {chosen['title']} · {chosen['genres']}")

    n = st.slider("Số phim tương tự", min_value=5, max_value=20, value=10, key="similar_n")
    similar = service.similar_movies(chosen["movie_id"], n=n)

    if not similar:
        st.info("Không tìm được phim tương tự.")
        return

    st.dataframe(
        [{"#": i + 1, "Phim": s.title, "Năm": s.year, "Thể loại": s.genres,
          "Cosine": round(s.score, 4), "Tag chung": s.reason or "—"}
         for i, s in enumerate(similar)],
        use_container_width=True, hide_index=True,
    )

    with st.expander("Vì sao dùng Tag Genome cho phần này?", expanded=False):
        st.markdown(
            "Điểm tương đồng là **cosine giữa hai vector Tag Genome** — 1.128 chiều do "
            "GroupLens tính ngoài động cho mọi phim, phản ánh thể loại, chủ đề, tông "
            "màu. Nhờ vậy gợi ý **có lý do đọc được**: “Star Wars IV” dẫn tới các phim "
            "cùng tag `star wars`, `androids`, `space opera`.\n\n"
            "Đây cũng là điểm phải nói thẳng khi trình bày: `genome-scores.csv` của "
            "GroupLens **không có timestamp** — tức được tính ngoài động trên toàn "
            "corpus, không cắt được theo thời gian. Báo cáo (Chương 6) đo định "
            "lượng rủi ro này bằng cách phân tầng kết quả theo 3 kỷ nguyên người dùng."
        )
