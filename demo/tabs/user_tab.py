"""Tab 1 — Gợi ý theo người dùng: đặt ALS (CF) cạnh BM25 (content-based).

Màn hình này trả lời RQ1 của báo cáo *bằng cách nhìn*: với một user cụ thể, CF và
CB đều ra gợi ý hợp lý nhưng **không trùng nhau** — minh hoạ trực quan rằng hai
hướng tiếp cận học tín hiệu khác nhau (tương tác vs nội dung).
"""
from __future__ import annotations

import streamlit as st

from demo.loader import MODEL_LABELS, get_service


def render() -> None:
    service = get_service()
    known = service.known_user_ids()

    st.markdown(
        "Nhập `userId` của MovieLens 25M. Hai bảng cạnh nhau: **ALS** học từ tương tác "
        "của những người dùng giống bạn; **BM25** học từ nội dung phim (thể loại + tag) "
        "nên vẫn gợi ý được cả phim bạn chưa từng xem."
    )

    col_input, col_hint = st.columns([1, 2])
    with col_input:
        user_id = st.number_input(
            "userId", min_value=1, max_value=162_540, value=known[0], step=1
        )
    with col_hint:
        st.caption("User có nhiều tương tác nhất: " + ", ".join(str(u) for u in known[:5]))
        n = st.slider("Số phim gợi ý", min_value=5, max_value=20, value=10)

    history = service.user_history(int(user_id), limit=12)
    if not history:
        st.warning(f"User {int(user_id)} không có lịch sử tương tác trong tập train.")
    else:
        with st.expander(f"Lịch sử xem gần đây ({len(history)} phim)", expanded=False):
            st.dataframe(
                [{"Phim": h["title"], "Năm": h["year"], "Thể loại": h["genres"],
                  "Rating": h["rating"]} for h in history],
                use_container_width=True, hide_index=True,
            )

    left, right = st.columns(2)
    for column, model_key in ((left, "cf-als"), (right, "cb2-bm25")):
        with column:
            st.subheader(MODEL_LABELS[model_key])
            recommendations = service.recommend_by_user(int(user_id), model_key, n=n)
            if not recommendations:
                st.info("Mô hình không trả được gợi ý nào cho user này.")
                continue
            st.dataframe(
                [{"#": i + 1, "Phim": r.title, "Năm": r.year, "Thể loại": r.genres,
                  "Điểm": round(r.score, 4), "Lý do": r.reason or "—"}
                 for i, r in enumerate(recommendations)],
                use_container_width=True, hide_index=True,
            )

    cf_titles = {r.title for r in service.recommend_by_user(int(user_id), "cf-als", n=n)}
    cb_titles = {r.title for r in service.recommend_by_user(int(user_id), "cb2-bm25", n=n)}
    only_cf = cf_titles - cb_titles
    only_cb = cb_titles - cf_titles
    overlap = cf_titles & cb_titles

    st.markdown("---")
    st.markdown(
        f"**So sánh:** {len(overlap)} phim hai bên cùng gợi ý · "
        f"{len(only_cf)} chỉ ALS · {len(only_cb)} chỉ BM25."
    )
    if only_cf:
        st.caption("Chỉ ALS gợi ý: " + "; ".join(sorted(only_cf)[:5]))
    if only_cb:
        st.caption("Chỉ BM25 gợi ý: " + "; ".join(sorted(only_cb)[:5]))
