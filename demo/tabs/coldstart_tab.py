"""Tab 3 — Cold-start item: màn hình trung tâm khi bảo vệ.

Giao thức cold-start giữ lại 1.381 phim (10% catalog, phân tầng theo 5 nhóm năm
phát hành) và **xoá toàn bộ tương tác train** của các phim đó. Hệ quả đo được:

    - mọi mô hình CF cho điểm **đúng bằng 0** với các phim held-out — không phải
      "điểm thấp" mà là không có tín hiệu nào để học;
    - các mô hình content vẫn xếp hạng được, vì chúng dựa vào **nội dung phim** chứ
      không vào tương tác.

Đây là luận điểm trung tâm của đồ án, và tab này cho thấy nó **trực quan** thay vì
chỉ nằm trên bảng số liệu.
"""
from __future__ import annotations

import streamlit as st

from demo.loader import MODEL_LABELS, get_service


def render() -> None:
    service = get_service()
    cold_items = service.coldstart_items(limit=200)

    st.markdown(
        "**Giao thức cold-start:** 1.381 phim (10% catalog) bị giữ lại theo 5 nhóm năm "
        "phát hành, và **mọi tương tác train của các phim đó bị xoá**. "
        "Hệ thống được hỏi: *với một phim chưa có ai xem, mô hình có xếp hạng được không?*"
    )

    if not cold_items:
        st.error(
            "Thiếu `data/processed/coldstart/heldout_items.npy`. "
            "Chạy: venv/bin/python -m src.eval.cold_start_item_split"
        )
        return

    labels = [f"{c['title']} ({c['movie_id']})" for c in cold_items]
    selected = st.selectbox("Chọn một phim cold-start", labels)
    chosen = cold_items[labels.index(selected)]

    st.markdown(
        f"**Phim được hỏi:** {chosen['title']} · {chosen['genres']} · "
        f"không có tương tác nào trong tập train."
    )

    n = st.slider("Số gợi ý", min_value=5, max_value=20, value=10, key="cold_n")
    result = service.coldstart_movies_for(chosen["item_idx"], n=n)

    if not result.get("available", False):
        st.warning(
            "Chưa có artifact cho split cold-start. Sinh bằng: "
            "`venv/bin/python scripts/save_demo_artifacts.py --split coldstart`"
        )
        return

    user_id = next(
        (v["user_id"] for v in result["models"].values() if v.get("user_id")), None
    )
    st.caption(
        f"Đo trên user `{user_id}` — user có lịch sử train dày nhất trong tập đánh giá. "
        "Các mô hình ở đây **được fit trên split cold-start**, nên chúng không hề thấy "
        "tương tác nào của phim đang xét."
    )

    for model_key, payload in result["models"].items():
        st.subheader(MODEL_LABELS.get(model_key, model_key))
        score = payload["cold_item_score"]
        rank = payload["cold_item_rank"]

        if score == 0.0:
            st.error(
                f"**Điểm của phim cold-start = {score:.6f}** — đúng bằng không, không phải "
                "“điểm thấp”. Mô hình này chỉ dựa vào tương tác cộng tác, mà phim "
                "cold-start **không có tương tác nào** trong tập train, nên không có gì "
                "để học. Phim này không thể lọt vào top-N (thứ hạng "
                f"{rank:,} — bằng nhau ở đáy bảng)."
            )
        else:
            st.success(
                f"**Điểm của phim cold-start = {score:.6f}**, thứ hạng **#{rank:,}** trong "
                "bảng xếp hạng của user — nhờ dựa vào nội dung (thể loại + tag) của "
                "chính phim đó, không cần ai đã xem nó trước."
            )

        st.dataframe(
            [{"#": i + 1, "Phim": r["title"], "Năm": r["year"],
              "Thể loại": r["genres"], "Điểm": round(r["score"], 4)}
             for i, r in enumerate(payload["items"])],
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.info(
        "**Cách đọc kết quả:** đây không phải “CF kém” — CF **không có gì để học** "
        "với phim mới. Vì vậy hệ thống thật cần **kết hợp**: CF cho lịch sử dày, "
        "content-based cho phim mới. Báo cáo đo điều này ở giao thức `coldstart`: "
        "mọi mô hình CF đạt NDCG@10 = 0,0000, còn CB-4 đạt 0,0347 (p < 1e-190)."
    )
