# Demo Streamlit — hệ thống gợi ý phim

Demo trực tiếp cho **hệ thống gợi ý trên MovieLens 25M**, dùng đúng những mô hình
đã chạy trong báo cáo (không phải bản riêng cho demo).

## Ba tab — mỗi tab một câu hỏi

| Tab | Câu hỏi | Kết quả mong đợi |
|---|---|---|
| **1. Gợi ý cho người dùng** | CF và CB trả lời khác nhau không? | ALS và BM25 đứng cạnh nhau, ít phim trùng |
| **2. Phim tương tự** | Content-based có **giải thích** được không? | Mỗi gợi ý kèm tag genome chung |
| **3. Cold-start** | Phim mới chưa có ai xem thì sao? | ALS điểm **đúng 0,000000**, BM25 xếp hạng được |

Tab 3 là màn hình trung tâm khi bảo vệ: nó biểu diễn trực quan luận điểm trung tâm
của đồ án (kết luận đảo chiều giữa giao thức thường và giao thức cold-start).

## Chạy

```bash
# 1. Sinh artifact (một lần, ~80s)
venv/bin/python scripts/save_demo_artifacts.py                      # split full
venv/bin/python scripts/save_demo_artifacts.py --split coldstart    # split cold-start

# 2. Chạy demo
venv/bin/streamlit run demo/app.py
```

Mở `http://localhost:8501`.

## Yêu cầu

| Thứ | Cần có |
|---|---|
| `data/processed/` | Chạy 3 bước tiền xử lý (xem README gốc) |
| `data/processed/coldstart/` | `venv/bin/python -m src.eval.cold_start_item_split` |
| `demo/artifacts/` | Sinh bằng script ở trên (~230 MB, **không** commit) |

> `data/` và `demo/artifacts/` đều bị `.gitignore` — người clone repo phải chạy lại
> pipeline. Đây là giới hạn đã biết của đồ án (dữ liệu MovieLens không được
> phân phối lại theo giấy phép GroupLens).

## Vì sao có hai bộ artifact

| Bộ | Fit trên split | Dùng cho |
|---|---|---|
| `demo/artifacts/` | `full` — 158.774 user | Tab 1, Tab 2 |
| `demo/artifacts/coldstart/` | `coldstart` — tương tác của 1.381 phim held-out **đã bị xoá** | Tab 3 |

Tab 3 **bắt buộc** dùng bộ cold-start: nếu dùng mô hình fit trên split `full`, nó đã
học tương tác của phim held-out và sẽ xếp hạng phim đó bình thường — sai lệch hoàn
toàn so với giao thức mà báo cáo đo.

## Cấu trúc

```
demo/
├── app.py                     # entry point, 3 tab
├── loader.py                  # @st.cache_resource: nạp artifact một lần (~1,6s)
├── tabs/
│   ├── user_tab.py            # tab 1
│   ├── similar_tab.py         # tab 2
│   └── coldstart_tab.py       # tab 3
└── artifacts/                 # sinh ra, gitignore
```

Tầng nghiệp vụ nằm ở `src/api/` (không phụ thuộc Streamlit):

| Module | Vai trò |
|---|---|
| `src/api/model_artifacts.py` | Lưu / nạp model đã fit (`factors.npz` cho CF, `state.pkl` cho CB) |
| `src/api/recommender_service.py` | `recommend_by_user`, `similar_movies`, `coldstart_movies_for` |
| `src/api/item_similarity.py` | Cosine trên Tag Genome — cơ sở của gợi ý "phim tương tự" |

`src/api/` **không** import `src.eval.*` — evaluator chấm điểm trên nhãn test, còn
tầng serving chỉ biết `TrainData` và hợp đồng `score_users`.

## Giới hạn cần nói khi trình bày

- **Metadata hậu kỳ**: `genome-scores.csv` của GroupLens không có timestamp — tính
  ngoài động trên toàn corpus. Báo cáo (Chương 6) đo định lượng rủi ro này.
- **Cold-start là mô phỏng**: 10% phim bị giữ lại theo 5 nhóm năm phát hành; mọi user
  vẫn có lịch sử train. Không phải phim mới thật.
- **Chỉ offline**: không A/B test, không đo tác động thực tế.
- Demo dùng 2 trong 9 mô hình của đồ án (ALS — mạnh nhất CF; BM25 — mạnh nhất CB).
