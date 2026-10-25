# Roadmap Nghiên Cứu & Triển Khai: Recommendation System cho Ecommerce

**Phạm vi:** Collaborative Filtering (implicit feedback) + Content-based Filtering
**Thời gian:** 8 tuần
**Mục tiêu cuối:** Báo cáo nghiên cứu + mini project có dataset, model, đánh giá bằng thước đo chuẩn

---

## Giai đoạn 0: Chuẩn bị (trước Tuần 1)

- [ ] Cài đặt môi trường: Python 3.10+, Jupyter/Colab
- [ ] Cài thư viện chính: `pandas`, `numpy`, `scikit-learn`, `implicit`, `lightfm`, `scipy`
- [ ] Tạo repo/folder project với cấu trúc:
  ```
  /data          -> dataset thô và đã xử lý
  /notebooks     -> notebook thử nghiệm
  /src           -> code module (data processing, model, eval)
  /reports       -> báo cáo, biểu đồ kết quả
  ```
- [ ] Đọc nhanh 2-3 bài overview về Recommendation System (Google's ML Crash Course - Recommendation Systems là điểm khởi đầu tốt, ngắn gọn)

---

## Tuần 1: Nền tảng lý thuyết + Chọn dataset

**Mục tiêu:** Hiểu đủ lý thuyết để viết được phần "Giới thiệu" và "Tổng quan phương pháp" của báo cáo.

### Nội dung học
- [ ] Phân loại các hướng tiếp cận: Content-based, Collaborative Filtering (User-based, Item-based), Matrix Factorization, Hybrid, Deep learning-based (chỉ cần hiểu ý tưởng, không cần implement)
- [ ] Explicit feedback vs Implicit feedback — khác biệt và hệ quả (tại sao không dùng RMSE cho implicit)
- [ ] Vấn đề Cold-start: user mới, item mới
- [ ] Đọc sơ lược về ALS for implicit feedback (paper gốc: "Collaborative Filtering for Implicit Feedback Datasets" - Hu, Koren, Volinsky 2008) — không cần đọc sâu toán, chỉ cần hiểu ý tưởng confidence-based weighting

### Công việc thực hành
- [ ] Chọn dataset: **Retailrocket** (khuyến nghị vì có event-based structure gần ecommerce thật) hoặc **Amazon Reviews** (nếu ưu tiên content-based với text mô tả phong phú)
- [ ] Tải dataset, đọc qua cấu trúc file, viết mô tả dataset vào báo cáo (số lượng user, item, interaction, khoảng thời gian)

### Deliverable tuần 1
- Đề cương báo cáo (outline) hoàn chỉnh
- Dataset đã tải về, đã đọc hiểu cấu trúc

---

## Tuần 2: Khám phá & Tiền xử lý dữ liệu (EDA)

**Mục tiêu:** Hiểu rõ đặc điểm dữ liệu trước khi đưa vào model — bước này quyết định phần lớn chất lượng kết quả sau này.

### Công việc
- [ ] EDA: phân phối số lượng interaction/user, số lượng interaction/item (thường sẽ thấy long-tail distribution — vài item rất phổ biến, đa số item ít tương tác)
- [ ] Xác định loại implicit signal sẽ dùng (nếu Retailrocket: view/addtocart/transaction — có thể gán trọng số khác nhau, ví dụ transaction=3, addtocart=2, view=1)
- [ ] Lọc dữ liệu: loại bỏ user có quá ít interaction (ví dụ <5), loại item quá hiếm (ví dụ <5 lượt) để tránh nhiễu
- [ ] Xây dựng ma trận user-item (sparse matrix dùng `scipy.sparse`)
- [ ] **Chia train/test theo thời gian** (quan trọng — tránh data leakage): lấy interaction gần nhất của mỗi user làm test set, phần còn lại làm train (leave-one-out hoặc time-based split), KHÔNG chia random

### Kiến thức cần nắm chắc
- [ ] Tại sao chia random theo dòng sẽ gây leak thông tin tương lai trong bài toán recommendation
- [ ] Sparsity của ma trận user-item và ảnh hưởng đến model

### Deliverable tuần 2
- Notebook EDA hoàn chỉnh với biểu đồ (phân phối interaction, sparsity)
- Ma trận user-item đã xử lý, train/test set đã tách theo thời gian
- Phần "Dữ liệu và tiền xử lý" trong báo cáo

---

## Tuần 3: Xây dựng Baseline + Collaborative Filtering (ALS)

**Mục tiêu:** Có baseline để so sánh, sau đó xây model chính đầu tiên.

### Công việc
- [ ] **Baseline 1 - Popularity-based**: gợi ý top item được tương tác nhiều nhất (không cá nhân hóa) — dùng làm mốc so sánh tối thiểu
- [ ] **Baseline 2 - Random**: gợi ý ngẫu nhiên — mốc sàn thấp nhất
- [ ] **Model chính - ALS (Alternating Least Squares) cho implicit feedback**, dùng thư viện `implicit`
  ```python
  from implicit.als import AlternatingLeastSquares
  model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=15)
  model.fit(user_item_matrix)
  ```
- [ ] Thử nghiệm hyperparameter: số latent factors, regularization, iterations — ghi lại ảnh hưởng đến kết quả

### Kiến thức cần nắm
- [ ] Ý tưởng Matrix Factorization: phân rã ma trận user-item thành 2 ma trận latent factor (user embedding × item embedding)
- [ ] Vai trò của confidence weighting trong ALS implicit (khác gì so với ALS cho explicit rating)

### Deliverable tuần 3
- Model ALS đã train, có thể sinh ra top-N gợi ý cho từng user
- So sánh sơ bộ (định tính) giữa baseline và ALS

---

## Tuần 4: Đánh giá định lượng (Evaluation Metrics)

**Mục tiêu:** Đây là phần cốt lõi để báo cáo có tính thuyết phục — đánh giá đúng cách, đúng thước đo.

### Nội dung
- [ ] Hiểu và implement các ranking metrics:
  - **Precision@K**: tỷ lệ item đúng (có trong test set) trong top-K gợi ý
  - **Recall@K**: tỷ lệ item đúng được tìm thấy trong top-K trên tổng số item đúng
  - **NDCG@K** (Normalized Discounted Cumulative Gain): có tính đến thứ hạng, item đúng ở vị trí cao được điểm cao hơn
  - **MAP@K** (Mean Average Precision): trung bình precision tại các vị trí có item đúng
- [ ] Có thể dùng thư viện hỗ trợ sẵn (`implicit` có hàm đánh giá built-in) hoặc tự viết hàm eval để hiểu rõ cơ chế
- [ ] Chạy đánh giá cho: Popularity baseline, Random baseline, ALS — lập bảng so sánh

### Lưu ý quan trọng
- [ ] Đảm bảo khi tính Recall/Precision, loại trừ các item mà user đã tương tác trong tập train (không gợi ý lại cái đã biết)
- [ ] Chạy với nhiều giá trị K (ví dụ K=5, 10, 20) để thấy xu hướng

### Deliverable tuần 4
- Bảng so sánh Precision@K, Recall@K, NDCG@K giữa các phương pháp
- Biểu đồ (bar chart hoặc line chart theo K) minh họa
- Phần "Phương pháp đánh giá" trong báo cáo

---

## Tuần 5: Content-based Filtering

**Mục tiêu:** Xây dựng hướng tiếp cận thứ 2 để so sánh và giải quyết cold-start.

### Công việc
- [ ] Trích xuất đặc trưng sản phẩm: category, tên, mô tả (nếu có text) → dùng TF-IDF hoặc CountVectorizer
- [ ] Tính similarity giữa các item (cosine similarity trên vector đặc trưng)
- [ ] Xây dựng logic gợi ý: dựa trên lịch sử item user đã tương tác → tìm item tương tự (similarity cao) chưa tương tác
- [ ] Xử lý trường hợp cold-start item mới: vì content-based không cần dữ liệu tương tác, có thể gợi ý ngay dựa trên đặc trưng

### Deliverable tuần 5
- Model content-based hoàn chỉnh, sinh được top-N gợi ý
- Đánh giá bằng cùng bộ thước đo (Precision@K, Recall@K, NDCG@K) để so sánh công bằng với ALS

---

## Tuần 6: So sánh, Phân tích & Hybrid đơn giản (tùy chọn)

**Mục tiêu:** Tổng hợp kết quả, rút ra insight, và nếu còn thời gian, thử kết hợp 2 phương pháp.

### Công việc
- [ ] Lập bảng so sánh tổng thể: Popularity / Random / ALS (Collaborative) / Content-based — theo tất cả metrics
- [ ] Phân tích: phương pháp nào tốt hơn ở nhóm user có nhiều interaction? Phương pháp nào tốt hơn ở user/item mới (cold-start)?
- [ ] (Tùy chọn, nếu còn thời gian) Hybrid đơn giản: kết hợp điểm số của ALS và Content-based theo trọng số (weighted hybrid), hoặc switching (dùng content-based khi user/item mới, dùng ALS khi đủ dữ liệu)
- [ ] Case study định tính: chọn 2-3 user cụ thể, xem gợi ý thực tế của từng phương pháp trông như thế nào — phần này giúp báo cáo sinh động, dễ trình bày khi bảo vệ

### Deliverable tuần 6
- Bảng so sánh đầy đủ, biểu đồ tổng hợp
- Phần "Kết quả và Thảo luận" trong báo cáo gần như hoàn chỉnh

---

## Tuần 7: Viết báo cáo

**Mục tiêu:** Hoàn thiện văn bản báo cáo.

### Cấu trúc báo cáo đề xuất
1. **Giới thiệu**: bài toán recommendation trong ecommerce, động lực (liên hệ project ecommerce thực tế của bạn)
2. **Tổng quan lý thuyết**: các hướng tiếp cận (trình bày đầy đủ cả những gì không implement: Deep learning-based, clickstream-based, ANN/FAISS cho scale — như phần mở rộng lý thuyết)
3. **Phạm vi và phương pháp đề xuất**: giải thích lý do chọn Collaborative Filtering (ALS implicit) + Content-based, tại sao phù hợp với dữ liệu ecommerce
4. **Dữ liệu**: mô tả dataset, EDA, cách xử lý implicit feedback, cách chia train/test
5. **Thực nghiệm**: chi tiết model, hyperparameter
6. **Kết quả và đánh giá**: bảng số liệu, biểu đồ, phân tích
7. **Thảo luận**: hạn chế, so sánh với hướng nâng cao chưa implement (NCF, LayoutLM-style multimodal nếu liên quan, hybrid phức tạp hơn)
8. **Kết luận và hướng phát triển**: đặc biệt nhấn vào khả năng áp dụng sang project ecommerce thật của bạn

### Deliverable tuần 7
- Bản thảo báo cáo đầy đủ

---

## Tuần 8: Review, hoàn thiện, chuẩn bị bảo vệ

- [ ] Đọc lại toàn bộ báo cáo, kiểm tra tính nhất quán số liệu
- [ ] Chuẩn bị slide trình bày (nếu cần)
- [ ] Liệt kê trước các câu hỏi phản biện có thể gặp và chuẩn bị câu trả lời, ví dụ:
  - "Tại sao không dùng RMSE?" → vì dữ liệu implicit không có rating rõ ràng
  - "Tại sao chia train/test theo thời gian mà không random?" → tránh data leakage
  - "Nếu scale lên hàng triệu item thì sao?" → đề cập ANN/FAISS trong phần thảo luận
  - "So với Neural CF/Deep learning thì sao?" → trình bày trade-off (độ phức tạp vs độ chính xác, dữ liệu cần nhiều hơn)
- [ ] Demo thử nghiệm nhanh (nếu có thể) áp dụng lên tập nhỏ dữ liệu ecommerce thật của bạn để minh họa khả năng ứng dụng

### Deliverable tuần 8
- Báo cáo hoàn chỉnh, sẵn sàng nộp/bảo vệ

---

## Thư viện chính sẽ dùng xuyên suốt

| Mục đích | Thư viện |
|---|---|
| Xử lý dữ liệu | `pandas`, `numpy`, `scipy.sparse` |
| Collaborative Filtering (ALS implicit) | `implicit` |
| Content-based (TF-IDF, similarity) | `scikit-learn` |
| Hybrid (tùy chọn, thay thế cho tự viết) | `lightfm` |
| Trực quan hóa | `matplotlib`, `seaborn` |

## Rủi ro cần lưu ý trước

- Dataset quá lớn → nên lọc subset (1 category, hoặc 1 khoảng thời gian) ngay từ Tuần 2 để tránh train quá lâu
- Đừng bỏ qua bước chia train/test theo thời gian — đây là lỗi phổ biến nhất khiến kết quả đánh giá sai lệch
- Dành đủ thời gian buffer ở Tuần 8 — viết báo cáo luôn tốn thời gian hơn dự kiến