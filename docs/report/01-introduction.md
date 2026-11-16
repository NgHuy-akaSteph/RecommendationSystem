# Chương 1 — Mở đầu

> **Trả lời RQ1–RQ3:** chương này đặt bối cảnh, nêu ba câu hỏi nghiên cứu mà toàn bộ
> báo cáo nhằm trả lời, và nói rõ phạm vi cũng như những gì đồ án **không** làm.
> Chương 5 trả lời RQ1 và RQ2; Chương 6 trả lời RQ3.

---

## 1.1 Bối cảnh và động lực

Hệ thống gợi ý là thành phần cốt lõi của mọi nền tảng nội dung số — streaming phim,
thương mại điện tử, nhạc, tin tức. Ở quy mô lớn, vấn đề không còn là *"nên giới thiệu
gì"* mà là *"trong hàng triệu phim, làm sao đưa đúng 10 phim đúng lên đầu màn hình
của đúng người này"*. Cách tiếp cận phổ biến nhất: **xếp hạng** (`top-N
recommendation`) — mô hình trả về danh sách N phim được xếp theo mức độ phù hợp
dự đoán.

Bài toán xếp hạng khác bài toán dự đoán rating ở một điểm cốt lõi: **thứ tự quan
trọng hơn giá trị tuyệt đối**. Người dùng không xem phim đứng thứ 500, dù mô hình
dự đoán sai 0,2 điểm; nhưng họ sẽ không cuộn qua 10 phim để tìm phim đứng thứ nhất.
Hệ quả trực tiếp: metric phải đo **thứ hạng** (NDCG@10, HR@10, Recall@10), không
phải **sai số tuyệt đối** (RMSE, MAE) — một lựa chọn chi tiết nhưng quyết định toàn
bộ thiết kế đánh giá ở §2.4.

Hai hướng tiếp cận chính, mỗi hướng có một điểm mạnh không thể thay thế:

- **Collaborative filtering (CF)** — học từ hành vi của cộng đồng: "người giống bạn
  đã xem phim này". Khám phá được mối liên quan mà không ai viết ra, kể cả mối
  liên quan giữa hai phim không hề giống nhau về nội dung. Nhưng mô hình này **không
  có gì để học** với một phim chưa ai xem.
- **Content-based filtering (CB)** — học từ đặc trưng nội dung của chính phim: thể
  loại, tag, mô tả. Không cần ai đã xem phim đó, và **giải thích được** vì sao phim
  được gợi ý. Nhưng nó chỉ tìm được phim *giống nội dung*, và phụ thuộc chất lượng
  đặc trưng.

Câu hỏi kinh điển — "hướng nào thắng?" — trên thực tế là câu hỏi sai, vì câu trả
lời phụ thuộc **điều kiện**. Đồ án này lấy câu hỏi đó làm trung tâm, và trả lời nó
bằng **phép đo** chứ không bằng **khẳng định**.

## 1.2 Câu hỏi nghiên cứu

> **RQ1 — Với cùng một giao thức đánh giá và độ bao phủ sweep tương đương, CF và
> Content-based thắng ở đâu?**

Điểm mấu chốt là *độ bao phủ sweep tương đương*. Trước đây, khi CF được quét
nhiều tham số hơn content-based, mọi tuyên bố "CF thắng" đều mang nghi ngờ thiên
lệch: có thể CF không mạnh hơn, chỉ được tinh chỉnh kỹ hơn. Đồ án này quét **51
config CF** và **93 config CB** (144 tổng) rồi mới so sánh, để biến kết luận từ
"có vẻ" thành "có cơ sở".

> **RQ2 — Điều gì xảy ra khi phim mới chưa có tương tác nào (cold-start item)?**

Đây không phải trường hợp hiếm. Trong mọi hệ thống thực, mỗi ngày có hàng trăm
phim mới đến, và đó là lúc content-based trở thành lựa chọn duy nhất. Đồ án dựng
một giao thức riêng: giữ lại 1.381 phim (10% catalog) theo 5 nhóm năm phát hành,
xoá toàn bộ tương tác train của chúng, rồi đo lại.

> **RQ3 — Liệu kết luận có bị đảo chiều nếu phân tầng theo thời kỳ người dùng —
> tức là liệu tín hiệu content có "biết trước" phim tương lai không?**

Đây là câu hỏi nguy hiểm nhất và cũng là câu hỏi mà đồ án này cho rằng các bài báo
thường bỏ qua. `genome-scores.csv` của GroupLens **không có timestamp** — Tag
Genome được tính ngoài động trên toàn corpus, tức có thể chứa thông tin về các phim
chưa phát hành tại thời điểm một user đang xem. Nếu vậy, kết quả content-based có
thể lạc quan do **rò rỉ metadata hậu kỳ** chứ không phải do nội dung thật sự hữu ích.

Ba câu hỏi này tạo thành một cấu trúc báo cáo: **Chương 5** trả lời RQ1–RQ2 bằng
số liệu, **Chương 6** trả lời RQ3 bằng phân tầng theo ba kỷ nguyên người dùng.

## 1.3 Đóng góp của đồ án

1. **Một giao thức, một đường chạy.** Trước đó có hai evaluator song song với
   protocol khác nhau (global 70/30 và per-user 80/20), dẫn tới kết quả không thể so
   sánh. Đồ án hợp nhất 9 mô hình trên **một** evaluator, chạy trên **ba** giao thức
   (`dev` / `full` / `coldstart`).

2. **Bằng chứng protocol trên từng dòng kết quả.** Mỗi dòng trong 27 bảng mang
   `split_id` + `catalog_id` + `seed`. Câu "các mô hình này được đánh giá trên cùng
   một giao thức" trở thành thứ **kiểm chứng được** bằng cách so ba trường ký tự,
   không phải lời tuyên bố.

3. **Sweep cân bằng (51 CF / 93 CB).** Biến kết luận "CF thắng" từ một tuyên bố
   có nghi ngờ thiên lệch thành kết luận có cơ sở. Bằng chứng sweep được commit
   trong repo (`results/final/ablation-*.csv`, 144 dòng) nên người đọc tự đếm được.

4. **Đo trực tiếp rủi ro rò rỉ metadata hậu kỳ.** Chương 6 phân tầng theo ba kỷ
   nguyên người dùng và thêm một đối chứng "chỉ dùng genres" (không rò rỉ) — cho
   thấy phần lớn lợi thế của tag đến từ **tính chất nội dung**, không phải từ rò rỉ
   thời gian.

5. **Kết luận đảo chiều ở cold-start, có kiểm định thống kê.** CB thắng áp đảo ở
   giao thức cold-start (NDCG@10 = 0,0347 so với 0,0000 của mọi mô hình CF), Wilcoxon
   cho `p < 1e-190`. Kết luận được nêu kèm **điều kiện** ("CF thắng ở giao thức
   thường"), không phải khẩu hiệu chung.

6. **Tầng serving và demo chạy được.** `src/api/` biến mô hình đánh giá offline
   thành API gợi ý (`recommend_by_user`, `similar_movies`, `coldstart_movies_for`),
   và demo Streamlit 3 tab cho thấy các kết luận trên **bằng mắt** — đặc biệt là
   tab cold-start, nơi ALS cho điểm **đúng bằng 0,000000** với phim chưa có ai xem
   trong khi content-based vẫn xếp hạng được.

## 1.4 Phạm vi và giới hạn

| **Trong phạm vi** | **Ngoài phạm vi** |
|---|---|
| MovieLens 25M, explicit rating (rating 0,5–5,0) | Deep learning (NCF, BERT4Rec) |
| 9 mô hình — 3 CF, 4 content-based, 2 baseline | Session-based, social, knowledge-based |
| Full ranking trên 13.816 ứng viên | A/B test, đo tác động thực tế |
| 3 giao thức: `dev` / `full` / `coldstart` | Bảng "no-leak" sạch (xem ghi chú bên dưới) |
| Metric độ chính xác + hành vi + thống kê | Scaled/real-time, hệ thống phân tán |
| Metadata hậu kỳ — **phải nêu rõ và đo** | Hybrid CF+CB ở tầng late fusion |
| Demo offline, chạy local | Deploy production, API công khai |

> **Ghi chú về hàng "bảng no-leak":** con số "mất 66% user" **không có bảng số
> trong repo**. Repo hiện chỉ có split `temporal80_20` (per-user) và biến thể
> `temporal80_20+coldstart_item`. Một nhánh "no-leak" sạch không dựng được trên split
> per-user vì đặc trưng item được dùng chung cho toàn corpus — đó chính là nội dung
> phân tích ở Chương 6, thay vì một bảng số. Vì vậy báo cáo **không** ghi con số
> 66% ở bất kỳ đâu.

Bốn giới hạn cần nhắc khi trình bày:

- **Metadata hậu kỳ.** Đã đo và phân tích ở Chương 6, nhưng số tuyệt đối của CB vẫn
  lạc quan hơn môi trường production. Đồ án chứng minh được *thứ hạng bất biến qua
  các kỷ nguyên*, không chứng minh được *không rò rỉ* — khác nhau về mức độ khẳng
  định.
- **Cold-start là mô phỏng.** 10% phim bị giữ lại theo 5 nhóm năm phát hành, nhưng mọi
  user vẫn có lịch sử train. Đây không phải tình huống user mới thật sự.
- **Chỉ đánh giá offline.** Không có kiểm định thực tế với người dùng.
- **Giao thức LOO chưa chạy.** Lựa chọn full ranking dựa trên trích dẫn học thuật
  (Krichene & Rendle, 2020), chưa có bằng chứng Kendall τ đo trong repo.

## 1.5 Cấu trúc báo cáo

Mỗi chương mở đầu bằng câu "Trả lời RQ nào":

| Chương | Trả lời | Nội dung cốt lõi |
|---|---|---|
| **2** Literature Review | — | Cơ sở lý thuyết: CF, content-based, metric xếp hạng |
| **3** Dataset & Preprocessing | — | Dữ liệu, EDA, bốn quyết định tiền xử lý và ảnh hưởng của chúng |
| **4** Methodology | — | 9 mô hình, 3 giao thức, bằng chứng protocol |
| **5** Experimental Results | **RQ1, RQ2** | Bảng chính, cân bằng sweep, đảo chiều cold-start, kiểm định |
| **6** Phân tích chống rò rỉ & tranh luận | **RQ3** | Phân tầng 3 kỷ nguyên, metric hành vi, khi nào phương pháp nào thắng |
| **7** Discussion | — | Tổng hợp, đánh đổi CF↔CB, bài học thiết kế protocol, giới hạn, hàm ý |
| **8** Conclusion & Future Work | — | Đóng góp, hướng phát triển |
| **Phụ lục** | — | Bảng sweep đầy đủ, CI và Wilcoxon, quy trình tái lập, cách kiểm chứng số |

**Chương 3 và Chương 4** được viết theo hướng "đọc được mà vẫn kiểm chứng được": mỗi
quyết định thiết kế đều kèm lý do và chỉ rõ nó ảnh hưởng tới kết luận nào ở đâu.
**Chương 6** là chương đặc biệt của đồ án — nơi biến "con số" thành "kết luận đáng
tin", và nơi tự nêu giới hạn của chính kết luận chống rò rỉ.

---

**Tài liệu tham khảo chương này:** Harper & Konstan (2015); Järvelin & Kekäläinen
(2002); Krichene & Rendle (2020); Ricci et al. (2015).
