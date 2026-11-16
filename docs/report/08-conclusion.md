# Chương 8 — Kết luận và hướng phát triển

---

## 8.1 Đóng góp

Đồ án này đánh giá 9 mô hình gợi ý — 3 collaborative filtering, 4 content-based, 2
baseline — trên **một** giao thức đánh giá thống nhất, với ba giao thức khác nhau và
bằng chứng protocol trên từng dòng kết quả. Năm đóng góp chính:

**1. Một giao thức, một đường chạy.** Trước đó có hai evaluator song song với protocol
khác nhau (global 70/30 và per-user 80/20), khiến kết quả không thể so sánh. Đồ án hợp
nhất về một evaluator duy nhất, tách `TrainData` khỏi `Dataset` để mô hình **về mặt
cấu trúc** không thể chạm nhãn test, và chạy 9 mô hình × 3 giao thức.

**2. Bằng chứng protocol từng dòng.** Mỗi dòng trong 27 bảng kết quả mang `split_id`
+ `catalog_id` + `seed`. Câu "các mô hình được đánh giá trên cùng một giao thức" trở
thành thứ **kiểm chứng được** bằng cách so ba trường ký tự, không phải lời tuyên bố.
Cùng cơ chế cho phép `scripts/check_report_numbers.py` đối chiếu mọi con số trong
báo cáo với CSV và báo **0 lệch**.

**3. Cân bằng sweep — biến kết luận "CF thắng" thành kết luận có cơ sở.** 51 config CF
vs 93 config CB (144 tổng), bằng chứng sweep commit trong repo
(`results/final/ablation-*.csv`). Ở giao thức `full`, CF thắng: ALS NDCG@10 =
**0,1375** so với CB-4 **0,1054** (chênh 23,3% theo mẫu số ALS), với CI bootstrap
95% không chạm nhau.

**4. Kết luận đảo chiều ở cold-start, có kiểm định thống kê.** Ở giao thức
`coldstart`, mọi mô hình CF đạt NDCG@10 = **0,0000** — *đúng bằng không*, vì tương
tác train của 1.381 phim held-out bị xoá hoàn toàn — trong khi CB-4 đạt **0,0347**
(Wilcoxon `p < 1e-190`). Kết luận được nêu kèm **điều kiện**: "CF thắng ở giao thức
thường", không phải khẩu hiệu chung.

**5. Đo trực tiếp rủi ro rò rỉ metadata hậu kỳ.** Vì `genome-scores.csv` không có
timestamp, đồ án phân tầng kết quả theo ba kỷ nguyên người dùng và thêm đối chứng
"chỉ dùng genres". Kết quả: thứ hạng **không đảo chiều** ở bất kỳ tầng nào
(CB-2 ≥ CB-1 > CB-3 ở cả ba), và đối chứng chỉ-genres chỉ đạt 0,0026–0,0060 — tức
phần lớn lợi thế của tag đến từ **tính chất nội dung**, không phải từ rò rỉ thời gian.
Chương 6 cũng nêu rõ **giới hạn của chính kết luận này**: chứng minh được *thứ hạng
bất biến*, không chứng minh được *không rò rỉ*.

Ngoài năm đóng góp lý thuyết, đồ án còn xây dựng **tầng serving** (`src/api/`) và
**demo Streamlit 3 tab** — nơi tab cold-start cho thấy ALS trả về điểm **0,000000** với
phim chưa có ai xem trong khi content-based vẫn xếp hạng được. Điều này biến kết
luận từ bảng số thành hiện tượng **quan sát được**.

## 8.2 Hướng phát triển

| Hướng | Chi phí | Ghi chú |
|---|---|---|
| **Chạy sampled LOO → Kendall τ** | ~20 phút | **Ưu tiên cao nhất.** Biến lựa chọn full ranking từ trích dẫn thành kết quả thực nghiệm trong repo. Script đã sẵn sàng, chưa chạy |
| **Hybrid CF+CB ở tầng fusion** | Trung bình | Nối hai kết quả trái chiều của đồ án; khác với CB-4 (chỉ fusion trong nhóm content) |
| **Biến thể giới hạn số tag/phim của CB-1** | 1 run | Cắt confound tag ↔ popularity (Pearson 0,768) — mối lớn nhất còn lại của kết luận chống rò rỉ |
| **Bảng Wilcoxon so chéo CF↔CB** | Thấp | Cần bảng riêng — câu hỏi khác với kiểm định trong từng bảng |
| **Đo năng lực xếp hạng ngoài top-10** | Trung bình | NDCG@50 đã có, nhưng chưa phân tích sâu |
| **Giao thức cold-start cho user mới** | Cao | Bổ sung cho cold-start item hiện tại — vẫn giữ lịch sử train của mọi user |
| **Deep learning (NCF, BERT4Rec)** | Cao | Ngoài phạm vi đồ án; cần cân nhắc khi catalog và lịch sử lớn hơn nhiều |
| **A/B test hoặc đánh giá online** | Cao | Bổ sung cho đánh giá offline; cần hạ tầng sản phẩm |

Hai hướng đầu là ưu tiên cao vì chúng **đóng lỗ hổng của kết luận hiện tại** thay vì
chỉ mở rộng phạm vi: chạy LOO biến một giả định lý thuyết thành bằng chứng đo được,
và cắt confound popularity biến một kết luận "chưa chứng minh được không rò rỉ"
thành kết luận có điều kiện kiểm soát.

## 8.3 Lời kết

Bài toán gợi ý không có một mô hình tốt nhất — nó có một mô hình **đúng với điều
kiện**. Collaborative filtering thắng khi lịch sử tương tác dày và catalog đã ổn
định; content-based thắng khi phim mới chưa có ai xem, khi cần giải thích, và khi
cần phủ rộng catalog. Đồ án này đo được cả hai chiều đó **trên cùng một giao thức**,
với bằng chứng protocol đủ để người đọc tự kiểm chứng, và với phân tích rò rỉ
metadata đủ để biết phần nào của kết luận là chắc chắn và phần nào là có điều kiện.

Giá trị lớn nhất của đồ án không nằm ở con số NDCG nào, mà ở chỗ nó biến một câu
hỏi quen thuộc — "CF hay content-based tốt hơn?" — thành một câu hỏi có thể đo
được: **"ở điều kiện nào, với bằng chứng nào, và với điều kiện giới hạn nào?"**
