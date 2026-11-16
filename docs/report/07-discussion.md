# Chương 7 — Thảo luận

> **Tổng hợp RQ1–RQ3** từ số liệu ở Chương 5–6, rồi rút ra bài học thiết kế
> protocol, giới hạn còn lại, và hàm ý thực tiễn.

---

## 7.1 Tổng hợp phát hiện

Bốn phát hiện chính của đồ án, theo thứ tự mức độ chắc chắn:

**1. CF thắng ở giao thức thường — và kết luận này đã được cân bằng.**
Ở giao thức `full` (158.774 user), ALS đạt NDCG@10 = 0,1375 so với CB-4 Hybrid
0,1054, chênh 23,3% theo mẫu số ALS. Khoảng cách này lớn hơn nhiễu thống kê (CI
bootstrap 95% không chạm nhau). Quan trọng hơn: sweep đã cân bằng (51 config CF vs
93 config CB, 144 tổng, bằng chứng commit trong repo), nên kết luận **không** còn
mang nghi ngờ thiên lệch về độ bao phủ tinh chỉnh — vốn là điểm yếu của các so
sánh CF-vs-CB trong tài liệu.

**2. CB thắng áp đảo ở cold-start item — đây là điều kiện cụ thể, không phải "CF
kém."**
Ở giao thức `coldstart`, mọi mô hình CF đạt NDCG@10 = **0,0000** — và con số này là
*đúng bằng không*, không phải "rất thấp": tương tác train của 1.381 phim held-out bị
xoá hoàn toàn, nên mô hình lọc cộng tác không có gì để học. Trong khi đó CB-4 đạt
**0,0347**, CB-1 0,0328, CB-2 0,0306, CB-3 0,0305; Wilcoxon cho `p < 1e-190` so với
mọi mô hình CF. Điểm mấu chốt nằm ở chỗ khác: đây **không** phải bằng chứng CF
kém, mà là bằng chứng **CF không thể áp dụng** khi phim mới chưa có ai xem.

**3. Coverage đảo ngược với độ chính xác.**
ALS chính xác nhất (0,1375) nhưng chỉ chạm **14,94%** catalog; CB-3 Tag Genome kém
nhất trong nhóm CB (0,0713) nhưng chạm **67,20%**. Gini đi cùng hướng: MostPop đạt
0,9985 (dồn suất cực đại), CB-3 đạt 0,9407. Đây là hệ quả trực tiếp của cấu trúc
long-tail ở §3.2.2 và nó là câu trả lời cho câu hỏi "mô hình nào tốt nhất" — câu
hỏi không có câu trả lời nếu không nói mục tiêu.

**4. Tín hiệu content không bị rò rỉ thời gian theo cách đo được.**
Phân tầng theo ba kỷ nguyên người dùng, thứ hạng **không đảo chiều ở bất kỳ tầng
nào**: CB-2 BM25 dẫn CB-1 TF-IDF dẫn CB-3 Genome ở cả ba tầng, và CF không bị CB
vượt ở tầng nào. Đối chứng "chỉ dùng genres" (không rò rỉ) chỉ đạt 0,0026–0,0060 ở
mọi tầng — tức phần lớn lợi thế của tag đến từ **tính chất nội dung**, không phải
từ việc "biết trước" phim tương lai.

## 7.2 Đánh đổi CF ↔ Content-based

| | **Collaborative Filtering** | **Content-based** |
|---|---|---|
| **Thế mạnh** | Khám phá mẫu hợp tác ẩn; không cần đặc trưng phim; độ chính xác cao khi lịch sử dày | Không cần tương tác; **giải thích được**; phủ rộng; sống được với phim mới |
| **Điểm yếu** | Cold-start item = 0,0000; dồn suất (coverage 14,94%); không giải thích được | Giới hạn bởi chất lượng đặc trưng; dễ "tự xem lại"; thiếu serendipity; chậm hơn ~4,6× khi chấm điểm |
| **Vị trí trong đồ án** | NDCG@10 = 0,1375 | 0,1054 → **0,0347** ở cold-start |
| **Khi nào thắng** | Catalog ổn định, lịch sử user dày | Luồng phim mới đến liên tục; cần giải thích; cần phủ rộng |

### 7.2.1 Một phát hiện tinh tế: CF–CB **thu hẹp** theo độ dài lịch sử

Kết quả phân tầng theo nhóm user (§5.5) cho thấy điều ít ngời hơn trực giác
"content-based chỉ tốt cho user mới":

| Nhóm user | ALS | CB-4 Hybrid | Chênh lệch |
|---|---:|---:|---:|
| 5–30 tương tác | 0,1241 | 0,0777 | **0,0465** |
| 31–100 tương tác | 0,1139 | 0,0830 | 0,0309 |
| > 100 tương tác | 0,1786 | 0,1567 | **0,0219** |

Khoảng cách CF–CB **thu hẹp hơn một nửa** khi đi từ nhóm lịch sử ngắn sang nhóm
lịch sử dài. Nghĩa là content-based **cạnh tranh được** với CF ngay ở nhóm user
lịch sử ngắn — nhóm mà trực giác cho rằng CF sẽ yếu vì thiếu dữ liệu. Trái lại,
content-based cần **tín hiệu** để dựng hồ sơ; với user chỉ xem 5 phim, hồ sơ đó
gần như rỗng và điểm CB chỉ đạt 0,0777. CF thắng vì lịch sử ngắn vẫn **có** tín
hiệu, còn CB thì không.

### 7.2.2 Một phát hiện nữa: tín hiệu quan trọng là *có tương tác*, không phải *mức hài lòng*

Ở §6.3, chế độ trọng số `centered` tệ hơn `binary_pos` ở cả ba nhóm user (chênh
−0,023 / −0,020 / −0,031, `p < 1e-44`), trong khi `rating` gần như tương đương
`binary_pos`. Nói cách khác: biết user chấm 4 sao hay 5 sao gần như không thêm
thông tin so với biết user đã xem phim đó hay không. Hệ quả thực tiễn: hệ thống
không cần lưu mức độ hài lòng nếu chỉ dùng cho mô hình nội dung — một lợi ích
đơn giản về không gian lưu trữ và độ phức tạp.

## 7.3 Bài học về thiết kế protocol

Bốn bài học này có giá trị vượt ra ngoài phạm vi đồ án.

**7.3.1 Split per-user quan trọng hơn tưởng — và con số phải tái lập được**

| Cách chia | User hợp lệ |
|---|---:|
| global 70/30 theo quantile timestamp | 4.399 |
| global 75/25 | 5.622 |
| **per-user 80/20 (dùng ở đây)** | **158.774** |

Hệ số tăng **×36,1**. Điều này không chỉ là "nhiều user hơn": split toàn cục loại
bỏ toàn bộ user đã ngừng xem — tức loại bỏ chính nhóm user lịch sử ngắn mà phát
hiện ở §7.2.1 cần để kiểm chứng. Một protocol có thể **tự tạo ra** kết luận sai
lệch chỉ bằng cách chọn cách chia.

> **Con số cũ "4.587 user / ×34,6" đã bị loại** — không tái lập được từ dữ liệu
> hiện tại. Bảng ở §3.3.3 liệt kê mọi biến thể đã thử, và con số **4.399 / ×36,1**
> là con số có bằng chứng.

**7.3.2 "Cùng giao thức" phải là bằng chứng, không phải lời tuyên bố**

Việc mỗi dòng kết quả mang `split_id` + `catalog_id` + `seed` biến một tuyên bố
chung chung thành phép so sánh ba trường ký tự. Cùng cơ chế đó cho phép
`scripts/check_report_numbers.py` đối chiếu mọi con số trong báo cáo với 27 bảng
CSV và báo **0 lệch** — tức người đọc có thể tái xác minh mà không cần tin tác giả.

**7.3.3 Tái lập được > nhanh**

BPR với `num_threads=1` tốn 39,84s thay vì ~7s. Đánh đổi này là đúng cho một đồ án
đánh giá, nhưng **không** phải lựa chọn mặc định cho hệ thống production. Nguyên tắc
rút ra: *khi so sánh thuật toán, thứ tự ưu tiên là tính tái lập; khi triển khai,
thứ tự ưu tiên là tốc độ* — và phải nói rõ mình đang ở trường hợp nào.

**7.3.4 Siêu tham số có thể chết ở giao thức khác**

`pop_prior` với λ = 0,4 cho NDCG@10 = **0,1081** ở giao thức thường — **cao hơn**
bản content thuần (0,1054). Nhưng ở giao thức cold-start nó rơi về **0,0010**, và
`frac_unseen@10` rơi từ 0,2223 xuống 0,0036. Nguyên nhân: mọi phim held-out có
popularity bằng 0 trong train, nên prior không phân biệt được chúng.

Bài học: **một siêu tham số phải được đánh giá trên mọi giao thức mà hệ thống sẽ
hoạt động**, không chỉ trên giao thức dùng để tinh chỉnh. Nếu không, ta chọn được
một cấu hình **tối ưu cho bài toán dễ** và không biết mình đã hy sinh bài toán khác.

## 7.4 Giới hạn

Bốn giới hạn cần nói thẳng khi trình bày:

**7.4.1 Metadata hậu kỳ — giới hạn lớn nhất**

Đã đo ở Chương 6, nhưng cần phân biệt hai mức khẳng định:

- **Đã chứng minh được:** thứ hạng giữa các mô hình **bất biến** qua ba kỷ nguyên;
  đối chứng "chỉ genres" cho thấy phần lớn lợi thế của tag đến từ tính chất nội dung.
- **Chưa chứng minh được:** *không rò rỉ*. Đây là thiếu bằng chứng, không phải bằng
  chứng không rò rỉ. Thêm nữa, confound tag ↔ popularity (Pearson 0,768) **vẫn
  còn** — chưa có biến thể nào cắt được nó. Số tuyệt đối của CB vẫn lạc quan hơn
  môi trường production.

**7.4.2 Cold-start là mô phỏng, không phải phim mới thật**

1.381 phim bị giữ lại theo 5 nhóm năm phát hành, nhưng mọi user vẫn có lịch sử
train. Chưa có ai **thật sự là người dùng mới** trong giao thức này. Nói cách khác,
đồ án đo được "hệ thống xử lý thế nào với **item** mới", chứ không phải "hệ thống xử
lý thế nào với **user** mới" — và user mới thật sự còn khó hơn, vì không chỉ thiếu
tín hiệu item mà còn thiếu toàn bộ lịch sử.

**7.4.3 Chỉ đánh giá offline**

Không có A/B test, không đo thời gian xem, không đo tỉ lệ giữ chân, không đo hài
lòng. Offline metric tốt **không** bảo đảm hành vi thực tế tốt — đây là khoảng
cách thường gặp giữa nghiên cứu và triển khai.

**7.4.4 Giao thức LOO chưa chạy**

Lựa chọn full ranking dựa trên trích dẫn Krichene & Rendle (2020), **chưa** có
bằng chứng Kendall τ đo trong repo. Script `scripts/run_sampled_loo_comparison.py`
đã sẵn sàng nhưng chưa chạy. Đây là hướng phát triển ưu tiên cao nhất (§8.2) vì
nó biến một lựa chọn *dựa trên lý thuyết* thành lựa chọn *có bằng chứng trong
repo*.

Ngoài ra: chỉ 9 mô hình, không deep learning; catalog bị lọc còn 13.816 phim (§3.4);
và chi phí chấm điểm của mô hình nội dung cao hơn CF ~4,6× (115,14s so với 24,91s),
điều đáng cân nhắc khi triển khai.

## 7.5 Hàm ý thực tiễn

**7.5.1 Hệ thống thật nên kết hợp, không chọn một**

Kết luận của đồ án không phải "dùng CF" hay "dùng CB" mà là **mỗi hướng có điều
kiện thắng riêng**:

| Tình huống | Nên dùng | Vì sao |
|---|---|---|
| User có lịch sử dày, catalog ổn định | **CF** (ALS) | NDCG@10 = 0,1375; tín hiệu cộng tác dày |
| Phim mới vừa đến | **CB** (BM25/Genome) | CF không có gì để học (0,0000) |
| Cần giải thích gợi ý cho người dùng | **CB** | Tag chung là lý do đọc được |
| Cần tránh dồn suất, phủ rộng catalog | **CB** (coverage 49–67%) | ALS chỉ chạm 14,94% |
| Catalog thay đổi mạnh, phim mới đến liên tục | **CB** | CF cần lịch sử để học |

Trong hệ thống thật, lựa chọn phụ thuộc **vòng đời của item**: một phim mới đến sẽ
được gợi ý bằng content-based cho tới khi tích luỹ đủ tương tác, rồi chuyển sang
được xếp hạng bằng CF. Đó là lý do chiến lược late fusion (CB-4) quan trọng, và
cũng là lý do `pop_prior` phải được kiểm tra trên cả hai giao thức.

**7.5.2 Ba quy tắc rút ra cho mọi đồ án đánh giá hệ thống gợi ý**

1. **Luôn đánh giá trên ≥ 2 giao thức** (thường + cold-start) trước khi chọn mô hình.
   Một cấu hình tối ưu cho giao thức dễ có thể vô dụng ở giao thức khác (§7.3.4).
2. **Cân bằng độ bao phủ sweep** giữa các nhóm phương pháp trước khi kết luận
   "nhóm này thắng" (§7.3.1).
3. **Đo rủi ro rò rỉ dữ liệu một cách trực tiếp**, đừng chỉ suy luận từ việc
   "không thấy vấn đề gì" (Chương 6).

## 7.6 Đối chiếu với kỳ vọng ban đầu

| Kỳ vọng ban đầu | Kết quả đo được | Đã xử lý ở đâu |
|---|---|---|
| CF luôn thắng content-based | Sai — đảo chiều hoàn toàn ở cold-start item | §5.3, §7.1.2 |
| BM25 tốt hơn TF-IDF | Đúng, ở cả giao thức `full` lẫn cả ba tầng kỷ nguyên | §2.3.2, §6.2 |
| Tag leak làm cho content thắng giả | Chưa chứng minh được — thứ hạng bất biến qua 3 tầng | §6.2, §6.5.2 |
| Mô hình nội dung chỉ tốt cho user mới | Sai — CB **thu hẹp** khoảng cách với CF khi lịch sử dài | §5.5, §7.2.1 |
| Thêm popularity prior sẽ tốt hơn | Đúng ở giao thức thường, **chết** ở cold-start | §4.3.3, §7.3.4 |
| Mô hình tốt nhất là mô hình NDCG cao nhất | Không — coverage đảo ngược với độ chính xác | §6.4, §7.1.3 |

Bảng này là phần trung thực nhất của chương: nó cho thấy **năm trong sáu kỳ vọng
ban đầu đã bị dữ liệu bác bỏ**, và mỗi lần bác bỏ đều dẫn tới một kết luận có
giá trị hơn.

---

**Tài liệu tham khảo chương này:** Krichene & Rendle (2020); Shani & Gunawardana
(2011); Ricci et al. (2015); McFee et al. (2020).
