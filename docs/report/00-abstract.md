# Tóm tắt (Abstract)

Hệ thống gợi ý là thành phần cốt lõi của mọi nền tảng nội dung số, và bài toán trung
tâm của nó — xếp hạng top-N — khác bài toán dự đoán rating ở chỗ **thứ tự quan trọng
hơn giá trị tuyệt đối**. Hai hướng tiếp cận chính, collaborative filtering (CF) và
content-based filtering (CB), thường được so sánh như hai lựa chọn thay thế, nhưng
thực ra mỗi hướng thắng ở một vùng bài toán khác nhau: CF khi lịch sử tương tác dày,
CB khi phim mới chưa có tương tác nào. Báo cáo này đặt câu hỏi đó làm trung tâm và
trả lời bằng **phép đo** chứ không bằng khẳng định.

**Phương pháp.** Đồ án hợp nhất **9 mô hình** (3 CF, 4 content-based, 2 baseline) trên
**một giao thức đánh giá duy nhất**: full-ranking trên toàn bộ 13.816 ứng viên, split
thời gian 80/20 theo từng người dùng, ngưỡng relevant ≥ 4,0, chấm theo lô 2.000 user.
Ba giao thức được dùng cho ba mục đích khác nhau: `dev` (20.000 user) để quét 144
config tham số — **51 CF / 93 CB**, cân bằng để kết luận không mang nghi ngờ thiên lệch;
`full` (158.774 user) cho bảng kết quả chính; `coldstart` (10.992 user) cho bài toán
phim mới. Mỗi dòng kết quả mang `split_id` + `catalog_id` + `seed`, biến câu "cùng
giao thức" thành thứ kiểm chứng được thay vì lời tuyên bố.

**Kết quả chính.** Ở giao thức thường, **CF thắng**: ALS đạt NDCG@10 = 0,1375 so với
CB-4 Hybrid 0,1054 (chênh 23,3%), với khoảng cách lớn hơn nhiễu thống kê (CI bootstrap
95% không chạm nhau). Nhưng ở giao thức **cold-start**, kết luận **đảo chiều áp đảo**:
mọi mô hình CF đạt NDCG@10 = **0,0000** — *đúng bằng không*, vì tương tác train của
1.381 phim held-out bị xoá hoàn toàn nên mô hình lọc cộng tác không có gì để học —
trong khi CB-4 đạt **0,0347** (Wilcoxon `p < 1e-190`). Báo cáo còn đo được hiệu tượng
**coverage đảo ngược với độ chính xác**: ALS chính xác nhất nhưng chỉ chạm 14,94%
catalog, còn CB-3 Tag Genome chạm 67,20%.

**Đóng góp.** (i) Loại bỏ hai evaluator song song với protocol khác nhau, thay bằng
một đường chạy duy nhất có bằng chứng protocol trên từng dòng. (ii) Cân bằng độ bao
phủ sweep (51 CF / 93 CB) để biến kết luận "CF thắng" từ nghi ngờ thiên lệch thành
kết luận có cơ sở — bằng chứng sweep được commit trong repo để người đọc tự đếm.
(iii) Nêu kết luận đảo chiều ở cold-start **kèm điều kiện**, kèm kiểm định thống kê.
(iv) Đo trực tiếp rủi ro rò rỉ metadata hậu kỳ — vì `genome-scores.csv` của GroupLens
không có timestamp — bằng cách phân tầng theo ba kỷ nguyên người dùng: thứ hạng
**không đảo chiều** ở bất kỳ tầng nào, và đối chứng "chỉ dùng genres" (không rò rỉ)
chỉ đạt 0,0026–0,0060, chứng tỏ phần lớn lợi thế của tag đến từ tính chất nội dung
chứ không phải từ việc "biết trước" phim tương lai. Chương tranh luận cũng nêu rõ
giới hạn của chính kết luận này: chứng minh được *thứ hạng bất biến*, chưa chứng minh
được *không rò rỉ*. (v) Xây dựng tầng serving và demo Streamlit 3 tab, nơi tab
cold-start cho thấy trực quan ALS trả điểm **0,000000** với phim chưa có ai xem
trong khi content-based vẫn xếp hạng được — biến kết luận từ bảng số thành hiện
tượng quan sát được.

---

**Từ khóa:** recommender systems, collaborative filtering, content-based filtering,
cold-start, MovieLens 25M, offline evaluation, NDCG, Tag Genome, metadata leakage
