# Development Roadmap

Trạng thái các phase của đồ án và việc còn lại.

Đọc kèm: [`codebase-summary.md`](codebase-summary.md), [`system-architecture.md`](system-architecture.md).

---

## 1. Mốc tổng thể

**Trạng thái hiện tại:** pipeline đánh giá hoàn chỉnh, kết quả đủ bằng chứng để viết
báo cáo. Đường chạy duy nhất, 9 mô hình × 3 giao thức, 27 bảng + 10 hình trong
`results/final/` và `results/figures/`, 288+ test xanh. **Demo Streamlit đã chạy
được** (`demo/`) — 3 tab: gợi ý theo user, phim tương tự có lý do, cold-start.

| Hạng mục | Trạng thái |
|---|---|
| Tiền xử lý dữ liệu | ✅ Hoàn tất |
| 9 mô hình (CF 3 + CB 4 + baseline 2) | ✅ Hoàn tất |
| Giao thức đánh giá full-ranking theo lô | ✅ Hoàn tất |
| Bảng benchmark CF / CB / cold-start | ✅ Hoàn tất |
| Kiểm định thống kê (bootstrap + Wilcoxon) | ✅ Hoàn tất |
| Chỉ số ngoài độ chính xác + phân tầng kỷ nguyên | ✅ Hoàn tất |
| Sweep tham số (159 config, đã chạy đủ) | ✅ Hoàn tất |
| Dọn code legacy + tài liệu | 🔄 Đang làm |
| Báo cáo | ✅ Hoàn tất (9 chương + abstract + phụ lục) |

---

## 2. Lịch sử phase

Nguồn chi tiết: `plans/261113-2106-unify-recsys-and-recommendation-cb/`.

| Phase | Nội dung | Trạng thái | Ngày |
|---|---|---|---|
| 00 | Port hạ tầng + pipeline của `recommendation-cb` sang RS | ✅ Complete | 2026-11-14 |
| 01 | Đảm bảo tính đúng của evaluator mới (4 lỗi protocol cũ) | ✅ Complete | 2026-11-14 |
| 02 | Đưa Item-CF và BPR sang giao thức batch chung | ✅ Complete | 2026-11-14 |
| 03 | Chạy bảng so sánh chung CF/CB (9 mô hình × 3 giao thức) | ✅ Complete | 2026-11-14 |
| 04a | Xoá code legacy (evaluator cũ, recommenders cũ, notebooks) | ✅ Complete | 2026-11-14 |
| 04d | Dọn artifact repo (xoá `data/legacy/`, viết lại `.gitignore`) | ✅ Complete | 2026-11-14 |
| 04b | Viết tài liệu kiến trúc + cập nhật docs so sánh | ✅ Complete | 2026-11-14 |
| 04c | Rà soát README mô tả đúng một đường chạy | ✅ Complete | 2026-11-14 |
| 05 | Sửa sai lệch số liệu + commit bằng chứng sweep | ✅ Complete | 2026-11-14 |
| 06 | Viết báo cáo | ✅ Complete (9 chương + abstract + phụ lục) | 2026-11-15 |
| 07 | Tầng serving + demo Streamlit (`src/api/`, `demo/`) | ✅ Complete | 2026-11-15 |

---

## 3. Việc đã hoàn thành — chi tiết

### Phase 00–03: dựng pipeline

- Lọc catalog 13.816 phim có Tag Genome (giữ 98,7% rating).
- Split temporal per-user 80/20 → 158.774 user hợp lệ để đánh giá.
- Chấm điểm theo lô 2.000 user (tránh ma trận 8,9 GB).
- Port 9 mô hình: Random, MostPop, ALS, Item-CF, BPR, TF-IDF, BM25, Tag Genome, Hybrid.
- Chạy sweep **đủ** 159 config trên `dev` (CF 51 / CB 93 / baseline 2 / best 13).
- Chạy `full` và `coldstart` với config đã khoá.
- Kiểm định thống kế: bootstrap CI 95%, Wilcoxon signed-rank trong từng bảng.
- Chỉ số ngoài độ chính xác: coverage, Gini, novelty, ILD, `frac_unseen`.
- Bảng phân tầng kỷ nguyên chống rò rỉ tag.

### Phase 04: dọn dẹp

- Xoá 23 file code legacy: `src/evaluation/`, `src/recommenders/`, `src/data_loader.py`,
  `notebooks/`, `scripts/export_models.py`.
- Banner "tài liệu legacy" cho 8 file `docs/*.md` còn mô tả protocol cũ.
- Xoá `data/legacy/` (461 MB) và 393 artifact trung gian trong `results/`.
- Viết lại `.gitignore`: chỉ giữ `results/final/*.csv` + `results/figures/*.png`.

### Phase 05: sửa sai lệch trước khi viết báo cáo

| Vấn đề | Cách sửa |
|---|---|
| Thứ hạng CB sai trong README ("CB-1 > CB-2 > CB-3") | Sửa thành "CB-2 ≥ CB-1 > CB-3" theo `tag-leakage-strata.csv` |
| Số config sweep ghi 85, thực tế 93 | Sửa ở README, `benchmark_tables.py`, `build_final_tables.py`, phase-03 docs |
| Bảng sweep bị gitignore | Copy 5 bảng `ablation-*.csv` vào `results/final/` để commit |
| 38 metrics CSV thiếu `split_id`/`catalog_id` | Chạy lại 7 config có trong bảng ablation, sinh lại bảng |
| Thiếu 3 file docs theo chuẩn dự án | Viết `codebase-summary.md`, `system-architecture.md`, `development-roadmap.md` |
| Runner thiếu CB-4 | Thêm `best-cb4-hybrid-content.yaml` vào `CONFIGS` + test chặn tách lệch |

### Phase 07: tầng serving + demo Streamlit

| Vấn đề | Cách sửa |
|---|---|
| Mô hình trả `item_idx` thô, không có `movieId`/title | Thêm `src/api/recommender_service.py` — ánh xạ `userId`/`movieId` ra metadata |
| Demo phải fit lại mỗi lần mở (~33s) | `src/api/model_artifacts.py` lưu/nạp artifact (`factors.npz` cho CF, `state.pkl` cho CB) → mở trong ~1,6s |
| User lạ / user mới làm crash | Fallback sang MostPop kèm nhãn lý do |
| Gợi ý trả lại phim user đã xem | Tái dùng `mask_seen_items` — sentinel `-1e9` |
| Gợi ý "phim tương tự" bị nhiễu (dùng profile scorer) | `src/api/item_similarity.py` — cosine trực tiếp trên Tag Genome |
| Demo CF dùng nhầm mô hình fit trên split `full` | Sinh bộ artifact riêng cho split cold-start (`--split coldstart`) |

Kết quả: 3 tab chạy thật, tab cold-start đo được ALS = **0,000000** với phim held-out
trong khi BM25 xếp hạng được (thứ hạng #371 thay vì #4.186 đáy bảng).

### Phase 06: viết báo cáo (đóng gói Markdown)

| Vấn đề | Cách sửa |
|---|---|
| Còn 6 chương + abstract + phụ lục | Viết đủ 9 file trong `docs/report/` theo khung `notes/report-structure.md` |
| Số trong báo cáo có thể "trôi" so với CSV | `scripts/check_report_numbers.py` mở rộng quét **cả 10 file** báo cáo (trước chỉ 2) |
| CSV ghi số lớn bằng dấu phân cách nghìn (`162,540`) | Checker nhận diện giá trị gốc thay vì hiểu là số thập phân |
| Phụ lục 144 dòng dễ chép tay sai | `scripts/build_report_appendix.py` **sinh tự động** từ `results/final/*.csv` |
| Demo cần minh hoạ trong báo cáo | Thêm §6.6 với 3 ảnh chụp màn hình trong `results/figures/demo/` |

Kết quả: `check_report_numbers.py` → **0 lệch** trên 28.028 giá trị / 27 bảng CSV;
`pytest tests/` → **289 test xanh**.

---

## 4. Kết quả đạt được

| Bảng CF (`full`, 158.774 user) | NDCG@10 | | Bảng CB (`full`) | NDCG@10 |
|---|---|---|---|---|
| ALS | **0,1375** | | CB-4 Hybrid | **0,1054** |
| Item-CF (RS) | 0,1290 | | CB-2 BM25 | 0,0998 |
| BPR (RS) | 0,1099 | | CB-1 TF-IDF | 0,0937 |
| MostPop | 0,0739 | | CB-3 Tag Genome | 0,0713 |
| Random | 0,0012 | | | |

| Bảng cold-start item (1.381 phim held-out) | NDCG@10 |
|---|---|
| CB-4 Hybrid | **0,0347** |
| CB-1 TF-IDF | 0,0328 |
| CB-2 BM25 | 0,0306 |
| CB-3 Tag Genome | 0,0305 |
| ALS / Item-CF / BPR / MostPop | **0,0000** |

**Ba kết luận chính:**

1. **CF thắng CB ở giao thức thường** — ALS 0,1375 > CB-4 0,1054 (chênh 23%). Kết luận
   **đã cân bằng**: cả hai nhóm đều quét đủ sweep (CF 51, CB 93).
2. **CB thắng áp đảo ở cold-start item** — mọi mô hình CF = 0,0000 (đúng bằng không, vì
   phim held-out không có tương tác train nào để học), còn CB đạt ≈ 0,035. Luận điểm
   trung tâm của đồ án.
3. **Coverage đảo ngược với độ chính xác** — ALS chính xác nhất nhưng chỉ chạm 14,9%
   catalog; CB-3 kém nhất trong CB nhưng chạm 67,2%.

---

## 5. Việc còn lại

### 5.1 Bắt buộc trước khi nộp báo cáo

| Việc | Ghi chú |
|---|---|
| Viết báo cáo | **Đã xong** — 9 chương + abstract + phụ lục trong `docs/report/` (mục lục: `docs/report/README.md`) |
| Nêu rõ giới hạn metadata hậu kỳ | Bắt buộc theo điều kiện của phase 00 — **đã viết** ở Chương 6 §6.1 và mục giới hạn §6.5.2 |
| Nêu rõ `pop_prior` giết cold-start | Bảng chính lấy bản content thuần; **đã giải thích** ở Chương 5 §5.3.2 |
| Nêu rõ BPR `num_threads=1` | Số trong bảng là số tái lập được — **đã viết** ở Chương 4 §4.5 và Chương 2 §2.2.5 |

### 5.2 Tùy chọn (nếu còn thời gian)

| Việc | Chi phí | Giá trị |
|---|---|---|
| Chạy sampled LOO → Kendall τ | ~20 phút | Biến lựa chọn "full ranking" từ trích dẫn thành **kết quả thực nghiệm** của đồ án. Script **đã sửa** (dedup `exp_name`, đọc metric gốc `results/{exp_name}_metrics.csv`) và **chưa chạy** — cần chạy rồi điền vào Chương 7 §7.4 |
| Bảng so sánh CF↔CB dùng Wilcoxon chung | Thấp | Hiện Wilcoxon chỉ chạy trong từng bảng — cần bảng riêng vì đây là câu hỏi khác |
| Chạy `full` cho các config còn lại của sweep | Cao (~50 phút) | Chỉ cần nếu muốn bảng sweep ở `full` thay vì `dev` |

### 5.3 Ngoài phạm vi (đã chốt)

- Deploy thật (demo chạy local, không có server công khai)
- Báo cáo DOCX (chỉ Markdown)
- Mô hình hybrid CF+CB (late fusion hai nhóm)
- Bảng no-leak (mất 66% tập user)
- Biến thể "giới hạn số tag/phim" của CB-1
- Kiểm định Mann-Whitney giữa các tầng thời kỳ

> **Cập nhật 2026-11-15:** "Demo Streamlit" **đã làm** (Phase 07) và chuyển khỏi
> danh sách ngoài phạm vi. Demo dùng 2 trong 9 mô hình (ALS + BM25), 3 tab; xem
> [`demo/README.md`](../demo/README.md). Vẫn **không** làm deploy thật, API HTTP
> công khai, hay đăng nhập người dùng.

---

## 6. Rủi ro đã biết

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Metadata hậu kỳ làm số tuyệt đối lạc quan | Trung bình | Đo bằng bảng phân tầng kỷ nguyên + Pearson với popularity; **phải nêu trong báo cáo** |
| Cold-start là mô phỏng, không phải phim mới thật | Trung bình | Ghi rõ: 10% phim bị giữ lại theo 5 nhóm năm phát hành, mọi user vẫn có lịch sử train |
| Người clone repo không chạy lại được kiểm định thống kê | Thấp | README ghi rõ phải chạy lại pipeline (thiếu `*_peruser.npz` / `*_topk.npy`) |
| Không có CI | Thấp | `pytest tests/` chạy 2,5 phút; có thể thêm workflow sau khi nộp |
| `notes/`/`plans/` bị gitignore nhưng 18 file đã track từ trước | Thấp | Đã ghi nhận trong `.gitignore`; quyết định giữ nguyên theo user 2026-11-14 |
