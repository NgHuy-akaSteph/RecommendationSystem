# Phụ lục

> Tất cả số trong phụ lục sinh **tự động** từ `results/final/*.csv` bằng `scripts/build_report_appendix.py` — không chép tay.

---

## A. Bảng sweep đầy đủ (144 dòng)

Bảng sweep không nhúng vào báo cáo vì quá dài (144 dòng × nhiều cột tham số). Toàn bộ **đã commit** trong repo:

| File | Số dòng | Nội dung |
|---|---:|---|
| `results/final/ablation-cf.csv` | 51 | 4 mô hình, mỗi dòng mang `params` + `split_id`/`catalog_id`/`seed` |
| `results/final/ablation-cb1.csv` | 15 | 1 mô hình, mỗi dòng mang `params` + `split_id`/`catalog_id`/`seed` |
| `results/final/ablation-cb2.csv` | 21 | 1 mô hình, mỗi dòng mang `params` + `split_id`/`catalog_id`/`seed` |
| `results/final/ablation-cb3.csv` | 22 | 2 mô hình, mỗi dòng mang `params` + `split_id`/`catalog_id`/`seed` |
| `results/final/ablation-cb4.csv` | 35 | 2 mô hình, mỗi dòng mang `params` + `split_id`/`catalog_id`/`seed` |

**Tổng: 144 dòng = 51 CF + 93 CB.**

### A.1. Biên độ sweep **theo model**

Quy ước: biên độ = **(max − min) / max**, tính trên NDCG@10 của giao thức `dev` (20.000 user, `split_id=temporal80_20`, `seed=42`).

| Model | n | min | max | Biên độ | Config tốt nhất |
|---|---:|---:|---:|---:|---|
| `cf-als` | 15 | 0,1135 | 0,1383 | **17,94%** | `cf-als-f064-a005-reg0p1` |
| `cf-itemknn` | 3 | 0,1263 | 0,1297 | **2,57%** | `cf-itemknn-k200` |
| `rs-itemcf` | 12 | 0,1056 | 0,1297 | **18,58%** | `rs-itemcf-k200-binary-pos` |
| `rs-bpr` | 21 | 0,0364 | 0,1106 | **67,07%** | `rs-bpr-f64-i10-reg0p00001-lr0p01` |
| `cb4-hybrid` | 32 | 0,0726 | 0,1067 | **32,00%** | `cb4-06-late-bm25-pop-lam0040` |
| `cb2-bm25` | 21 | 0,0821 | 0,0986 | **16,82%** | `cb2-01-k1-05-b000` |
| `cb1-tfidf` | 15 | 0,0036 | 0,0914 | **96,06%** | `cb1-01-add-tags-tfidf` |
| `cb3-genome` | 19 | 0,0067 | 0,0703 | **90,45%** | `cb3-04d-item-knn-tau020-k100` |
| `cb3-ridge` | 3 | 0,0030 | 0,0125 | **75,98%** | `cb3-05-ridge-lambda100` |

> **Không trộn hai quy ước.** Biên độ *theo model* tính trên lưới tham số của riêng mô hình đó; biên độ *theo nhóm* (§4.2.2) tính trên toàn bảng ablation của nhóm. Hai con số khác nhau và cả hai đều đúng.

---

## B. Bootstrap CI 95% và kiểm định Wilcoxon

### B.1. Bootstrap CI 95% (giao thức `full`, 158.774 user)

*Nguồn: `results/final/bootstrap-ci.csv`.*

| `exp_name` | NDCG@10 | CI 95% thấp | CI 95% cao | n users |
|---|---:|---:|---:|---:|
| `full_cf-als-f064-a005-reg0p1` | 0,1375 | 0,1366 | 0,1383 | 158.774 |
| `full_cf-itemknn-k200` | 0,1290 | 0,1281 | 0,1299 | 158.774 |
| `full_rs-itemcf-k200-binary-pos` | 0,1290 | 0,1281 | 0,1299 | 158.774 |
| `full_rs-bpr-f64-i10-reg0p00001-lr0p01` | 0,1099 | 0,1092 | 0,1107 | 158.774 |
| `full_cb4-06-late-bm25-pop-lam0040` | 0,1081 | 0,1073 | 0,1089 | 158.774 |
| `full_cb4-04-late-bm25-w075` | 0,1054 | 0,1047 | 0,1062 | 158.774 |
| `full_cb2-01-k1-05-b000` | 0,0998 | 0,0990 | 0,1005 | 158.774 |
| `full_cb4-03-timedecay-hl0365` | 0,0937 | 0,0930 | 0,0944 | 158.774 |
| `full_baseline-mostpop` | 0,0739 | 0,0732 | 0,0746 | 158.774 |
| `full_cb3-04d-item-knn-tau020-k100` | 0,0713 | 0,0707 | 0,0720 | 158.774 |
| `full_cb3-05-ridge-lambda100` | 0,0129 | 0,0126 | 0,0131 | 158.774 |
| `full_cb1-00-genres-binary-profile` | 0,0040 | 0,0039 | 0,0042 | 158.774 |
| `full_baseline-random` | 0,0012 | 0,0011 | 0,0012 | 158.774 |

**Đọc kết quả:** CI của ALS [0,1366; 0,1383] **không chạm** CI của CB-4 [0,1047; 0,1062] — khoảng cách giữa CF và CB lớn hơn nhiễu thống kê.

### B.2. Wilcoxon signed-rank — CF

*Nguồn: `results/final/wilcoxon-cf.csv`.*

| Mốc | So với | Chênh lệch TB | % user tốt hơn | % user kém hơn | p-value |
|---|---|---:|---:|---:|---:|
| ALS | full_baseline-random | 0,1363 | 53,52% | 0,41% | <1e-300 |
| ALS | full_baseline-mostpop | 0,0636 | 43,36% | 16,97% | <1e-300 |
| ALS | full_rs-bpr-f64-i10-reg0p00001-lr0p01 | 0,0275 | 36,33% | 24,54% | <1e-300 |
| ALS | full_rs-itemcf-k200-binary-pos | 0,0085 | 32,86% | 26,70% | 2.9e-182 |

### B.3. Wilcoxon signed-rank — CB

*Nguồn: `results/final/wilcoxon-cb.csv`.*

| Mốc | So với | Chênh lệch TB | % user tốt hơn | % user kém hơn | p-value |
|---|---|---:|---:|---:|---:|
| CB-4 Hybrid | full_cb3-04d-item-knn-tau020-k100 | 0,0341 | 32,32% | 13,90% | <1e-300 |
| CB-4 Hybrid | full_cb4-03-timedecay-hl0365 | 0,0117 | 26,05% | 17,73% | <1e-300 |
| CB-4 Hybrid | full_cb2-01-k1-05-b000 | 0,0057 | 23,94% | 19,18% | 1.47e-225 |

### B.4. Wilcoxon signed-rank — Cold-start

*Nguồn: `results/final/wilcoxon-coldstart.csv`.*

| Mốc | So với | Chênh lệch TB | % user tốt hơn | % user kém hơn | p-value |
|---|---|---:|---:|---:|---:|
| CB-4 Hybrid | coldstart_baseline-mostpop | 0,0347 | 11,08% | 0,00% | 8.77e-201 |
| CB-4 Hybrid | coldstart_rs-itemcf-k200-binary-pos | 0,0347 | 11,08% | 0,00% | 8.77e-201 |
| CB-4 Hybrid | coldstart_cf-als-f064-a005-reg0p1 | 0,0347 | 11,08% | 0,00% | 8.77e-201 |
| CB-4 Hybrid | coldstart_rs-bpr-f64-i10-reg0p00001-lr0p01 | 0,0347 | 11,08% | 0,00% | 8.77e-201 |
| CB-4 Hybrid | coldstart_baseline-random | 0,0344 | 11,07% | 0,15% | 1.07e-195 |
| CB-4 Hybrid | coldstart_cb3-04d-item-knn-tau020-k100 | 0,0042 | 7,49% | 5,77% | 1.04e-06 |
| CB-4 Hybrid | coldstart_cb2-01-k1-05-b000 | 0,0041 | 6,88% | 4,67% | 7.07e-12 |
| CB-4 Hybrid | coldstart_cb4-03-timedecay-hl0365 | 0,0019 | 6,65% | 5,58% | 0.00392 |

### B.5. Wilcoxon chéo CF↔CB — ALS (CF) đối chiếu từng mô hình CB

*Nguồn: `results/final/wilcoxon-cross-cf-cb.csv` — giao thức `full`, 158.774 user.*

| Mô hình CB | Chênh lệch TB | % user ALS tốt hơn | % user CB tốt hơn | % hòa | rank-biserial | p-value |
|---|---:|---:|---:|---:|---:|---:|
| CB-3 Tag Genome | 0,0662 | 43,25% | 16,05% | 40,71% | 0,4587 | <1e-300 |
| CB-1 TF-IDF | 0,0438 | 39,38% | 20,80% | 39,82% | 0,3089 | <1e-300 |
| CB-2 BM25 | 0,0377 | 38,45% | 21,94% | 39,61% | 0,2733 | <1e-300 |
| CB-4 Hybrid | 0,0320 | 37,06% | 23,28% | 39,67% | 0,2284 | <1e-300 |

**Đọc kết quả:** ALS thắng **cả 4** mô hình CB với `p < 1e-300`. Đây là kiểm định **chéo CF↔CB** — khác với B.2–B.4 vốn kiểm định *trong từng bảng*.

> **Giới hạn:** kiểm định Wilcoxon ở B.2–B.4 được chạy **trong từng bảng**, **không** so chéo CF↔CB — vì hai nhóm được tinh chỉnh với số config khác nhau, nên so chéo sẽ trộn hiệu ứng của *thuật toán* với hiệu ứng của *mức độ tinh chỉnh*. Bảng B.5 bên dưới trả lời câu hỏi khác: hai *họ* thuật toán có khác nhau theo từng user không.

---

## C. Thống kê dataset đầy đủ

*Nguồn: `results/final/dataset-summary.csv`.*

| Chỉ số | Giá trị |
|---|---:|
| Số user | 162,540 |
| Số item | 13,816 |
| Số rating | 24,674,113 |
| Mật độ ma trận | 1.0988% |
| Rating >= 4 | 50.1% |
| Rating trung bình | 3.54 |
| Rating/user (trung vị) | 71 |
| Rating/user (p90) | 351 |
| Rating/item (trung vị) | 288 |
| Rating thuộc 20% phim phổ biến nhất | 84.4% |
| Item có >= 1 tag | 13,633 |
| Khoảng thời gian | 1995-2019 |
| Số thể loại | 19 |
| Số tag genome | 1,128 |
| Relevance genome trung bình | 0.116 |
| Cosine trung bình giữa hai phim (genome thô) | 0.61 |
| Lượt gán tag (sau làm sạch) | 937,428 |
| Từ tag khác nhau (sau làm sạch) | 30,067 |
| Tag/phim (trung vị) | 21 |
| Rating train | 19,801,784 |
| Rating test (user hợp lệ) | 4,831,397 |
| Rating test liên quan (>= 4) | 2,288,027 |
| User hợp lệ để đánh giá | 158,774 |
| Phim liên quan mỗi user trong test (trung vị) | 8 |
| User tập phát triển | 20,000 |

---

## D. Quy trình tái lập

Môi trường: Python 3.12, seed = 42. Hướng dẫn đầy đủ (cấu trúc repo, mọi lệnh): [`docs/codebase-summary.md`](../codebase-summary.md).

```bash
# 1. Tiền xử lý
venv/bin/python -m src.data.load_movielens_filtered
venv/bin/python -m src.data.build_item_content_features
venv/bin/python -m src.data.split_temporal_per_user

# 2. Thí nghiệm (một config ví dụ)
venv/bin/python -m src.eval.run_experiment_from_config configs/best-cf-als.yaml

# 3. Gom bảng kết quả + hình
venv/bin/python scripts/build_final_tables.py
venv/bin/python -m src.eval.make_result_figures

# 4. Kiểm chứng
venv/bin/python scripts/check_report_numbers.py   # phải báo 0 lệch
venv/bin/python -m pytest tests/ -q
```

---

## E. So sánh giao thức đánh giá — full ranking vs sampled LOO

*Nguồn: `results/final/sampled-vs-full.csv` — `scripts/run_sampled_loo_comparison.py`.*

| `exp_name` | NDCG@10 full | NDCG@10 LOO | Hạng full | Hạng LOO |
|---|---:|---:|---:|---:|
| `cf-als-f064-a005-reg0p1` | 0,1383 | 0,5959 | 1 | 3 |
| `cf-itemknn-k200` | 0,1297 | 0,5973 | 2 | 1 |
| `rs-itemcf-k200-binary-pos` | 0,1297 | 0,5973 | 2 | 1 |
| `rs-bpr-f64-i10-reg0p00001-lr0p01` | 0,1106 | 0,5030 | 4 | 4 |
| `cb4-06-late-bm25-pop-lam0040` | 0,1067 | 0,4924 | 5 | 5 |
| `cb4-04-late-bm25-w075` | 0,1043 | 0,4700 | 6 | 6 |
| `cb2-01-k1-05-b000` | 0,0986 | 0,4541 | 7 | 7 |
| `cb4-03-timedecay-hl0365` | 0,0921 | 0,4335 | 8 | 9 |
| `baseline-mostpop` | 0,0723 | 0,4491 | 9 | 8 |
| `cb3-04d-item-knn-tau020-k100` | 0,0703 | 0,3472 | 10 | 10 |
| `cb3-05-ridge-lambda100` | 0,0125 | 0,1878 | 11 | 11 |
| `baseline-random` | 0,0011 | 0,0453 | 12 | 12 |

**Đọc kết quả:** Kendall τ = 0,908 (`p = 4,81e-05`) giữa thứ hạng full-ranking và sampled LOO. Hàng rào này **bảo vệ** lựa chọn full ranking: thứ tự xếp hạng được giữ, nên kết luận xếp hạng của đồ án không đổi nếu dùng LOO.

---

## F. Cách kiểm chứng số liệu trong báo cáo

Mọi con số trong báo cáo đều lấy từ `results/final/*.csv` (31 bảng, đã commit). Script kiểm chứng đối chiếu báo cáo với CSV:

```bash
venv/bin/python scripts/check_report_numbers.py
```

Kết quả mong đợi: **0 lệch** trên toàn bộ giá trị từ các bảng CSV.

Bổ sung:

- Biên độ sweep trong báo cáo được **tính động** từ `ablation-*.csv` (`src/eval/sweep_spread.py`), không hard-code — người đọc chạy lại là tự xác minh.
- `pytest tests/test_report_sweep_spreads.py tests/test_report_numbers_consistency.py` kiểm tra hồi quy tài liệu.

**Giới hạn của cơ chế này:** script kiểm tra *số đã ghi trong báo cáo* khớp *số trong CSV*. Nó **không** kiểm tra báo cáo có **đủ** mọi bảng hay không, và không kiểm tra diễn giải văn bản có đúng với số hay không — phần đó là trách nhiệm của người đọc.
