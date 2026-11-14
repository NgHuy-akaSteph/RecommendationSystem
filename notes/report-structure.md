# Khung Báo Cáo — Recommendation System (MovieLens 25M)

**Tổng số trang:** 40–45 trang
**Dataset:** MovieLens 25M (lọc còn 13.816 phim có Tag Genome)
**Mô hình:** 9 mô hình — 3 CF, 4 Content-based, 2 baseline
**Nguồn sự thật:** mọi số liệu trong khung này lấy từ `results/final/` (27 bảng) và
`results/figures/` (9 hình), đã commit trong repo. Không chép tay số.

> **Khung này thay thế hoàn toàn** `notes/report-structure.md` cũ (viết cho MovieLens 1M,
> 2 methods, 18 config — lệch toàn bộ với hiện trạng).

---

## Abstract (1 trang)

**Người phụ trách:** Người Phụ C

**Nội dung — 4 khối, mỗi khối 2–3 câu:**
1. **Bối cảnh:** CF và Content-based là hai hướng chính; mỗi hướng thắng ở một vùng
   bài toán (CF khi lịch sử tương tác dày, CB khi phim mới chưa có tương tác).
2. **Phương pháp:** hợp nhất 9 mô hình trên **một giao thức đánh giá duy nhất** (full
   ranking, split thời gian 80/20 theo từng user, ngưỡng relevant ≥ 4,0), với 3 giao
   thức (`dev` / `full` / `coldstart`) và bằng chứng protocol `split_id` + `catalog_id`
   trên từng dòng kết quả.
3. **Kết quả chính:** CF thắng CB ở giao thức thường (ALS 0,1375 vs CB-4 0,1054 —
   chênh 23%, đã cân bằng sau sweep 51 CF / 93 CB config); **CB thắng áp đảo ở
   cold-start item** (mọi mô hình CF = 0,0000, CB đạt 0,0347).
4. **Đóng góp:** kết luận "CF thắng" trước đây bị nghi ngờ thiên lệch về độ bao phủ
   sweep — báo cáo này cân bằng được điều đó, đồng thời phân tầng 3 kỷ nguyên để chứng
   minh tín hiệu content không đảo chiều theo thời gian (chống rò rỉ metadata hậu kỳ).

**Từ khóa:** recommender systems, collaborative filtering, content-based filtering,
cold-start, MovieLens 25M, offline evaluation, NDCG

---

## Chapter 1: Introduction (3 trang)

**Người phụ trách:** Người Phụ A

### 1.1 Bối cảnh và động lực
- Vai trò personalization trong streaming/ecommerce
- Bài toán xếp hạng (top-N) khác bài toán dự đoán rating — ảnh hưởng trực tiếp tới
  lựa chọn metric (không dùng RMSE/MAE)

### 1.2 Câu hỏi nghiên cứu
> **RQ1:** Với cùng một giao thức đánh giá và độ bao phủ sweep tương đương, CF và
> Content-based thắng ở đâu?
> **RQ2:** Điều gì xảy ra khi phim mới chưa có tương tác nào (cold-start item)?
> **RQ3:** Liệu kết luận có bị đảo chiều nếu phân tầng theo thời kỳ người dùng —
> tức là liệu tín hiệu content có "biết trước" phim tương lai không?

### 1.3 Đóng góp của đồ án
1. **Một giao thức, một đường chạy** — không còn hai evaluator song song như ở protocol cũ
2. **Bằng chứng protocol trên từng dòng kết quả** (`split_id` + `catalog_id` + `seed`)
3. **Sweep cân bằng** (51 CF / 93 CB) — biến kết luận "thắng" từ thiên lệch thành có cơ sở
4. **Phân tầng kỷ nguyên** — đo trực tiếp rủi ro rò rỉ metadata hậu kỳ
5. **Kết luận đảo chiều ở cold-start** — nêu rõ điều kiện để mỗi hướng thắng

### 1.4 Phạm vi và giới hạn
| Phạm vi | Ngoài phạm vi |
|---|---|
| MovieLens 25M, explicit rating, offline | Deep learning, session-based |
| 9 mô hình (không deep learning) | Demo/deploy, A/B test |
| Full ranking, 3 giao thức | Bảng no-leak (mất 66% user) |
| Metadata hậu kỳ — **phải nêu rõ** | Hybrid CF+CB ở tầng late fusion |

### 1.5 Cấu trúc báo cáo
- Mỗi chương trả lời RQ nào (đặt ở đầu chương)

---

## Chapter 2: Literature Review & Theoretical Background (8 trang)

**Người phụ trách:** Người 1 (2.2) + Người 2 (2.3–2.4) + Người Phụ A (2.1)

### 2.1 Tổng quan về hệ thống gợi ý (1 trang)
- Định nghĩa, ứng dụng, phân loại: Content-based / CF / Hybrid / Knowledge-based
- Bảng so sánh nhanh ưu – nhược – điều kiện thắng

### 2.2 Collaborative Filtering (3,5 trang)
**Người 1**

#### 2.2.1 User-based CF (0,5 trang)
- Trực giác, độ đo tương đồng (cosine, Pearson), ưu/nhược

#### 2.2.2 Item-based CF (1 trang)
- Trực giác "người xem phim này cũng xem phim kia", vì sao scale tốt hơn user-based
- Quan hệ với hàng đầu mục trong đồ án: `rs-itemcf` (trọng số rating thật) và
  `cf-itemknn` (trọng số nhị phân)

#### 2.2.3 Matrix Factorization (1 trang)
- Phân rã user–item thành latent factor, trực giác

#### 2.2.4 ALS cho implicit feedback (1 trang)
- Hu, Koren & Volinsky (2008): confidence weighting
- **Vì sao chỉ dùng tương tác relevant** (rating ≥ 4) → `c_ui = 1 + α`, không phải `1 + α·rating`
- Tóm tắt thuật toán (không nặng toán)

**Tài liệu tham khảo chính:**
- Hu, Koren, Volinsky (2008). Collaborative filtering for implicit feedback datasets. *ICDM*.
- Koren, Bell, Volinsky (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8).

#### 2.2.5 BPR-MF (0,5 trang)
- Rendle et al. (2009): pairwise ranking loss, negative sampling
- Vì sao `num_threads=1` (tái lập được tuyệt đối, đánh đổi 40s thay vì 7s)

### 2.3 Content-based Filtering (2 trang)
**Người 2**

#### 2.3.1 Biểu diễn đặc trưng (0,5 trang)
- 3 nguồn: genres (19), tag (30.067 từ sau làm sạch), Tag Genome (1.128 chiều)

#### 2.3.2 TF-IDF và BM25 (1 trang)
- Công thức TF-IDF, định nghĩa BM25 (`k1`, `b`)
- **Vì sao BM25 hay hơn TF-IDF ở đây** — bằng chứng từ `tag-leakage-strata.csv`

#### 2.3.3 Cách dựng hồ sơ user (0,5 trang)
- Trung bình có trọng số của vector item; 3 chế độ trọng số `binary_pos` /
  `centered` / `rating` (+ time-decay)

#### 2.3.4 Tag Genome và vấn đề metadata hậu kỳ (0,5 trang)
- Nhóm tính ngoài dõi, không có timestamp
- **Đây là mục rủi ro phải nêu trước khi trích dẫn kết quả CB**

### 2.4 Metric đánh giá bài toán xếp hạng (1,5 trang)
**Người 2**
- Vì sao không dùng RMSE/MAE
- Precision@k, Recall@k, HR@k, NDCG@k (định nghĩa Järvelin & Kekäläinen 2002)
- **Định nghĩa đúng `P@k = hits / k`** — phân biệt với `hits / len(top_k)`
- Metric ngoài độ chính xác: Coverage, Gini, Novelty, ILD, `frac_unseen`
- Kiểm định thống kê: bootstrap CI 95%, Wilcoxon signed-rank ghép cặp

---

## Chapter 3: Dataset & Preprocessing (5 trang)

**Người phụ trách:** Người 1

### 3.1 MovieLens 25M — tổng quan (1,5 trang)

#### Nội dung 7 file thô
| File | Dòng | Nội dung |
|---|---:|---|
| `ratings.csv` | 25.000.095 | `userId,movieId,rating,timestamp` |
| `movies.csv` | 62.423 | `movieId,title,genres` |
| `tags.csv` | 1.093.360 | tag do người dùng tự gán (có timestamp) |
| `genome-scores.csv` | 15.584.448 | Tag Genome |
| `genome-tags.csv` | 1.128 | tên tag của genome |

- Khoảng thời gian 1995–2019; 162.540 user, 62.423 phim; giấy phép GroupLens (cấm phân phối lại)

#### Thống kê sau tiền xử lý (bảng `dataset-summary.csv`)
| Chỉ số | Giá trị |
|---|---:|
| Số user | 162.540 |
| Số item | 13.816 |
| Số rating | 24.674.113 |
| Mật độ ma trận | 1,0988% |
| Rating ≥ 4 | 50,1% |
| Rating trung bình | 3,54 |
| Rating/user (trung vị / p90) | 71 / 351 |
| Rating thuộc 20% phim phổ biến nhất | 84,4% |
| Tag/phim (trung vị) | 21 |

### 3.2 Phân tích khám phá dữ liệu (1,5 trang)

**Hình cần dùng (đã có trong `results/figures/`):**
1. `eda-distributions.png` — phân phối rating, phân phối tương tác/user
2. `eda-long-tail.png` — đường cong long-tail của độ phổ biến phim
3. `eda-content-features.png` — phân bố thể loại / số tag / độ dài tài liệu
4. `eda-genome.png` — phân bố điểm genome, độ tương quan cosine giữa hai phim

**Phát hiện chính:**
- Long-tail rất lệch (84,4% rating nằm trong 20% phim phổ biến nhất) → CF dễ dồn suất
- 183 phim không có tag → cần xử lý riêng
- Tag bắt đầu xuất hiện từ **2005-12-24** — mốc quan trọng cho Chương 6

### 3.3 Tiền xử lý (2 trang)

#### 3.3.1 Lọc catalog theo Tag Genome (0,5 trang)
- Giữ 13.816 phim có genome, **giữ 98,7% khối lượng rating**
- **Lý do:** mọi mô hình phải so sánh trên *cùng* tập ứng viên — nếu không, CB-3
  không thể chạy và so sánh không công bằng

#### 3.3.2 Làm sạch đặc trưng nội dung (0,5 trang)
- 1.093.360 lượt gán tag → 937.428 lượt sau làm sạch → 30.067 từ khác nhau
- 13.633/13.816 phim có ≥ 1 tag

#### 3.3.3 Chia tập thời gian theo từng user (1 trang)
- **Phương pháp:** trong mỗi user, sort theo timestamp, lấy 80% đầu → train, 20% cuối → test
- **Điều kiện giữ lại user:** train ≥ 5 tương tác **và** test có ≥ 1 rating ≥ 4
- **Kết quả:** 158.774 user hợp lệ (train 19.801.784 rating, test 4.831.397 rating,
  trong đó 2.288.027 rating relevant)

> **Điểm phải nhấn:** split global 70/30 của protocol cũ chỉ đánh giá được **4.587 user**
> (phần lớn user ML-25M ngừng hoạt động trước 2015 nên không có dữ liệu test).
> Split per-user đánh giá được **158.774 user** — tăng 34,6 lần số user đánh giá.

#### 3.3.4 Dựng ma trận và chuẩn hoá chỉ số (0,5 trang)
- Ma trận thưa CSR, remap chỉ số user/item về 0..n-1
- Ngưỡng relevant: rating ≥ 4,0 (`RELEVANCE_THRESHOLD`)

---

## Chapter 4: Methodology (8 trang)

**Người phụ trách:** Người 1 (4.1–4.2, 4.4) + Người 2 (4.3, 4.5)

### 4.1 Thiết lập thực nghiệm (1 trang)
**Người 1**
- Môi trường: Python 3.12, `implicit`, `scipy`, `pandas`, `numpy`; seed = 42
- Cấu trúc repo, hướng dẫn chạy lại (`docs/codebase-summary.md`)
- **Tại sao cấu trúc `TrainData`/`Dataset` quan trọng** — mô hình không có đường nào chạm nhãn test

### 4.2 Baseline (0,5 trang)
**Người 1**
- **Random (sanity):** kiểm tra evaluator — mọi metric phải ≈ 0
- **MostPop:** xếp hạng theo số lượt rating relevant trong train

### 4.3 Mô hình Content-based (3 trang)
**Người 2**

| Mô hình | Cấu hình | Ghi chú |
|---|---|---|
| CB-1 TF-IDF | TF-IDF trên tag + genres | sweep 15 config |
| CB-2 BM25 | BM25 trên tag, lưới `k1` × `b` | sweep 21 config |
| CB-3 Tag Genome | 1.128 chiều genome, chuẩn hoá z-score/min-max, SVD | sweep 22 config |
| CB-4 Hybrid | Late fusion BM25 + Genome | sweep 35 config |

- **Cơ chế late fusion:** chuẩn hoá điểm rồi tổng trọng số (`w`)
- **Bẫy `pop_prior`:** bản có λ_pop = 0,4 đạt NDCG@10 = 0,1081 ở giao thức thường
  (cao hơn bản content thuần 0,1054) nhưng **sụp về 0,0010 ở cold-start** vì mọi phim
  held-out có popularity = 0 → `frac_unseen@10` rơi từ 0,2223 xuống 0,0036.
  → **Bảng chính lấy bản content thuần**; bản có prior chỉ để đối chiếu.

### 4.4 Mô hình Collaborative Filtering (2 trang)
**Người 1**

| Mô hình | Thư viện / cài đặt | Ghi chú |
|---|---|---|
| ALS | `implicit.als.AlternatingLeastSquares` | `c_ui = 1 + α`, chỉ relevant, `factors`/`α`/`reg`/`iterations` |
| Item-CF (RS) | tự cài, trọng số = **rating thật** | sweep `k_neighbors`, `weight_mode` |
| BPR-MF | `implicit.bpr` | `num_threads=1` — tái lập được tuyệt đối |
| ItemKNN-CF | tự cài, trọng số **nhị phân** | trong registry, không vào bảng chính |

### 4.5 Giao thức đánh giá (1,5 trang)
**Người 2**

#### 4.5.1 Ba giao thức
| Giao thức | User | Dùng để |
|---|---:|---|
| `dev` | 20.000 | **chọn tham số** (sweep 159 config) |
| `full` | 158.774 | bảng chính — chạy **một lần** sau khi khoá config |
| `coldstart` | 10.992 | cold-start item (1.381 phim giữ lại theo 5 nhóm năm phát hành) |

#### 4.5.2 Vì sao full ranking, không sampled
- Chấm trên toàn bộ 13.816 ứng viên; sampled LOO có thể đảo thứ hạng (Krichene & Rendle 2020)
- Giao thức thay thế LOO + 100 negative đã cài nhưng **chưa chạy** — nêu rõ trong phần giới hạn

#### 4.5.3 Bằng chứng protocol
- Mỗi dòng kết quả mang `split_id` + `catalog_id` + `seed`
- **Không có các cột này, "cùng giao thức" chỉ là lời tuyên bố**

---

## Chapter 5: Experimental Results (8 trang)

**Người phụ trách:** Người 1 (5.2) + Người 2 (5.3)

### 5.1 Kết quả chính — bảng chung (2 trang)
**Người 2**

**Bảng CF (`full`, 158.774 user) — `benchmark-cf.csv`:**
| Mô hình | NDCG@10 | | Bảng CB (`full`) — `benchmark-cb.csv`: | NDCG@10 |
|---|---:|---|---|---:|
| ALS | **0,1375** | | CB-4 Hybrid | **0,1054** |
| Item-CF (RS) | 0,1290 | | CB-2 BM25 | 0,0998 |
| BPR (RS) | 0,1099 | | CB-1 TF-IDF | 0,0937 |
| MostPop | 0,0739 | | CB-3 Tag Genome | 0,0713 |
| Random | 0,0012 | | | |

**Hình:** `results-accuracy.png`, `results-metric-vs-k.png`

**Diễn giải:** CF thắng CB-4 **23%** ở `full`.

### 5.2 Cân bằng sweep — kết luận "CF thắng" có cơ sở chưa? (2 trang)
**Người 1**

| Nhóm | Config trong `configs/` | Đã chạy | Chưa chạy |
|---|---:|---:|---:|
| CF (ALS + Item-CF + BPR) | 51 | **51** | 0 |
| CB (CB-1…CB-4) | 93 | **93** | 0 |

- **Bằng chứng:** `results/final/ablation-{cf,cb1,cb2,cb3,cb4}.csv` (144 dòng = 51 CF + 93 CB,
  mỗi dòng mang `params` + `split_id`/`catalog_id`) — người đọc kiểm chứng được mà không phải tin lời
- **Quy ước biên độ sweep:** công thức `(max − min) / max`, tính trên **NDCG@10** của
  giao thức `dev` (20.000 user, `split_id=temporal80_20`, `seed=42`). Mọi biên độ dưới
  đây đều dùng chung công thức này.
- **Biên độ theo model** (mỗi model là một lưới tham số riêng):

  | Model | n | min → max | Biên độ |
  |---|---:|---|---:|
  | `cf-als` | 15 | 0,1135 → 0,1383 | **17,94%** |
  | `cf-itemknn` | 3 | 0,1263 → 0,1297 | **2,57%** |
  | `rs-bpr` | 21 | 0,0364 → 0,1106 | **67,07%** |
  | `rs-itemcf` | 12 | 0,1056 → 0,1297 | **18,58%** |
  | `cb1-tfidf` | 15 | 0,0036 → 0,0914 | **96,06%** |
  | `cb2-bm25` | 21 | 0,0821 → 0,0986 | **16,82%** |
  | `cb3-genome` | 19 | 0,0067 → 0,0703 | **90,45%** |
  | `cb3-ridge` | 3 | 0,0030 → 0,0125 | **75,98%** |
  | `cb4-hybrid` | 32 | 0,0726 → 0,1067 | **32,00%** |

- **Biên độ theo nhóm** (toàn bảng ablation của nhóm):

  | Nhóm | File | n | min → max | Biên độ |
  |---|---|---:|---|---:|
  | CF | `ablation-cf.csv` | 51 | 0,0364 → 0,1383 | **73,65%** |
  | CB-1 | `ablation-cb1.csv` | 15 | 0,0036 → 0,0914 | **96,06%** |
  | CB-2 | `ablation-cb2.csv` | 21 | 0,0821 → 0,0986 | **16,82%** |
  | CB-3 | `ablation-cb3.csv` | 22 | 0,0030 → 0,0703 | **95,72%** |
  | CB-4 | `ablation-cb4.csv` | 35 | 0,0726 → 0,1067 | **32,00%** |

  > ⚠️ **Hai quy ước này không được trộn.** "CF 17,94%" là biên độ của **riêng lưới
  > ALS** (mẫu số max); biên độ của **toàn nhóm CF** là **73,65%** vì nhóm gồm thêm
  > BPR (biên độ 67,07%). Tương tự, CB-3 (95,72%) gồm cả lưới `cb3-ridge` (75,98%).
  > ⚠️ `ablation-cb4.csv` chứa **3 dòng `model=cb1-tfidf`** (họ time-decay `cb4-03-*`)
  > — đây là họ time-decay đặt trên nền CB-1, **không phải** hybrid; biên độ nhóm
  > CB-4 vẫn đúng vì 32/35 dòng là `cb4-hybrid`.
- **Đối chiếu `dev` ↔ `full`:** `dev` (20.000 user) là tập con của `full` (158.774 user),
  cùng `train.parquet`/`test.parquet` — khác biệt duy nhất là ai được chấm điểm. NDCG@10
  lệch ≤ 1,7% và **thứ hạng không đảo** → dùng biên độ sweep (`dev`) trong báo cáo là
  hợp lệ, nhưng **phải ghi rõ mức user của từng bảng**.
- **Điểm quan trọng về ALS:** nhạy với `α` (α=40 → 0,113470, tức −18% so với α=5 =
  0,137905) chứ **không** nhạy với `reg`/`iterations` (biên độ 0,40% trong lát cắt đó).
  Biên độ **toàn lưới ALS 15 config là 17,94%** theo công thức `(max−min)/max`; nếu
  dùng mẫu số trung bình thì ra 18,65% — hai tài liệu phải ghi **cùng một quy ước**.
- **Hình:** thanh ngang so sánh NDCG@10 toàn sweep CF vs CB
  (`results/figures/results-sweep-range.png`)

### 5.3 Cold-start item — kết luận đảo chiều (2 trang)
**Người 2**

| Mô hình | NDCG@10 (`coldstart`) | NDCG@10 (`full`) |
|---|---:|---:|
| CB-4 Hybrid | **0,0347** | 0,1054 |
| CB-1 TF-IDF | 0,0328 | 0,0937 |
| CB-2 BM25 | 0,0306 | 0,0998 |
| CB-3 Tag Genome | 0,0305 | 0,0713 |
| ALS / Item-CF / BPR / MostPop | **0,0000** | 0,1375 / 0,1290 / 0,1099 / 0,0739 |

> **0,0000 là *đúng bằng không*, không phải "rất thấp":** mọi tương tác train của phim
> held-out bị xoá, nên mô hình lọc cộng tác không có gì để học.

- `frac_unseen@10`: CB-4 = 0,2223 — có thể gợi ý phim chưa từng xem
- **Wilcoxon:** CB-4 vs mọi mô hình CF đều `p < 1e-190`
- **Hình:** `results-accuracy.png` (panel cold-start)

### 5.4 Kiểm định thống kê (1 trang)
**Người 2**
- **Bootstrap CI 95%** (`bootstrap-ci.csv`): ALS 0,1375 [0,1366, 0,1383] — không chạm
  CI của CB-4 [0,1047, 0,1062]
- **Wilcoxon signed-rank ghép cặp** (`wilcoxon-{cf,cb,coldstart}.csv`): chạy **trong từng
  bảng**, không so chéo CF↔CB (hai bên được tinh chỉnh với số config khác nhau)

### 5.5 Kết quả theo mức độ hoạt động của user (1 trang)
**Người 1**
- 3 nhóm theo số tương tác train: 5–30 / 31–100 / >100 (`coldstart-user-buckets.csv`)
- **Phát hiện:** mô hình CB **cải thiện theo độ dài lịch sử** (CB-2: 0,0793 → 0,0799 → 0,1418)
  chứ không giảm — ngược với trực giác "content-based chỉ tốt cho user mới"

---

## Chapter 6: Phân tích chống rò rỉ & tranh luận (6 trang)

**Người phụ trách:** Người 2 (lead) + Người 1

> **Chương này là điểm khác biệt của đồ án** — nơi biến "con số" thành "kết luận đáng tin".

### 6.1 Rủi ro metadata hậu kỳ (1,5 trang)
- `genome-scores.csv` không có timestamp — GroupLens tính ngoài dõi
- Tag bắt đầu từ **2005-12-24**; trung vị ngày kết thúc huấn luyện của user tập phát triển
  là **2006-09-25** → phần lớn user tập phát triển **train trước khi tag tồn tại**
- **Confound đo được:** Pearson(số tag của phim, độ phổ biến) = **0,768**
  (Spearman 0,700) → tag và popularity **không độc lập**

### 6.2 Bảng phân tầng 3 kỷ nguyên (2 trang)
`tag-leakage-strata.csv` — chia user theo thời kỳ rating:

| Mô hình | < 2006 | 2006–2012 | ≥ 2012 |
|---|---:|---:|---:|
| CB-2 BM25 | **0,0824** | **0,1263** | **0,1094** |
| CB-1 TF-IDF (tag) | 0,0707 | 0,1256 | 0,1083 |
| CB-3 Tag Genome | 0,0516 | 0,0836 | 0,0920 |
| ALS | 0,1283 | 0,1507 | 0,1430 |
| Item-CF (RS) | 0,1299 | 0,1379 | 0,1229 |
| BPR (RS) | 0,1038 | 0,1186 | 0,1137 |
| MostPop | 0,0640 | 0,1046 | 0,0708 |
| **Chỉ genres (không rò rỉ)** | 0,0026 | 0,0041 | 0,0060 |

**Kết luận:** thứ hạng **không đảo chiều** ở bất kỳ tầng nào — **CB-2 ≥ CB-1 > CB-3**
ở cả ba tầng. Riêng "chỉ genres" (không rò rỉ) đạt 0,0026–0,0060 ở mọi tầng, tức phần
lớn lợi thế của tag là do **tính chất nội dung**, không phải do rò rỉ thời gian.
CF không bị CB vượt ở tầng nào.

> **Lưu ý bắt buộc khi viết:** thứ hạng là **CB-2 ≥ CB-1 > CB-3**, không phải
> "CB-1 > CB-2 > CB-3". Chênh lệch rất nhỏ ở tầng 2006–2012 (0,1263 vs 0,1256) nhưng
> BM25 dẫn ở cả ba tầng.

### 6.3 Ảnh hưởng của chế độ trọng số hồ sơ (1 trang)
`weight-mode-buckets.csv` + `weight-mode-paired-tests.csv`:
- `centered` **tệ hơn hẳn** `binary_pos` ở cả 3 nhóm (chênh −0,023 / −0,020 / −0,031;
  p < 1e-44)
- `rating` gần như tương đương `binary_pos` (p = 0,04 / 3,5e-06 / 1,0e-06)
- **Diễn giải:** tín hiệu quan trọng là **có tương tác hay không**, không phải **mức độ hài lòng**

### 6.4 Metric ngoài độ chính xác (1,5 trang)
`beyond-accuracy-full.csv`:

| Mô hình | Coverage@10 | Gini@10 | Novelty@10 | ILD@10 |
|---|---:|---:|---:|---:|
| ALS | 14,94% | 0,9732 | 3,6150 | 0,7333 |
| Item-CF (RS) | 29,92% | 0,9832 | 3,1504 | 0,6640 |
| CB-3 Tag Genome | 67,20% | 0,9407 | 5,5252 | 0,5208 |
| CB-2 BM25 | 49,48% | 0,9855 | 3,2252 | 0,6314 |
| CB-4 Hybrid | 48,89% | 0,9802 | 3,5064 | 0,5842 |
| MostPop | 2,06% | 0,9985 | 1,9550 | 0,5784 |

**Phát hiện trung tâm:** **coverage đảo ngược với độ chính xác** — ALS chính xác nhất
nhưng chỉ chạm 14,9% catalog; CB-3 kém nhất trong CB nhưng chạm 67,2%. Gini đi cùng
hướng ngược: CF dồn suất vào ít phim hơn.

**Hình:** `results-beyond-accuracy.png`, `results-accuracy-vs-novelty.png`

### 6.5 Khi nào phương pháp nào thắng (1 trang)
| Điều kiện | Thắng | Bằng chứng |
|---|---|---|
| Giao thức thường, lịch sử dày | **CF** (ALS 0,1375) | Chương 5.1 |
| Phim mới, chưa có tương tác | **CB** (0,0347 vs 0,0000) | Chương 5.3 |
| User lịch sử ngắn (5–30) | CB vẫn cạnh tranh được | Chương 5.5 |
| Muốn độ phủ rộng, tránh dồn suất | **CB** (coverage 49–67%) | Chương 6.4 |
| Muốn giải thích được | **CB** (tương đồng nội dung) | Chương 4.3 |

---

## Chapter 7: Discussion (4 trang)

**Người phụ trách:** Người 1 + Người 2

### 7.1 Tổng hợp phát hiện
1. CF thắng ở giao thức thường — và kết luận này **đã cân bằng** về độ bao phủ sweep
2. CB thắng áp đảo ở cold-start item — đây là điều kiện cụ thể, không phải "CF kém"
3. Coverage đảo ngược với độ chính xác
4. Tín hiệu content không bị rò rỉ thời gian (bảng phân tầng kỷ nguyên)

### 7.2 Đánh đổi CF ↔ Content-based
| | CF | Content-based |
|---|---|---|
| **Thế mạnh** | Khám phá mẫu hợp tác ẩn; không cần đặc trưng phim | Không cần tương tác; giải thích được; phủ rộng; sống được với phim mới |
| **Điểm yếu** | Cold-start item = 0; dồn suất (coverage 14,9%); không giải thích | Giới hạn bởi chất lượng đặc trưng; over-specialization; thiếu serendipity |
| **Vị trí trong đồ án** | 0,1375 | 0,1054 → 0,0347 (cold-start) |

### 7.3 Bài học về thiết kế protocol
- **Split per-user vs global:** 4.587 → 158.774 user đánh giá (×34,6)
- **Bằng chứng protocol trên từng dòng:** `split_id` + `catalog_id` — biến "cùng giao thức"
  từ lời tuyên bố thành thứ kiểm chứng được
- **Tái lập được > nhanh:** BPR `num_threads=1` đánh đổi 40s để số tuyệt đối ổn định
- **Bẫy popularity prior:** tốt ở giao thức thường, chết ở cold-start — cần đánh giá
  trên **cả hai** giao thức trước khi chọn

### 7.4 Giới hạn
- **Metadata hậu kỳ** (đã đo ở Chương 6, nhưng số tuyệt đối vẫn lạc quan hơn production)
- **Cold-start là mô phỏng**: 10% phim giữ lại theo 5 nhóm năm phát hành, mọi user vẫn
  có lịch sử train — không phải phim mới thật
- **Chỉ offline:** không A/B test, không đo tác động thực tế
- **Giao thức LOO chưa chạy:** lựa chọn full ranking chỉ dựa trên trích dẫn, chưa có
  bằng chứng Kendall τ đo trong repo
- **Chỉ 9 mô hình, không deep learning**

### 7.5 Hàm ý thực tiễn
- Hệ thống thật nên **kết hợp**: CF cho lịch sử dày, CB cho phim mới
- Nếu chỉ chọn một: CF cho catalog ổn định, CB nếu luồng phim mới đến liên tục
- **Luôn đánh giá trên ≥ 2 giao thức** (thường + cold-start) trước khi chọn mô hình

---

## Chapter 8: Conclusion & Future Work (2 trang)

**Người phụ trách:** Người Phụ B

### 8.1 Đóng góp
1. Hợp nhất 9 mô hình trên một giao thức, có bằng chứng protocol từng dòng
2. Cân bằng sweep 51 CF / 93 CB → kết luận "CF thắng" có cơ sở
3. Kết luận cold-start đảo chiều, có Wilcoxon
4. Đo trực tiếp rủi ro rò rỉ metadata hậu kỳ bằng phân tầng 3 kỷ nguyên
5. Bằng chứng sweep commit trong repo (`ablation-*.csv`) — kiểm chứng được

### 8.2 Hướng phát triển
| Hướng | Chi phí | Ghi chú |
|---|---|---|
| **Chạy sampled LOO → Kendall τ** | ~20 phút | Biến lựa chọn full ranking từ trích dẫn thành **kết quả thực nghiệm** — ưu tiên cao nhất |
| Hybrid CF+CB ở tầng fusion | Trung bình | Nối 2 kết quả trái chiều của đồ án |
| Biến thể giới hạn số tag/phim của CB-1 | 1 run | Cắt tương quan với popularity (Pearson 0,768) |
| Bảng Wilcoxon so chéo CF↔CB | Thấp | Cần bảng riêng — câu hỏi khác bảng trong bảng |
| Đo trực tiếp năng lực xếp hạng ngoài top-10 | Trung bình | NDCG@50 đã có, nhưng phân tích sâu hơn |
| Deep learning (NCF, BERT4Rec) | Cao | Ngoài phạm vi đồ án |

---

## References (1,5 trang)

**Người phụ trách:** Người Phụ A — **IEEE hoặc ACM style**

**Bắt buộc có:**
1. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *ICDM*, 263–272.
2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37.
3. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. *UAI*.
4. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: history and context. *ACM TiiS*, 5(4), 1–19.
5. Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM TOIS*, 20(4), 422–446.
6. Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *FnTIR*, 3(4), 333–389.
7. Krichene, W., & Rendle, S. (2020). On sampled metrics for item recommendation. *KDD*.
8. GroupLens. *MovieLens 25M* + Tag Genome — https://grouplens.org/datasets/movielens/25m/

**Bổ sung:** Ricci et al. (2015) *Recommender Systems Handbook*; Shani & Gunawardana (2011) *Evaluating Recommendation Systems*; Herlocker et al. (2004) metrics đánh giá; McFee et al. (2020) *metrics*; Chen et al. (2016) *LightGBM* (nếu nhắc).

---

## Appendix (2 trang)

**Người phụ trách:** Người Phụ C

- **A.** Bảng sweep đầy đủ — `results/final/ablation-{cf,cb1,cb2,cb3,cb4}.csv` (144 dòng)
- **B.** Bảng bootstrap CI và Wilcoxon đầy đủ
- **C.** Hình EDA bổ sung (long-tail, genome)
- **D.** Quy trình tái lập: `docs/codebase-summary.md` (môi trường, lệnh, cấu trúc repo)
- **E.** Bảng thống kê dataset đầy đủ — `dataset-summary.csv`

---

## Phân công chi tiết

| Người | Chương | Trang | Nội dung chính |
|---|---|---:|---|
| **Người 1** | 2.2, 3, 4.1, 4.2, 4.4, 5.2, 5.5, 6.3, 7 | ~19 | CF theory, dataset + EDA, ALS/Item-CF/BPR, cân bằng sweep, phân tầng theo user |
| **Người 2** | 2.3–2.4, 4.3, 4.5, 5.1, 5.3, 5.4, 6, 7 | ~18 | CB theory, metric, late fusion, bảng chính, cold-start, chống rò rỉ (lead Chương 6) |
| **Người Phụ A** | 1, 2.1, References | ~5 | Introduction, tổng quan RecSys, references |
| **Người Phụ B** | 8 | ~2 | Conclusion & future work |
| **Người Phụ C** | Abstract, Appendix | ~3 | Abstract, phụ lục |

**Tổng:** 40–45 trang

---

## Timeline

| Tuần | Người 1 | Người 2 | Deliverable |
|---|---|---|---|
| 1 | Chương 3 (dataset + EDA) | Chương 2.3–2.4 (CB theory + metric) | Xong Chương 2–3 |
| 2 | Chương 4.1, 4.2, 4.4 | Chương 4.3, 4.5 | Xong Chương 4 |
| 3 | Chương 5.2, 5.5 | Chương 5.1, 5.3, 5.4 | Xong Chương 5 + bảng |
| 4 | Chương 6.3, 7 | Chương 6.1, 6.2, 6.4, 6.5 | Xong Chương 6–7 |
| 5 | Người Phụ A/B/C viết phần được giao | Review chéo toàn bộ | Bản nháp đầy đủ |
| 6 | Sửa theo góp ý + đối chiếu số liệu với `results/final/` | Sửa theo góp ý | Bản cuối |

---

## Checklist trước khi nộp

- [ ] Mọi số liệu trong báo cáo **đối chiếu được** với `results/final/*.csv` — không chép tay
- [ ] Thứ hạng CB viết là **CB-2 ≥ CB-1 > CB-3** (không phải "CB-1 > CB-2")
- [ ] Số config sweep viết là **CF 51 / CB 93**
- [ ] Nêu rõ metadata hậu kỳ (Chương 6.1)
- [ ] Nêu rõ `pop_prior` giết cold-start (Chương 4.3)
- [ ] Nêu rõ BPR `num_threads=1` và lý do (Chương 2.2.5)
- [ ] Phân biệt "0,0000 = đúng bằng không" với "thấp" ở bảng cold-start (Chương 5.3)
- [ ] Mọi bảng trong báo cáo mang `split_id` + `catalog_id` khi trích dẫn từ repo
- [ ] Chương 6 nêu rõ **giới hạn** của chính kết luận chống rò rỉ
- [ ] Ghi rõ giao thức LOO **chưa chạy** (Chương 7.4)
