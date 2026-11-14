---
title: "So sánh xử lý dữ liệu giữa RecommendationSystem và recommendation-cb"
description: "Mô tả nguồn dữ liệu thô, pipeline tiền xử lý, cách chia train/test, dựng ma trận và các vấn đề leakage của hai dự án"
status: current
last_updated: 2026-11-14
tags: [data, pipeline, recsys, evaluation, movielens]
---

# So sánh xử lý dữ liệu: RecommendationSystem vs recommendation-cb

Tài liệu này mô tả **hiện trạng** (chưa sửa) của cách hai dự án lấy dữ liệu, tiền xử lý,
chia tập và dựng ma trận. Mục tiêu là làm rõ chỗ nào khác nhau, chỗ nào là lỗi, và chỗ nào
là quyết định có chủ đích — làm cơ sở cho kế hoạch hợp nhất tại
[`plans/261113-2106-unify-recsys-and-recommendation-cb/`](../plans/261113-2106-unify-recsys-and-recommendation-cb/plan.md).

## 0. Tóm tắt một dòng

| | RecommendationSystem (RS) | recommendation-cb (CB) |
|---|---|---|
| Nguyên tắc | Giữ **toàn bộ** catalog, split **global** theo thời gian | Lọc catalog theo Tag Genome, split **per-user** theo thời gian |
| Số user được đánh giá | **4.587** | **158.774** |
| Đặc trưng nội dung | Không có | 3 nguồn (genres / tag / genome) |
| Leakage thời gian | Chưa được phân tích | Đã chẩn đoán và ghi nhận tường minh |

Cả hai dùng **cùng một bộ MovieLens 25M** và **cùng ngưỡng relevant = rating ≥ 4.0**.

---

## 1. Nguồn dữ liệu thô

### 1.1. MovieLens 25M — nội dung

| File | Dòng (không header) | Nội dung |
|---|---:|---|
| `ratings.csv` | 25.000.095 | `userId,movieId,rating,timestamp` |
| `movies.csv` | 62.423 | `movieId,title,genres` |
| `tags.csv` | 1.093.360 | `userId,movieId,tag,timestamp` — metadata do người dùng tự gán |
| `genome-scores.csv` | 15.584.448 | `movieId,tagId,relevance` — Tag Genome |
| `genome-tags.csv` | 1.128 | `tagId,tag` — tên tag của genome |
| `links.csv` | 62.423 | `movieId,imdbId,tmdbId` |

Khoảng thời gian: **1995-01-09 → 2019-11-21**. 162.541 user, 62.423 phim.

### 1.2. Vị trí lưu trữ ở hai dự án

| | RS | CB |
|---|---|---|
| Đường dẫn raw | `data/raw/ml-25m/` — **đã có đủ 7 file (~1,2 GB)** | `ml-25m/` — **không tồn tại trên đĩa** (gitignore) |
| Nguồn | Đã tải về | Chỉ có `README-DATASET.md` mô tả dataset |
| Kích thước thực tế | Kiểm tra số dòng khớp 100% với GroupLens | — |

> **Kết luận cho kế hoạch:** RS đã có sẵn dữ liệu thô đầy đủ và đúng chuẩn. **Không cần
> tải lại raw data** — việc thay đổi nằm hoàn toàn ở pipeline xử lý phía sau.

### 1.3. Ràng buộc pháp lý

GroupLens **cấm phân phối lại** MovieLens. Cả hai repo đều gitignore thư mục raw:
RS (`.gitignore` → `data`), CB (`.gitignore` → `ml-25m/`).

### 1.4. Mốc thời gian quan trọng

```
Tag đầu tiên của MovieLens:  2005-12-24 13:00:10
Rating đầu tiên:              1995-01-09 11:46:49
```

Mốc này là gốc rễ của toàn bộ vấn đề leakage ở mục 5.

---

## 2. Pipeline tiền xử lý

### 2.1. RS — `src/data_loader.py`

Một lớp `ML25MLoader` (đọc CSV, ép dtype `int32`/`float16`) và một lớp `DataPreprocessor`
(xử lý). Điểm vào: `src/data_loader.py:13` và `src/data_loader.py:62`.

Luồng thực tế trong notebook `notebooks/00-data-preprocessing.ipynb`:

```
ratings.csv
  └─ DataPreprocessor.temporal_split(train_ratio=0.7)   → global temporal 70/30
      └─ DataPreprocessor.filter_cold_start(train, test) → test chỉ còn user/item ấm
          └─ DataPreprocessor.to_implicit(threshold=4.0) → phản hồi ẩn
              └─ build_user_item_matrix(...)              → CSR + user_map + movie_map
```

Đầu ra lưu vào `data/processed/`:

| File | Kích thước | Nội dung |
|---|---:|---|
| `train_ml25m.csv` | 451 MB | 17.500.066 rating |
| `test_ml25m.csv` | 15 MB | 548.359 rating (đã lọc cold-start) |
| `train_implicit_user_item.npz` | 17 MB | Ma trận CSR phản hồi ẩn |

### 2.2. CB — `src/data/`

Ba script tiền xử lý, mỗi script một bước, có cache parquet:

```
ml-25m/
  └─ src/data/load_movielens_filtered.py
       → data/processed/ratings_genome.parquet   (user_idx, item_idx, rating, ts)
       → data/processed/item_meta.parquet        (item_idx, movieId, title, year, genres)
       → data/processed/user_map.parquet         (user_idx, userId)
  └─ src/data/build_item_content_features.py
       → data/processed/item_genres.npz          (CSR [I, 19])
       → data/processed/item_tag_docs.parquet    (item_idx, tag_text, n_tags, ts_max)
       → data/processed/item_genome.npy          (dense [13816, 1128] float32)
       → data/processed/genome_tag_names.parquet
  └─ src/data/split_temporal_per_user.py
       → data/processed/train.parquet
       → data/processed/test.parquet
       → data/processed/dev_users.npy            (20.000 user mẫu)
```

Đường dẫn tập trung trong `src/paths.py` — không hard-code path ở đâu khác.

---

## 3. Xử lý khác biệt: lọc catalog và remap chỉ số

Đây là khác biệt lớn nhất về dữ liệu giữa hai bên.

### 3.1. RS — giữ toàn bộ catalog

Không lọc phim nào. `userId`/`movieId` gốc được giữ nguyên, chỉ map sang chỉ số liên tục
khi dựng ma trận (`src/data_loader.py:225`).

Hệ quả: ma trận có **62.423 cột phim**, trong đó phần lớn không có content feature nào.

### 3.2. CB — lọc theo Tag Genome

`src/data/load_movielens_filtered.py:26` đọc `genome-scores.csv` lấy danh sách `movieId`
có genome (13.816 phim), giữ lại rating của các phim đó, rồi remap:

- `movieId` → `item_idx` liên tục `0..13815`, thứ tự tăng dần theo `movieId`
- `userId` → `user_idx` liên tục, thứ tự tăng dần theo `userId`
- `timestamp` → `ts`

Số liệu CB đã ghi nhận (`results/final/dataset-summary.csv`):

| Chỉ số | Giá trị |
|---|---|
| Số user | 162.540 |
| Số item | 13.816 |
| Số rating | 24.674.113 |
| Mật độ ma trận | 1,0988% |
| Rating ≥ 4 | 50,1% |
| Rating trung bình | 3,54 |
| Rating/user (trung vị) | 71 |
| Rating/item (trung vị) | 288 |
| Rating thuộc 20% phim phổ biến nhất | 84,4% |

### 3.3. Vì sao lọc là quyết định đúng

- **Giữ ~98,7% rating** → mất gần như không có thông tin.
- **Cùng tập ứng viên cho mọi mô hình.** Nếu giữ 62.541 phim, các mô hình content-based
  sẽ có thêm hàng nghìn phim không có tag/genome, và kết luận "content-based kém CF" có
  thể chỉ là artifact của catalog chứ không phải chất lượng mô hình.

---

## 4. Chia train/test

### 4.1. RS — global temporal 70/30

`DataPreprocessor.temporal_split` (`src/data_loader.py:66`) sắp xếp **toàn bộ** rating
theo timestamp rồi cắt tại `int(len * 0.7)`.

Mốc cắt thực tế: **2015-01-22 16:43:05**.

Đo trên raw data:

| Mốc cắt | Rating train | User train | Rating test | User test |
|---|---:|---:|---:|---:|
| 2008-01-01 | 12.510.130 | 89.924 | 12.489.965 | 77.482 |
| 2010-01-01 | 14.338.258 | 100.781 | 10.661.837 | 67.091 |
| 2012-01-01 | 15.807.192 | 110.246 | 9.192.903 | 57.536 |
| **2015-01-22 (RS)** | 17.500.066 | 122.019 | 7.501.522 | 45.145 |

Sau `filter_cold_start` loại user/item lạnh, còn lại:

| | RS (đã lọc) |
|---|---:|
| Rating test | 548.359 |
| **User được đánh giá** | **4.587** |
| Item trong test | 18.624 (toàn bộ đều có trong train) |
| Khoảng thời gian test | 2015-01-22 → 2019-11-21 |

**Vấn đề:** 122.019 user có trong train, nhưng chỉ 4.587 user được đánh giá — **96% user
bị bỏ khỏi vòng đánh giá**. Nguyên nhân: phần lớn user ngừng hoạt động trước mốc 2015.

Hệ quả trực tiếp: mẫu đánh giá nhỏ → sai số lớn → mọi kết luận thống kê trên bảng RS
hiện tại đều yếu hơn hẳn so với thực tế.

### 4.2. CB — temporal per-user 80/20

`src/data/split_temporal_per_user.py:29` — với **mỗi user**:

- Sắp xếp theo `(user_idx, ts, item_idx)` (sắp ổn định khi nhiều rating trùng timestamp)
- `n_test = max(1, floor(counts * 0.2))` → 20% **mới nhất** vào test, 80% cũ vào train
- Tách `TrainData` (mô hình được thấy) và `Dataset` (chỉ evaluator thấy `test_rel_csr`)
  → **hàng rào chống leakage ở mức cấu trúc**

Điều kiện user hợp lệ (`select_valid_users`):

```
train >= 5 rating  VÀ  có ít nhất 1 rating >= 4 trong test
```

Kết quả đo trên raw data (tương đương CB):

| | RS (global 70/30) | CB (per-user 80/20) |
|---|---:|---:|
| User trong train | 122.019 | 162.540 |
| **User được đánh giá** | **4.587** | **158.777** |
| Rating test | 548.359 | 4.937.562 |
| Rating test relevant (≥ 4) | — | 2.303.075 |

> Con số CB ghi trong `results/final/full-summary.csv` là **158.774** (nhỏ hơn 3 user vì
> CB lọc catalog 13.816 phim có genome, còn phép tính tay ở trên dùng toàn bộ 62.423 phim).

### 4.3. So sánh tính chất

| Tiêu chí | Global 70/30 (RS) | Per-user 80/20 (CB) |
|---|---|---|
| Mốc thời gian | Một mốc chung cho **mọi** user | **Riêng cho từng user** |
| Train của user u luôn sớm hơn test của user u? | Không đảm bảo (test user u có thể sớm hơn train user v) | ✅ **Có, tuyệt đối** |
| Mọi user có dữ liệu test? | ❌ 96% user không có | ✅ 100% |
| Phụ thuộc mốc cắt toàn cục | Có | Không |
| Đánh giá được toàn bộ dân số user | Không | ✅ |

Per-user split **chặt hơn** global split ở tầng lịch sử người dùng: với mọi user u, mọi
interaction train đều có timestamp nhỏ hơn mọi interaction test của chính u đó.

### 4.4. Split bổ sung của CB

| Loại split | Mô tả | Mục đích |
|---|---|---|
| `dev` (20.000 user) | Mẫu ngẫu nhiên từ valid users, seed 42 | **Chọn tham số** — không dùng để báo cáo số cuối |
| `full` (158.774 user) | Toàn bộ valid users | Bảng kết quả chính |
| `coldstart` item | Giữ lại 10% phim (phân tầng theo năm phát hành, 5 nhóm phân vị), xoá mọi interaction train của chúng | Thí nghiệm then chốt: CF cho điểm 0, CB không đổi |

---

## 5. Rò rỉ dữ liệu (leakage) — phân tầng 3 mức

| Tầng | Nội dung | RS | CB |
|---|---|---|---|
| **L1 — Nhãn** | Model không được thấy rating của user trong test | Sạch (nhưng chỉ bằng kỷ luật) | Sạch ở mức **cấu trúc** (`TrainData` / `Dataset` tách riêng) |
| **L2 — Lịch sử user** | Train của user u luôn cũ hơn test của user u | Sạch | Sạch |
| **L3 — Đặc trưng item** | Tag / genome / popularity tính trên **toàn corpus** | ❌ **Chưa phân tích** | ❌ **Đã chẩn đoán, đã ghi nhận** |

### 5.1. Vì sao L3 không cắt được trên split per-user

Mốc dự đoán khác nhau theo từng user, nhưng đặc trưng phía item là **một ma trận dùng chung**.
Đo trên raw data RS:

```
Median mốc kết thúc train (per-user):  2006-09-26
User có train_end < 2006-01-01:          77.310  (47,6%)
User có train_end < 2012-01-01:         107.630  (66,2%)
```

Vì tag chỉ bắt đầu **2005-12-24**, gần một nửa số user kết thúc lịch sử train trước kỷ
nguyên tag → toàn bộ tag mà mô hình content-based dùng đều được tạo *sau* thời điểm dự đoán.

Thử cắt tag theo một mốc cố định và đo phần user còn đánh giá được:

| Mốc cắt đặc trưng | User hợp lệ còn lại | Tỉ lệ |
|---|---:|---:|
| (không cắt — CB hiện tại) | 158.777 | 100% |
| 2008-01-01 | 73.664 | 46% |
| 2012-01-01 | 53.965 | 34% |
| 2015-01-22 | 42.124 | 27% |

→ Cắt cứng làm mất quá nửa tập đánh giá. Đây là lý do CB chọn **đo thứ hạng** thay vì đo
điểm tuyệt đối.

### 5.2. Với global split của RS, leak L3 còn nặng hơn

| Mốc cắt | Tag trước mốc | Tag sau mốc (rò rỉ vào kỷ nguyên test) |
|---|---:|---:|
| 2015-01-22 (RS) | 399.567 (36,5%) | **693.793 (63,5%)** |

Và trong catalog 13.816 phim có genome: **11.207 phim (81,1%)** có tag trước mốc,
2.609 phim chỉ có tag sau mốc.

### 5.3. Bằng chứng leak L3 **không** làm méo thứ hạng

CB đã kiểm chứng bằng `src/eval/analyse_tag_leakage.py` (dùng lại metric per-user đã lưu,
không cần chạy lại mô hình). Nếu leak thổi phồng CB-1, lợi thế của nó phải **lớn nhất** ở
tầng user trước kỷ nguyên tag (100% tag đến từ tương lai) và **nhỏ nhất** ở tầng tag đã dày.

NDCG@10 theo tầng mốc kết thúc train:

| Mô hình | < 2006 (n=9.532) | 2006–2012 (n=3.705) | ≥ 2012 (n=6.763) |
|---|---|---|---|
| CB-1 TF-IDF (tag) | 0,0686 | 0,1261 | 0,1047 |
| CB-3 Genome | 0,0507 | 0,0831 | 0,0911 |
| MostPop | 0,0627 | 0,1022 | 0,0695 |
| Chỉ genres (không leak) | 0,0023 | 0,0037 | 0,0055 |
| ItemKNN-CF | 0,1305 | 0,1402 | 0,1228 |
| ALS | 0,1289 | 0,1529 | 0,1424 |

Tỷ lệ dẫn trước của CB-1 trong cùng tầng:

| | < 2006 | 2006–2012 | ≥ 2012 |
|---|---:|---:|---:|
| CB-1 / CB-3 | 1,35× | 1,52× | **1,15×** |
| CB-1 / MostPop | **1,09×** | 1,23× | **1,51×** |
| CB-1 / ItemKNN-CF | **0,53×** | 0,90× | **0,85×** |
| CB-1 / ALS | **0,53×** | 0,82× | **0,74×** |

**Mọi dòng đều đi ngược dự đoán của giả thuyết leakage.** CB-1 yếu nhất so với MostPop
đúng ở tầng mà tag hoàn toàn đến từ tương lai (1,09×), và mạnh nhất khi tag đã cùng thời
với user (1,51×). Khoảng cách với hai baseline CF **thu hẹp** khi tag trở nên cùng thời.

→ Kết luận: **số tuyệt đối** lạc quan hơn môi trường production, nhưng **so sánh giữa các
mô hình vẫn công bằng** vì điều kiện như nhau.

### 5.4. Confound thật sự tìm được: tag volume ↔ popularity

| Chỉ số | Giá trị |
|---|---:|
| Pearson(số tag của phim, popularity train) | **0,768** |
| Spearman | **0,700** |

Phim được tag nhiều là phim nhiều người quan tâm. TF-IDF vì thế mã hoá gián tiếp độ phổ
biến: vector phim nhiều tag dày hơn nên khớp với nhiều profile hơn, và L2-norm theo hàng
**không** khử được điều này (nó chuẩn hoá độ dài, không chuẩn hoá số chiều khác 0).

→ Báo cáo CB phải luôn kèm cột "có/không pop prior".

### 5.5. Hạn chế không vượt qua được

**Tag Genome không cắt được theo thời gian.** `genome-scores.csv` không có cột timestamp —
GroupLens tính hậu kỳ từ toàn bộ tag. Vì vậy CB-3 (Tag Genome) **luôn** ở tầng leak L3,
phải ghi rõ trong báo cáo. Đây là hạn chế của dữ liệu, không phải lựa chọn thiết kế.

---

## 6. Dựng ma trận và đặc trưng

### 6.1. RS — `build_user_item_matrix`

`src/data_loader.py:199` hỗ trợ 3 chế độ giá trị (loại trừ nhau):

| Chế độ | Giá trị ô | Dùng cho |
|---|---|---|
| `value_column="rating"` | Rating thật (1,0–5,0) | Popularity, User-CF, Item-CF (adjusted cosine cần mean-centering) |
| `confidence_alpha=1.0` | `1 + α · rating` | Implicit ALS |
| cả hai `None` | Binary 1,0 | BPR, mô hình chỉ cần biết "đã xem" |

**Vấn đề ngữ nghĩa của ALS:** `implicit` coi giá trị trong ma trận **chính là** `c_ui`, rồi
dùng `(c−1)` để tích luỹ `YᵀC_uY`. Vì vậy:

- Phải lưu `1 + α`, không phải `α`
- **Chỉ được lưu tương tác relevant.** Nếu lưu cả rating thấp với `c = 1` thì số hạng
  `(c−1)` bằng 0 nhưng `b += 1·factor` vẫn cộng vào — mô hình học "user thích phim này
  với độ tin cậy 1" cho đúng những phim user đã chấm thấp.

RS hiện dùng `c_ui = 1 + α · rating` với `α = 1.0` mặc định (`src/recommenders/als_recommender.py:83`)
và giữ **cả** rating thấp → vi phạm cả hai điều trên. CB đã sửa: `c_ui = 1 + α` chỉ trên
tương tác relevant (`src/models/als_cf_model.py:69`).

### 6.2. CB — `TrainData` / `Dataset`

`src/data/build_interaction_matrices.py` tách cấu trúc:

```python
@dataclass
class TrainData:      # Mô hình CHỈ được thấy cái này
    train_csr: csr_matrix      # [n_users, n_items], giá trị = rating
    train_df: pd.DataFrame     # giữ cột ts cho time-decay
    n_users: int
    n_items: int

@dataclass
class Dataset:       # Chỉ evaluator được thấy test_rel_csr
    train: TrainData
    test_rel_csr: csr_matrix   # 1 nếu rating >= ngưỡng
    eval_users: np.ndarray
```

Mô hình chỉ nhận `TrainData` nên **về mặt cấu trúc không thể chạm tới nhãn test** — đây là
hàng rào chống leakage mà unit test không bắt được (gõ nhầm `train_csr` thành
`test_rel_csr` vẫn chạy trơn tru vì hai ma trận cùng shape và dtype, chỉ có metric là
nhảy vọt).

Ngoài ra `_to_csr` chặn cặp `(user, item)` trùng lặp (`src/data/build_interaction_matrices.py:56`),
vì scipy sẽ cộng dồn và làm sai cả rating lẫn nhãn relevant.

### 6.3. Đặc trưng nội dung (chỉ CB có)

`src/data/build_item_content_features.py` — ba nguồn, cùng thứ tự `item_idx`:

| Nguồn | File | Dạng | Ghi chú |
|---|---|---|---|
| Genres | `item_genres.npz` | CSR [13816, 19] | One-hot, bỏ `(no genres listed)` |
| Tag người dùng | `item_tag_docs.parquet` | (item_idx, tag_text, n_tags, ts_max) | Làm sạch: lowercase, bỏ ký tự đặc biện, bỏ token < 2 ký tự |
| Tag Genome | `item_genome.npy` | dense [13816, 1128] float32 | Điểm relevance; **không có timestamp** |

Ghi chú thiết kế đáng chú ý: CB cố tình lưu cột `ts_max` trong `item_tag_docs.parquet` để
phase sau kiểm tra leakage theo thời gian.

---

## 7. Ngưỡng relevant và cách đo

| | RS | CB |
|---|---|---|
| Ngưỡng relevant | rating ≥ 4,0 | rating ≥ 4,0 (hằng số `paths.RELEVANCE_THRESHOLD`) |
| Mask lịch sử | `seen` từ train, truyền vào `model.recommend(..., seen=...)` | Mask toàn bộ lịch sử train trong evaluator theo lô |
| Giao thức chấm điểm | Gọi `model.recommend()` từng user (per-user API) | **Full ranking theo lô** — `score_users(batch)` |
| K | 5, 10, 20 | 5, 10, 20, 50 |
| Metric ngoài accuracy | Không có | Coverage@10, Gini@10, novelty, cold-start, Wilcoxon, bootstrap |

### 7.1. Bốn lỗi đánh giá của RS

Đã xác định, tất cả theo **cùng chiều làm số liệu đẹp lên**:

| # | Vị trí | Vấn đề |
|---|---|---|
| B1 | `src/evaluation/recommender_evaluator.py:70` | `precision_at_k` = `hits / len(top_k)` — phải là `hits / k`. Model trả 2 item khi K=10 → P@10 tối đa 0,5 thay vì 0,1 |
| B2 | `src/evaluation/evaluation_pipeline.py:216` | `except Exception → recommended = []` → lỗi model biến thành score 0, vẫn ra bảng "hợp lệ" |
| B3 | `src/evaluation/statistical_analysis.py:53` | Ghép cặp thống kê theo thứ tự dict, không theo `userId` → p-value vô nghĩa nếu 2 model đánh giá trên tập user khác nhau |
| B4 | `src/evaluation/evaluation_validation.py:26` | `expected_ratio=0.7` mặc định — protocol mới là per-user 80/20 |

Thêm: `RecommenderEvaluator.evaluate_user` (`src/evaluation/recommender_evaluator.py:199`)
**thiếu `HitRate@{k}`**, trong khi `evaluation_pipeline` ghi đủ P/R/NDCG/HR — hai nơi lệch nhau.

---

## 8. Bảng đối chiếu nhanh

| Khía cạnh | RS | CB | Hợp nhất sẽ theo |
|---|---|---|---|
| Raw data | `data/raw/ml-25m/` ✅ có | `ml-25m/` ❌ không có | **RS** (đã có sẵn) |
| Lọc catalog | Không lọc (62.423 phim) | Lọc genome (13.816 phim) | **CB** |
| Định dạng cache | CSV (451 MB) | Parquet + npz + npy | **CB** |
| Split | Global 70/30 | Per-user 80/20 | **CB** |
| User đánh giá | 4.587 | 158.774 | **CB** |
| Tập user đánh giá | Tất cả user trong test | `dev` 20k / `full` 158.774 | **CB** |
| Chống leakage | Kỷ luật | Cấu trúc (`TrainData`/`Dataset`) | **CB** |
| Đặc trưng nội dung | Không | 3 nguồn | **CB** |
| Chấm điểm | Per-user API | Full ranking theo lô | **CB** |
| Metric ngoài accuracy | Không | Coverage, Gini, cold-start, Wilcoxon | **CB** |
| Công thức confidence ALS | `1 + α·rating`, giữ cả rating thấp | `1 + α`, chỉ relevant | **CB** |
| Số lần eval | 1 lần per user (chậm) | Batch 2.000 user | **CB** |

---

## 9. Các câu hỏi còn tồn đọng

1. **Protocol kép** (bảng chính per-user + bảng phụ cắt đặc trưng tại mốc 2012-01-01) có
   cần thiết không, hay chỉ giữ bảng chính kèm caveat?
2. **CB-1 có biến thể giới hạn số tag mỗi phim** để cắt tương quan với popularity
   (Pearson 0,768) không? Chi phí 1 run, nhưng làm phình phạm vi.
3. Có cần kiểm định thống kê (Mann-Whitney) giữa các tầng thời kỳ, hay bảng mô tả là đủ?
4. Mốc cắt 2012-01-01 cho bảng phụ — có nên dùng mốc bám theo phân vị thay vì mốc cứng?

---

## 10. Nguồn tham chiếu

| Nội dung | Vị trí |
|---|---|
| Loader + preprocessor RS | `src/data_loader.py` |
| Pipeline đánh giá RS | `src/evaluation/` |
| Loader CB | `src/data/load_movielens_filtered.py` |
| Feature builder CB | `src/data/build_item_content_features.py` |
| Split per-user CB | `src/data/split_temporal_per_user.py` |
| Ma trận + hàng rào leakage CB | `src/data/build_interaction_matrices.py` |
| Cold-start item split | `src/eval/cold_start_item_split.py` |
| Chẩn đoán leakage tag | `src/eval/analyse_tag_leakage.py` |
| Báo cáo chẩn đoán leakage (CB) | `plans/reports/analysis-260925-0001-tag-leakage-diagnosis.md` |
| Thống kê dataset (CB) | `results/final/dataset-summary.csv` |
| Bảng kết quả full/dev (CB) | `results/final/full-summary.csv`, `dev-summary.csv` |
| Kế hoạch hợp nhất | `plans/261113-2106-unify-recsys-and-recommendation-cb/` |
| Giấy phép + mô tả dataset | `README-DATASET.md` (CB) |

---

## 11. Trạng thái sau hợp nhất (2026-11-14)

Tài liệu này mô tả **hiện trạng trước khi hợp nhất** — cột "RS" ở các mục 1–7 là
protocol cũ (global 70/30), **đã bị xoá hoàn toàn ở Phase 04a**. Mục này ghi trạng thái
sau hợp nhất để người đọc không hiểu nhầm mô tả cũ là trạng thái hiện hành.

### 11.1 Cột "RS" ở mục 1–7 giờ là lịch sử

| Nội dung | Cũ (đã xoá) | Nay |
|---|---|---|
| Loader | `src/data_loader.py` | `src/data/load_movielens_filtered.py` |
| Preprocessor | `src/data_loader.py` | `src/data/build_item_content_features.py` |
| Split | `src/data_loader.py` (global 70/30) | `src/data/split_temporal_per_user.py` (per-user 80/20) |
| Ma trận | `build_user_item_matrix` | `src/data/build_interaction_matrices.py` (`TrainData`/`Dataset`) |
| Evaluator | `src/evaluation/` | `src/eval/batched_scoring_evaluator.py` (full ranking theo lô) |
| Recommenders | `src/recommenders/` | `src/models/` (9 mô hình trong registry) |
| Notebooks | `notebooks/` (00–04) | — (xoá) |

Mọi dòng mô tả protocol cũ ở các mục 1–7 **giữ nguyên làm lịch sử**, vì chúng là bằng
chứng cho các đính chính ở mục 4.3, 5.2 và 7.1.

### 11.2 Bảng đối chiếu — trạng thái sau hợp nhất

| Khía cạnh | Trước (RS cũ) | Sau hợp nhất | Đã quyết định ở |
|---|---|---|---|
| Lọc catalog | Không lọc (62.423 phim) | **Lọc Tag Genome (13.816 phim)** | Mục 3.3 |
| Định dạng cache | CSV (451 MB) | **Parquet + npz + npy** | Phase 00 |
| Split | Global 70/30 | **Per-user 80/20** | Mục 4.3 |
| User đánh giá | 4.587 | **158.774** | Mục 4.3 |
| Tập user đánh giá | Tất cả user trong test | **`dev` 20k / `full` 158.774** | Phase 00 |
| Chống leakage | Kỷ luật | **Cấu trúc (`TrainData`/`Dataset`)** | Mục 6.2 |
| Đặc trưng nội dung | Không | **3 nguồn (genres / tag / genome)** | Mục 6.3 |
| Chấm điểm | Per-user API | **Full ranking theo lô 2.000 user** | Phase 00 |
| Metric ngoài accuracy | Không | **Coverage, Gini, novelty, ILD, cold-start, Wilcoxon** | Phase 03–04 |
| Confidence ALS | `1 + α·rating`, giữ cả rating thấp | **`1 + α`, chỉ relevant** | Mục 6.2 |
| Số lần eval | 1 lần per user (chậm) | **Batch 2.000 user** | Phase 00 |

### 11.3 Các câu hỏi ở mục 9 — đã trả lời

| Câu hỏi | Quyết định |
|---|---|
| 1. Protocol kép, cần bảng phụ cắt đặc trưng 2012? | **Không** — bảng phụ mất 66% tập user; dùng bảng phân tầng 3 kỷ nguyên (`results/final/tag-leakage-strata.csv`) thay thế |
| 2. Biến thể CB-1 giới hạn số tag/phim? | **Không** — ngoài phạm vi hợp nhất, để dành nghiên cứu riêng |
| 3. Mann-Whitney giữa các tầng thời kỳ? | **Không** — bảng mô tả đã đủ để lập luận |
| 4. Mốc cắt 2012-01-01 cứng hay theo phân vị? | **Theo phân vị** — 3 tầng kỷ nguyên (< 2006 / 2006–2012 / ≥ 2012) |

### 11.4 Nguồn tham chiếu sau hợp nhất

| Nội dung | Vị trí |
|---|---|
| Loader + preprocessor | `src/data/load_movielens_filtered.py`, `src/data/build_item_content_features.py` |
| Split per-user | `src/data/split_temporal_per_user.py` |
| Ma trận + hàng rào leakage | `src/data/build_interaction_matrices.py` |
| Định danh catalog / split | `src/data/catalog_identity.py` |
| Evaluator full-ranking theo lô | `src/eval/batched_scoring_evaluator.py` |
| Registry 9 mô hình | `src/eval/model_registry.py` |
| Cold-start item split | `src/eval/cold_start_item_split.py` |
| Chẩn đoán leakage tag | `src/eval/analyse_tag_leakage.py` |
| Bảng kết quả | `results/final/benchmark-{cf,cb,coldstart}.csv` |
| Tóm tắt codebase | `docs/codebase-summary.md` |

---

*Cập nhật: 2026-11-14. Số liệu đo trực tiếp trên raw data của RS bằng `venv/bin/python`;
số liệu CB trích từ `results/final/` của repo `recommendation-cb`.*
