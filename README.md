# RecSys2026

Hệ thống gợi ý trên MovieLens 25M. Tài liệu đặt trong `notes/` và `docs/`.

## Pipeline chính (đường chạy duy nhất)

Dữ liệu thô: `data/raw/ml-25m/` (không commit). Đường chạy mới đã hợp nhất content-based +
collaborative filtering theo **cùng một giao thức đánh giá** (full-ranking theo lô, mask lịch sử train).

```bash
# 1. Tiền xử lý (chạy theo thứ tự)
venv/bin/python -m src.data.load_movielens_filtered    # -> data/processed/ratings_genome.parquet
venv/bin/python -m src.data.build_item_content_features # -> đặc trưng genres / tag / genome
venv/bin/python -m src.data.split_temporal_per_user     # -> train.parquet, test.parquet, dev_users.npy

# 2. Chạy thí nghiệm
venv/bin/python -m src.eval.run_experiment_from_config configs/baseline-mostpop.yaml

# 3. Gom bảng kết quả (bảng rút gọn cho báo cáo + hình)
venv/bin/python scripts/build_final_tables.py
venv/bin/python -m src.eval.make_result_figures

# 4. Test
venv/bin/python -m pytest tests/
```

### Thứ gì nằm trong git, thứ gì không

| Đường dẫn | Trong git? | Ghi chú |
|---|---|---|
| `src/`, `tests/`, `scripts/`, `configs/`, `docs/` | ✅ Có | mã nguồn và 159 config YAML |
| `results/final/*.csv` (27 bảng) | ✅ Có | **bảng báo cáo cuối** — người clone repo xem được ngay |
| `results/figures/*.png` (9 hình) | ✅ Có | hình cho chương kết quả |
| `results/*_topk.npy`, `results/*_peruser.npz`, `results/*_metrics.csv` | ❌ Không | artifact sinh tự động, **~500 MB**; gitignore |
| `data/` (raw + cache) | ❌ Không | sinh lại bằng 3 bước tiền xử lý ở trên |
| `notes/`, `plans/` | ❌ Không | ghi chú nội bộ, không track |

Hai điều cần biết khi clone repo:

1. **`results/` chỉ có sẵn phần báo cáo.** Muốn chạy lại kiểm định thống kê
   (`beyond_accuracy_metrics`, `statistical_tests`, `user_bucket_analysis`) thì phải chạy lại
   pipeline — chúng đọc `*_peruser.npz` và `*_topk.npy` vốn không có trong git.
2. **`data/legacy/` đã bị xoá** — đó là dữ liệu của protocol cũ (global 70/30), không còn module
   nào đọc. Mô tả protocol đó nằm ở
   [`docs/data-processing-comparison-across-projects.md`](docs/data-processing-comparison-across-projects.md)
   mục 2.1, 4.1, 6.1, 7.1.

### Chạy bảng so sánh chung (9 mô hình × 3 giao thức)

```bash
# dev: chọn tham số (20.000 user) + sweep ALS (α × factors × reg × iterations)
venv/bin/python scripts/run_shared_benchmark.py --protocol dev --als-sweep

# full: bảng chính, 158.774 user — chạy MỘT lần sau khi khoá config
venv/bin/python scripts/run_shared_benchmark.py --protocol full

# cold-start item: giữ lại 10% phim (1.381 phim theo 5 nhóm năm phát hành)
venv/bin/python -m src.eval.cold_start_item_split
venv/bin/python scripts/run_shared_benchmark.py --protocol coldstart

# Chỉ số ngoài độ chính xác + kiểm định thống kê
# Ghi thẳng vào results/final/ — bản ở gốc results/ là artifact trung gian (bị gitignore)
venv/bin/python -m src.eval.beyond_accuracy_metrics --prefix full_ \
    --output results/final/beyond-accuracy-full.csv
venv/bin/python -m src.eval.beyond_accuracy_metrics --prefix coldstart_ \
    --output results/final/beyond-accuracy-coldstart.csv

# Gom bảng + Wilcoxon trong từng bảng
venv/bin/python -m src.eval.benchmark_tables --protocol full      # -> results/final/benchmark-{cf,cb}.csv
venv/bin/python -m src.eval.benchmark_tables --protocol coldstart # -> results/final/benchmark-coldstart.csv
venv/bin/python -m src.eval.analyse_tag_leakage --protocol full   # bảng phân tầng theo kỷ nguyên
```

Mỗi dòng `{exp}_metrics.csv` mang `split_id` + `catalog_id` + `seed` — đó là **bằng chứng**
các dòng trong một bảng thực sự chạy trên cùng tập ứng viên và cùng cách chia, không phải
lời tuyên bố. `benchmark_tables` cũng **raise** nếu cột `model` trong kết quả không khớp
mô hình mà bảng khai báo (config bị sửa sau khi chạy).

`scripts/build_final_tables.py` **không ghi đè** `results/final/benchmark-*.csv` — đó là
sản phẩm của phase 03 có bằng chứng protocol. Script này chỉ rút gọn thành
`full-summary.csv` / `coldstart-item.csv` / `dev-summary.csv` (vẫn giữ `split_id`,
`catalog_id`, `model`) và **bỏ qua phần thiếu dữ liệu thay vì crash**.

Giao thức: catalog 13.816 phim có Tag Genome · split temporal per-user 80/20 · `relevant = rating ≥ 4.0`
· `users: dev` (20k, chọn tham số) hoặc `full` (158.774, báo cáo).

Chi tiết đối chiếu với `recommendation-cb`: [`docs/data-processing-comparison-across-projects.md`](docs/data-processing-comparison-across-projects.md).

## Các mô hình trong `src/eval/model_registry.py`

| Key | Mô hình | Nguồn | Ghi chú |
|---|---|---|---|
| `random` | Random | CB | sanity check |
| `mostpop` | MostPop | CB | baseline tối thiểu |
| `cb1-tfidf` | TF-IDF tag | CB | content-based |
| `cb2-bm25` | BM25 tag | CB | content-based |
| `cb3-genome` | Tag Genome | CB | content-based (metadata hậu kỳ) |
| `cb3-ridge` | Genome + Ridge | CB | content-based |
| `cb4-hybrid` | Late fusion | CB | hybrid |
| `cf-itemknn` | ItemKNN-CF | CB | trọng số user **nhị phân relevant** |
| `cf-als` | ALS | CB | confidence `1 + α` |
| `rs-itemcf` | Item-CF | RS | cùng thuật toán với `cf-itemknn`, trọng số user **rating thật** |
| `rs-bpr` | BPR-MF | RS | ma trận nhị phân relevant, `num_threads=1` để tái lập được |

Sự khác biệt giữa `rs-itemcf` và `cf-itemknn` phải được nêu khi đọc bảng kết quả — xem
[`docs/project-changelog.md`](docs/project-changelog.md) mục Phase 02.

### Kết quả `dev` (20.000 user) sau Phase 02

| Model | NDCG@10 | HR@10 | fit | score |
|---|---|---|---|---|
| ALS (f=64, α=5, reg=0,1) | 0,1383 | 0,5410 | 14,5s | 5,4s |
| ItemKNN-CF (k=200) | 0,1297 | 0,4827 | 0,37s | 4,4s |
| Item-CF (binary_pos, k=200) | 0,1297 | 0,4827 | 0,35s | 4,2s |
| BPR (f=64, i=10, lr=0,01, reg=1e-5) | 0,1106 | 0,4606 | 40,9s | 2,8s |
| MostPop | 0,0723 | 0,3174 | 0,04s | 0,2s |
| Random | 0,0011 | 0,0100 | 0,00s | 0,4s |

`rs-itemcf` (binary_pos) khớp **tuyệt đối** với `cf-itemknn` — xác nhận cùng thuật toán.
BPR chạy với `num_threads=1` để **tái lập được**: `implicit` lấy mẫu âm bằng RNG chung
giữa các luồng nên >1 luồng cho kết quả khác nhau giữa hai lần chạy.

### Bảng chính — `full` (158.774 user), Phase 03

Nguồn: `results/final/benchmark-cf.csv`, `benchmark-cb.csv`. Cả hai bảng dùng **cùng**
split, catalog và tập user (kiểm chứng bằng cột `split_id` / `catalog_id`).

**Bảng CF** (`results/final/benchmark-cf.csv`)

| Mô hình | NDCG@10 | NDCG@20 | HR@10 | Coverage@10 | Gini@10 | fit | score |
|---|---|---|---|---|---|---|---|
| ALS | 0,1375 | 0,1567 | 0,5366 | 0,1494 | 0,9732 | 14,9s | 23,9s |
| Item-CF (RS) | 0,1290 | 0,1439 | 0,4870 | 0,2992 | 0,9832 | 0,35s | 33,6s |
| BPR (RS) | 0,1099 | 0,1222 | 0,4593 | 0,1975 | 0,9559 | 39,8s | 21,7s |
| MostPop | 0,0739 | 0,0831 | 0,3246 | 0,0206 | 0,9985 | 0,05s | 20,4s |
| Random (sanity) | 0,0012 | 0,0014 | 0,0103 | 1,0000 | 0,0545 | 0,00s | 23,2s |

**Bảng CB** (`results/final/benchmark-cb.csv`)

| Mô hình | NDCG@10 | NDCG@20 | HR@10 | Coverage@10 | Gini@10 | fit | score |
|---|---|---|---|---|---|---|---|
| CB-4 Hybrid | 0,1054 | 0,1126 | 0,4329 | 0,4889 | 0,9802 | 0,83s | 140,2s |
| CB-2 BM25 | 0,0998 | 0,1070 | 0,4205 | 0,4948 | 0,9855 | 0,64s | 115,1s |
| CB-1 TF-IDF | 0,0937 | 0,1004 | 0,4020 | 0,4695 | 0,9833 | 1,97s | 105,5s |
| CB-3 Tag Genome | 0,0713 | 0,0750 | 0,3381 | 0,6720 | 0,9407 | 0,27s | 28,2s |

**Bảng cold-start item** (`results/final/benchmark-coldstart.csv`, 1.381 phim bị giữ lại,
10.992 user đánh giá được)

| Mô hình | NDCG@10 | NDCG@20 | HR@10 | Coverage@10 |
|---|---|---|---|---|
| CB-4 Hybrid | 0,0347 | 0,0469 | 0,1108 | 0,1519 |
| CB-1 TF-IDF | 0,0328 | 0,0427 | 0,1013 | 0,1487 |
| CB-2 BM25 | 0,0306 | 0,0420 | 0,0988 | 0,1327 |
| CB-3 Tag Genome | 0,0305 | 0,0384 | 0,0964 | 0,2997 |
| ALS | 0,0000 | 0,0000 | 0,0000 | 0,1068 |
| Item-CF (RS) | 0,0000 | 0,0000 | 0,0000 | 0,1153 |
| BPR (RS) | 0,0000 | 0,0000 | 0,0000 | 0,1399 |
| MostPop | 0,0000 | 0,0000 | 0,0000 | 0,0140 |
| Random (sanity) | 0,0004 | 0,0005 | 0,0017 | 0,9997 |

Ba kết luận chính của bảng chung:

1. **CF thắng CB ở giao thức thường** — ALS 0,1375 > CB-4 Hybrid 0,1054. Đây là kết luận
   **đã cân bằng**: cả hai nhóm đều đã quét **đủ** sweep (CF 51 config, CB 93 config —
   xem [Cân bằng sweep — CF vs CB](#cân-bằng-sweep--cf-vs-cb)), nên chênh lệch 0,1375 vs
   0,1054 không còn là hệ quả của việc một bên được tinh chỉnh nhiều hơn bên kia.
2. **CB thắng áp đảo ở cold-start item** — cả 4 mô hình content đạt NDCG@10 ≈ 0,03
   trong khi **mọi** mô hình CF = 0,0000 (không phải "thấp" mà là *đúng bằng không*:
   phim held-out không có tương tác train nào nên mô hình lọc cộng tác không có gì để
   học). Đây là luận điểm trung tâm của đồ án.
3. **Coverage đảo ngược với độ chính xác** — ALS chính xác nhất nhưng chỉ chạm 14,9%
   catalog; CB-3 kém nhất trong CB nhưng chạm 67,2%. Gini đi cùng hướng ngược: CF dồn
   suất vào ít phim hơn.

### Cân bằng sweep — CF vs CB

Cả hai nhóm đã được quét **đủ** trong repo này, nên so sánh không còn bị thiên lệch về
độ bao phủ sweep:

| Nhóm | Config sweep trong `configs/` | **Đã chạy ở repo này** | Chưa chạy |
|---|---|---|---|
| CF (ALS + Item-CF + BPR) | 51 | **51** | 0 |
| CB (CB-1…CB-4) | 93 | **93** | 0 |

Bằng chứng từng config nằm ở `results/final/ablation-{cf,cb1,cb2,cb3,cb4}.csv` (144 dòng:
51 CF + 93 CB, mỗi dòng mang `params` + `split_id`/`catalog_id`) — người đọc repo kiểm
chứng được "đã quét đủ" mà không phải tin lời.

Sau khi sweep đủ, **kết luận đổi chiều so với bản trước**: CB không chỉ "thua sát" mà
thua rõ ở giao thức thường — 0,1054 (CB-4 Hybrid) so với 0,1375 (ALS), chênh **23%**.

| Nhóm CB | Best config | NDCG@10 (`dev`) | NDCG@10 (`full`) |
|---|---|---|---|
| CB-1 TF-IDF | `cb4-03-timedecay-hl0365` | 0,0921 | 0,0937 |
| CB-2 BM25 | `cb2-01-k1-05-b000` | 0,0986 | 0,0998 |
| CB-3 Tag Genome | `cb3-04d-item-knn-tau020-k100` | 0,0703 | 0,0713 |
| CB-4 Hybrid | `cb4-04-late-bm25-w075` (không pop prior) | 0,1043 | 0,1054 |

Vì vậy các bảng CF và CB vẫn được giữ **riêng** (`benchmark-cf.csv`, `benchmark-cb.csv`)
và Wilcoxon chỉ chạy **trong từng bảng**: đối chiếu CF↔CB là câu hỏi khác, cần bảng riêng.

#### Lưu ý: `pop_prior` giết chết cold-start

Bản CB-4 **có** popularity prior (`λ_pop = 0,4`) đạt NDCG@10 = 0,1081 ở giao thức thường
— cao hơn bản content thuần — nhưng **sụp** ở cold-start (0,0010, tức bằng 0) vì mọi phim
held-out có popularity = 0 nên bị đẩy xuống đáy danh sách. Chỉ số `frac_unseen@10` xác
nhận: 0,0036 (có prior) vs 0,2223 (không prior). Vì vậy **bảng chính lấy bản content thuần**
(`best-cb4-hybrid-content.yaml`); bản có prior chỉ để đối chiếu.

### Độ nhạy tham số của ALS — biên độ thật

Bản trước đây ghi "ALS không nhạy, biên độ 0,40%" — con số đó chỉ đúng cho **một lát cắt**
duy nhất của lưới tham số. Toàn bộ lưới đã chạy ở `dev` cho số khác:

| Lát cắt | Số config | NDCG@10 | Biên độ |
|---|---|---|---|
| `reg` × `iterations` (f=64, α=5) | 8 | 0,1377 – 0,1383 | **0,40%** |
| **Toàn lưới** (thêm `α` ∈ {1,5,10,40} và `factors` ∈ {64,128}) | 15 | 0,1135 – 0,1383 | **18,65%** |

`α` (confidence weight) là tham số gây sụp lớn nhất: α=40 kéo ALS xuống **0,1135**,
tức **−18%** so với α=5 (0,1379). Nói cách khác, ALS **không nhạy với `reg`/`iterations`
nhưng rất nhạy với `α` và `factors`**. Cấu hình thắng vẫn là `f=64, α=5, reg=0,1`.

## Tài liệu

**Tài liệu chính (theo repo):**

- [`docs/codebase-summary.md`](docs/codebase-summary.md): cấu trúc repo, module nào làm gì, một đường chạy duy nhất, bảng 9 mô hình.
- [`docs/system-architecture.md`](docs/system-architecture.md): luồng dữ liệu, hợp đồng `BaseRecommender`, hàng rào chống leakage.
- [`docs/development-roadmap.md`](docs/development-roadmap.md): trạng thái phase, kết quả đạt được, việc còn lại.
- [`docs/data-processing-comparison-across-projects.md`](docs/data-processing-comparison-across-projects.md): so sánh xử lý dữ liệu RS vs `recommendation-cb` + trạng thái sau hợp nhất.
- [`docs/project-changelog.md`](docs/project-changelog.md): nhật ký thay đổi dự án.

**Ghi chú nội bộ (`notes/`, không track trong git):**

- `notes/person1-implementation-guide.md`: pipeline Collaborative Filtering chính với baseline và ALS trên MovieLens 25M.
- `notes/person2-content-based-guide.md`: hướng dẫn Content-based Filtering.
- `notes/advanced-cf-bpr-svd-svdpp-ml1m-guide.md`: thực nghiệm độc lập với MovieLens 1M, giải thích và hướng dẫn chạy BPR, SVD và SVD++; gồm protocol đánh giá đúng cho explicit rating và Top-N ranking.

Mỗi hướng dẫn nêu rõ dataset, môi trường, cách chạy, cách đánh giá và các giới hạt cần trình bày khi bảo vệ. Không commit các dataset MovieLens thô hoặc kết quả chưa tái lập được.

### Giới hạn cần nêu khi bảo vệ

- **Metadata hậu kỳ**: mọi đặc trưng item (genres, tag, genome) tính trên toàn corpus, không
  cắt được theo thời gian (`genome-scores.csv` không có timestamp — GroupLens tính hậu kỳ).
  Số tuyệt đối lạc quan hơn môi trường production, nhưng **so sánh giữa các mô hình công
  bằng** vì điều kiện như nhau.
- **Bảng phân tầng theo kỷ nguyên** (`results/final/tag-leakage-strata.csv`) chứng minh thứ
  hạng **không đảo chiều** theo thời kỳ user: CB-2 ≥ CB-1 > CB-3 ở cả ba tầng
  (< 2006 / 2006–2012 / ≥ 2012) — BM25 dẫn hoặc ngang TF-IDF ở mọi tầng (chênh rất nhỏ:
  0,0824 vs 0,0707 / 0,1263 vs 0,1256 / 0,1094 vs 0,1083), còn Tag Genome luôn kém nhất.
  Riêng CB-1 vẫn dẫn CB-3 ở mọi tầng (1,37x / 1,50x / 1,18x theo ba tầng), tức
  phần lớn lợi thế của tag là do tính chất nội dung chứ không phải do rò rỉ thời gian.
- **Cold-start là mô phỏng**, không phải phim mới thật: 10% phim bị giữ lại ngẫu nhiên
  theo 5 nhóm năm phát hành. Mọi user vẫn có lịch sử train.
- **BPR chạy `num_threads=1`**: tốn 40s thay vì 7s mỗi lần chạy ở `dev`, đổi lại tái lập
  được tuyệt đối. Số trong bảng là số tái lập được.

## Protocol cũ đã bị xoá

`src/evaluation/`, `src/recommenders/`, `src/data_loader.py` và `notebooks/` (protocol đánh giá
cũ: split global 70/30) **đã bị xoá hoàn toàn ở Phase 04** — không còn đường chạy thứ hai
trong repo. Mô tả chi tiết protocol cũ (kèm số dòng tham chiếu) còn nằm trong
[`docs/data-processing-comparison-across-projects.md`](docs/data-processing-comparison-across-projects.md)
mục 2.1, 4.1, 6.1, 7.1.

Bốn lỗi đã đo ở protocol cũ và **không được sửa** (đã loại bỏ khỏi đường chạy thay vì vá):

| # | Lỗi | Trạng thái ở `src/eval/` |
|---|---|---|
| B1 | `precision_at_k` = `hits / len(top_k)` | ✅ Đúng — `hits / k` (Järvelin & Kekäläinen 2002) |
| B2 | `except Exception` nuốt lỗi mô hình | ✅ Đúng — exception nổi lên, `NaN` bị chặn |
| B3 | Ghép cặp thống kê theo thứ tự dict | ✅ Đúng — `load_all` raise khi tập user lệch |
| B4 | `validate_dataset_split` mặc định 0,7 | ✅ Không còn — split kiểm bằng test riêng |

Hai hàng rào bổ sung đã thêm ở Phase 01: item gợi ý ngoài catalog → **raise**; user không có tín hiệu
train → metric = 0 **một cách có nguyên tắc** và vẫn ở lại trong tập đánh giá.

Các file `docs/` còn lại (guide CF, ALS, BPR, HR@K…) đều mang banner **"tài liệu legacy"**:
phần *giải thích thuật toán* vẫn đúng và hữu ích, nhưng phần *đường chạy* và *số liệu* là cũ.
