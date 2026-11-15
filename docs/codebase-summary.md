# Codebase Summary

Tóm tắt cấu trúc `RecommendationSystem` — mỗi module làm gì, dữ liệu chảy qua đâu,
và chạy lại kết quả bằng cách nào.

Đọc kèm: [`system-architecture.md`](system-architecture.md) (luồng dữ liệu và hợp đồng
API), [`development-roadmap.md`](development-roadmap.md) (trạng thái và việc còn lại).

---

## 1. Một đường chạy duy nhất

Repo có **một** pipeline đánh giá. Protocol cũ (split global 70/30, evaluator riêng) đã
bị xoá hoàn toàn ở Phase 04a — không còn đường chạy thứ hai để nhầm số liệu.

```bash
# 1. Tiền xử lý (theo đúng thứ tự)
venv/bin/python -m src.data.load_movielens_filtered      # -> ratings_genome.parquet
venv/bin/python -m src.data.build_item_content_features  # -> genres / tag / genome
venv/bin/python -m src.data.split_temporal_per_user      # -> train.parquet, test.parquet, dev_users.npy

# 2. Chạy thí nghiệm
venv/bin/python -m src.eval.run_experiment_from_config configs/baseline-mostpop.yaml

# 3. Gom bảng + hình
venv/bin/python scripts/build_final_tables.py
venv/bin/python -m src.eval.make_result_figures

# 4. Test
venv/bin/python -m pytest tests/
```

Bảng so sánh chung 9 mô hình × 3 giao thức: `scripts/run_shared_benchmark.py`
(`--protocol dev|full|coldstart`).

Chi tiết cách chạy giao thức `full` (thay cho `dev`), thời gian/disk cần, và bẫy thường
gặp: [`run-full-protocol-guide.md`](run-full-protocol-guide.md).

---

## 2. Cây thư mục

```
RecommendationSystem/
├── src/
│   ├── paths.py                 đường dẫn + hằng số dùng chung (SEED, ngưỡng relevant)
│   ├── data/                    tiền xử lý dữ liệu (5 module)
│   ├── models/                  9 lớp mô hình + scorer dùng chung (20 module)
│   └── eval/                    evaluator + phân tích kết quả (13 module)
├── configs/                     159 file YAML mô tả mọi thí nghiệm
├── scripts/                     runner + công cụ gom bảng (11 script)
├── tests/                       17 file test, 219+ test
├── results/
│   ├── final/                   27 bảng CSV — **sản phẩm báo cáo**, được commit
│   ├── figures/                 9 hình PNG — cho chương kết quả
│   └── *                       artifact trung gian (~500 MB) — gitignore
├── docs/                        tài liệu
├── data/raw/ml-25m/             dữ liệu thô (gitignore)
└── data/processed/              cache pipeline (gitignore)
```

---

## 3. `src/data/` — tiền xử lý

| Module | Việc | Output |
|---|---|---|
| `load_movielens_filtered.py` | Đọc 7 file raw, lọc catalog chỉ giữ phim có Tag Genome (13.816 phim, giữ 98,7% rating), đánh lại chỉ số user/item | `ratings_genome.parquet`, `item_meta.parquet`, `user_map.parquet` |
| `build_item_content_features.py` | Dựng 3 nguồn đặc trưng: `genres` (19 thể loại), `tag` (30.067 tag sạch), `genome` (1.128 chiều) | `item_genres.npz`, `item_tag_docs.parquet`, `item_genome.npy` |
| `split_temporal_per_user.py` | Chia **theo thời gian trong từng user** 80/20; loại user có train < 5 hoặc test không có rating ≥ 4 | `train.parquet`, `test.parquet`, `dev_users.npy` |
| `build_interaction_matrices.py` | Dựng CSR, định nghĩa `TrainData` / `Dataset` — **hàng rào chống leakage** | (in-memory) |
| `catalog_identity.py` | Tính `catalog_id` (hash tập movieId) và `split_id` — bằng chứng hai dòng kết quả chạy trên cùng tập ứng viên | (metadata trong metrics CSV) |

**Vì sao lọc theo Tag Genome?** Mọi mô hình phải so sánh trên *cùng* tập ứng viên.
Lọc theo điều kiện của CB (có genome) giữ 98,7% rating nên không mất thông tin quan trọng.

**Vì sao split per-user?** Split global 70/30 chỉ đánh giá được 4.587 user (72% user
ML-25M ngừng hoạt động trước 2015, không có test). Per-user đánh giá được **158.774** user.

---

## 4. `src/models/` — 9 mô hình

Registry: [`src/eval/model_registry.py`](../src/eval/model_registry.py)

| Key | Lớp | Nhóm | Ghi chú |
|---|---|---|---|
| `random` | `RandomRecommender` | baseline | sanity check — mọi metric phải ≈ 0 |
| `mostpop` | `MostPopRecommender` | baseline | baseline tối thiểu |
| `cf-als` | `AlsCfModel` | CF | confidence `1 + α`, chỉ tương tác relevant |
| `rs-itemcf` | `RsItemCfModel` | CF | Item-CF, trọng số user = **rating thật** |
| `rs-bpr` | `RsBprModel` | CF | BPR-MF, ma trận nhị phân, `num_threads=1` để tái lập được |
| `cb1-tfidf` | `TfidfContentModel` | CB | TF-IDF trên tag, hồ sơ user = trung bình có trọng số |
| `cb2-bm25` | `Bm25ContentModel` | CB | BM25 trên tag, có tham số `k1`/`b` |
| `cb3-genome` | `GenomeContentModel` | CB | Tag Genome (metadata hậu kỳ) |
| `cb4-hybrid` | `HybridLateFusionModel` | CB | late fusion BM25 + Genome |

Hai mô hình **không** nằm trong bảng chính nhưng vẫn có trong registry:
`cf-itemknn` (ItemKNN-CF, trọng số nhị phân — khớp tuyệt đối với `rs-itemcf`) và
`cb3-ridge` (Genome + Ridge — kém Random, đã loại khỏi phạm vi).

### Scorer dùng chung

| Module | Vai trò |
|---|---|
| `content_profile_scorer.py` | Hồ sơ user = trung bình có trọng số của vector item |
| `content_item_knn_scorer.py` | Hồ sơ user = trung bình trên k hàng xóm gần nhất |
| `score_fusion_utils.py` | Chuẩn hoá điểm + late fusion có trọng số |
| `user_history_weights.py` | Chế độ trọng số: `binary_pos` / `centered` / `rating` (+ time-decay) |
| `seen_item_masking.py` | Mask lịch sử train khỏi danh sách gợi ý |
| `vector_normalization.py` | Chuẩn hoá vector (z-score / min-max) |
| `feature_cache_key.py` | Khoá cache ma trận tương đồng theo (đặc trưng, tham số) |

---

## 5. `src/eval/` — đánh giá

| Module | Việc |
|---|---|
| `topk_ranking_metrics.py` | P@k, R@k, HR@k, NDCG@k theo Järvelin & Kekäläinen (2002); trả **per-user** |
| `batched_scoring_evaluator.py` | Chấm điểm theo lô 2.000 user; ghi metrics CSV + per-user npz + topk npy |
| `model_registry.py` | Ánh xạ tên model → lớp |
| `run_experiment_from_config.py` | **Đường chạy duy nhất** cho một thí nghiệm từ YAML |
| `benchmark_tables.py` | Gom bảng CF / CB / cold-start + Wilcoxon trong từng bảng |
| `beyond_accuracy_metrics.py` | Coverage, Gini, novelty, ILD, `frac_unseen` |
| `statistical_tests.py` | Bootstrap CI 95% + Wilcoxon signed-rank ghép cặp |
| `user_bucket_analysis.py` | Metric theo độ dài lịch sử train của user (cold-start **người dùng**) |
| `cold_start_item_split.py` | Tạo split giữ lại 10% phim (phân tầng 5 nhóm năm phát hành) |
| `analyse_tag_leakage.py` | Bảng phân tầng 3 kỷ nguyên user — chống rò rỉ thời gian của tag |
| `make_result_figures.py` | Sinh 5 hình kết quả |
| `collect_results_to_table.py` | Gom `*_metrics.csv` thành bảng ablation |
| `sampled_loo_evaluator.py` | Giao thức thay thế: LOO + 100 negative (để đo Kendall τ) |

---

## 6. `configs/` — 159 file YAML

| Nhóm | Số | Nội dung |
|---|---|---|
| `cb1-*` | 15 | Sweep TF-IDF: thêm tag, trọng số thể loại, bigram, item-KNN, chuẩn hoá |
| `cb2-*` | 21 | Sweep BM25: lưới `k1` × `b` |
| `cb3-*` | 22 | Sweep Genome: chuẩn hoá, ngưỡng τ, SVD, item-KNN, Ridge |
| `cb4-*` | 35 | Sweep Hybrid: pop prior, late fusion, time-decay, 3 thành phần |
| `cf-*` / `rs-*` | 51 | Sweep CF: ALS (`factors` × `α` × `reg` × `iterations`), Item-CF, BPR |
| `baseline-*` | 2 | Random, MostPop |
| `best-*` | 13 | Config đã khoá tham số, dùng cho bảng chính |

Tất cả 159 config đều **đã chạy** trên `dev` (20.000 user) — bằng chứng từng config nằm
ở `results/final/ablation-{cf,cb1,cb2,cb3,cb4}.csv` (144 dòng: 51 CF + 93 CB, mỗi dòng
mang `params` + `split_id`/`catalog_id`).

---

## 7. `results/final/` — sản phẩm báo cáo

27 bảng CSV được commit; người clone repo xem được ngay mà không cần chạy lại.

| Nhóm | Bảng | Nội dung |
|---|---|---|
| Benchmark | `benchmark-cf.csv` (5 dòng) | Bảng CF chính, 158.774 user |
| | `benchmark-cb.csv` (4 dòng) | Bảng CB chính, cùng split/catalog |
| | `benchmark-coldstart.csv` (9 dòng) | 1.381 phim bị giữ lại |
| | `benchmark-{cf,cb}-dev.csv` | Bảng `dev` dùng để chọn tham số |
| Tổng hợp | `dev-summary.csv`, `full-summary.csv`, `coldstart-item.csv` | Rút gọn cho chương kết quả |
| Kiểm định | `bootstrap-ci.csv` | CI 95% của NDCG@10 từng mô hình |
| | `wilcoxon-{cf,cb,coldstart,cf-dev,cb-dev}.csv` | Wilcoxon ghép cặp trong từng bảng |
| Ngoài độ chính xác | `beyond-accuracy-{full,coldstart}.csv` | Coverage, Gini, novelty, ILD |
| Phân tầng | `coldstart-user-buckets.csv` | Metric theo độ dài lịch sử user |
| | `tag-leakage-strata.csv`, `tag-leakage-summary.csv` | Chống rò rỉ tag theo kỷ nguyên |
| | `weight-mode-buckets.csv`, `weight-mode-paired-tests.csv` | So 3 chế độ trọng số hồ sơ |
| Sweep | `ablation-{cf,cb1,cb2,cb3,cb4}.csv` | 144 dòng bằng chứng đã quét đủ (51 CF + 93 CB) |
| Dữ liệu | `dataset-summary.csv` | Thống kê dataset |

---

## 8. `tests/` — 17 file

| Nhóm test | Chặn cái gì |
|---|---|
| `test_topk_ranking_metrics.py` | Định nghĩa metric sai (đặc biệt `P@k = hits/k`, không phải `hits/len(top_k)`) |
| `test_eval_metric_correctness.py` | Metric tính sai trên ca lạ |
| `test_eval_leakage_guards.py` | Rò rỉ: mô hình chạm nhãn test, item ngoài catalog, user không có tín hiệu |
| `test_batched_scoring_evaluator.py` | Chấm điểm theo lô sai |
| `test_split_temporal_per_user.py` | Chia sai chiều thời gian |
| `test_phase2_content_models.py`, `test_phase2_scorers.py`, `test_phase2_rs_models.py` | Logic từng mô hình/scorer |
| `test_phase3_models.py`, `test_phase3_hybrid.py`, `test_phase3_score_fusion.py` | Fusion + chuẩn hoá điểm |
| `test_phase3_benchmark_tables.py` | Bảng benchmark lẫn split / lệch model; runner không phủ hết bảng; bảng final thiếu bằng chứng protocol |
| `test_phase3_docs_and_final_tables.py` | Tài liệu quay lại tuyên bố cũ (biên độ ALS, số config sweep, thứ hạng CB) |
| `test_phase4_beyond_accuracy.py`, `test_phase4_protocols.py`, `test_phase4_pipeline_guards.py` | Chỉ số ngoài độ chính xác, split cold-start, guard chống kết quả mồ côi |

---

## 9. Quy ước cần biết

| Quy ước | Giá trị | Ở đâu |
|---|---|---|
| Seed | 42 | `src/paths.py::SEED` |
| Ngưỡng relevant | rating ≥ 4,0 | `src/paths.py::RELEVANCE_THRESHOLD` |
| Tỷ lệ test | 20% theo thời gian, mỗi user | `src/data/split_temporal_per_user.py` |
| Min train | 5 tương tác | `src/data/split_temporal_per_user.py::MIN_TRAIN` |
| Batch chấm điểm | 2.000 user | `src/eval/batched_scoring_evaluator.py` |
| K đo | 5, 10, 20, 50 | `src/eval/topk_ranking_metrics.py::DEFAULT_K_LIST` |
| Mẫu dev | 20.000 user | `src/paths.py::N_DEV_USERS` |

**Mọi dòng kết quả trong `results/final/` mang `split_id` + `catalog_id` + `seed`** — đó là
bằng chứng chúng chạy trên cùng tập ứng viên, không phải lời tuyên bố. Test
`test_final_tables_have_protocol_evidence` chặn việc mất các cột này.
