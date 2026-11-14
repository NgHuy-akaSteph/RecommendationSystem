# Project Changelog

Ghi lại các thay đổi đáng kể của `RecommendationSystem`. Mỗi mục ghi ngày, phạm vi,
thay đổi và cách kiểm chứng.

---

## 2026-11-15 — Viết báo cáo Chương 5–6 + vá 3 sai lệch

**Phạm vi:** Viết prose cho Chương 5 (Kết quả) và Chương 6 (Chống rò rỉ & tranh luận)
vào `docs/report/`, đồng thời vá ba sai lệch đã phát hiện trong tài liệu và script.
Không đổi thuật toán, không chạy lại thực nghiệm.

### Ba sai lệch đã vá

| # | Vị trí | Vấn đề | Cách sửa |
|---|---|---|---|
| 1 | `notes/report-structure.md` §5.2 | Quy ước biên độ sweep **trộn hai loại**: "CF 17,94%" là biên độ riêng lưới ALS, biên độ toàn nhóm CF thật là **73,65%** | Tách thành **hai bảng** (theo model / theo nhóm), ghi công thức `(max−min)/max` + protocol `dev`; thêm test chặn hồi quy |
| 2 | `scripts/run_sampled_loo_comparison.py` | Merge theo `exp_name` → **mất dòng `cb4-04`** + **dòng trùng** (2 config `rs-itemcf` trùng `exp_name`) → Kendall τ sai | Dedup `best-*.yaml` theo `exp_name`; đổi nguồn metric sang `results/{exp_name}_metrics.csv` (đủ 12 `exp_name`, có `split_id`/`catalog_id`); assert chống trùng/mất |
| 3 | `results/figures/` | §5.2 yêu cầu hình "thanh ngang so sánh toàn sweep" nhưng **không tồn tại** | Thêm `figure_sweep_range()` → `results/figures/results-sweep-range.png` |

### Đính chính quy ước biên độ ALS (18,65% → 17,94%)

Repo từng có **hai quy ước biên độ cùng tồn tại**: `(max−min)/max` → ALS 17,94% và
`(max−min)/trung bình` → ALS 18,65%. `README.md` + changelog ghi 18,65%, còn
`notes/report-structure.md` ghi 17,94% → **hai tài liệu nói khác nhau**.

Đã **chốt một quy ước duy nhất**: `(max − min) / max` (mẫu số max) — khớp với biên
độ các nhóm khác trong cùng bảng. Sửa đồng thời `README.md`,
`docs/project-changelog.md`, `plans/.../phase-03-...md` và test
`test_docs_state_full_als_spread` (nay chặn cả việc dùng lại 18,65% ngoài chú thích đính chính).

### Nội dung báo cáo mới

| File | Nội dung |
|---|---|
| `docs/report/README.md` | Mục lục 8 chương, nguồn sự thật, quy ước bất di bất dịch, mapping chương ↔ file CSV, 3 giao thức |
| `docs/report/05-experimental-results.md` | §5.1 bảng CF/CB (`full`) · §5.2 cân bằng sweep + biên độ theo model/nhóm + đối chiếu `dev`↔`full` · §5.3 cold-start + `frac_unseen@10` + Wilcoxon · §5.4 bootstrap CI + lý do không so chéo CF↔CB · §5.5 nhóm user |
| `docs/report/06-leakage-analysis-and-discussion.md` | §6.1 metadata hậu kỳ + confound Pearson 0,768 · §6.2 phân tầng 3 kỷ nguyên + đối chứng "chỉ genres" · §6.3 chế độ trọng số hồ sơ · §6.4 metric ngoài độ chính xác · §6.5 bảng "khi nào thắng" + **mục giới hạn của chính kết luận chống rò rỉ** |

### Thay đổi mã nguồn

| Nhóm | Nội dung |
|---|---|
| Mới | `src/eval/sweep_spread.py` — nguồn sự thật cho biên độ sweep (theo model / theo nhóm), công thức `(max−min)/max`, `MODEL_OWNER` xác định file sở hữu mỗi model |
| Mới | `scripts/check_report_numbers.py` — đối chiếu **mọi** số trong báo cáo với `results/final/*.csv`; số nào không truy được thì `exit 1` |
| Sửa | `scripts/run_sampled_loo_comparison.py` — dedup `exp_name`, nguồn metric gốc, assert chống trùng/mất |
| Sửa | `src/eval/make_result_figures.py` — thêm `figure_sweep_range()`; 8 hình cũ **không đổi byte** (đã kiểm bằng md5) |
| Sửa | `notes/report-structure.md` §5.2 — tách biên độ theo model / theo nhóm |

### Test chặn hồi quy mới

| Test | Chặn cái gì |
|---|---|
| `tests/test_report_sweep_spreads.py` (12 test) | Biên độ trong tài liệu khớp `ablation-*.csv` (**tính động**, không hard-code); cấm trộn model/nhóm; cấm quay lại quy ước mẫu số trung bình |
| `tests/test_report_numbers_consistency.py` (9 test) | Số trong báo cáo truy được về `results/final/`; kèm phép thử **chèn số sai** để chứng minh checker thật sự bắt lỗi |
| `tests/test_phase3_sampled_loo_comparison.py` (12 test) | Script LOO không đọc `dev-summary.csv` trung gian; dedup đúng; merge không trùng/mất dòng |

### Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| `venv/bin/python scripts/check_report_numbers.py` | **0 lệch** (2 file báo cáo × 27 bảng CSV) |
| `pytest tests/` | xanh toàn bộ |
| 8 hình cũ trong `results/figures/` | md5 **không đổi** — chỉ thêm `results-sweep-range.png` |

### Còn lại (đã gác theo yêu cầu)

| Việc | Lý do |
|---|---|
| Chạy sweep trên protocol `full` (144 config) | Người dùng tự chạy sau khi có báo cáo |
| Chạy thật `run_sampled_loo_comparison.py` để lấy Kendall τ | Script đã sửa + test; cần ~20 phút chạy |
| Viết Chương 1–4, 7, 8 | Ngoài phạm vi đợt này |
| Đẩy commit lên `origin/v1` | Chưa được yêu cầu |

---

## 2026-11-14 — Sửa sai lệch số liệu trước khi viết báo cáo

**Phạm vi:** Audit toàn repo trước khi viết báo cáo. Tìm và sửa **6 sai lệch** giữa mô tả
trong tài liệu và bằng chứng trong `results/`. Không đổi thuật toán, không đổi số liệu đã
công bố.

### Vấn đề phát hiện và cách sửa

| # | Vấn đề | Sửa |
|---|---|---|
| 1 | `README.md` + changelog ghi thứ hạng "CB-1 > CB-2 > CB-3" ở bảng phân tầng kỷ nguyên — **ngược lại** với `results/final/tag-leakage-strata.csv` (CB-2 dẫn ở cả 3 tầng) | Sửa thành "CB-2 ≥ CB-1 > CB-3", bổ sung số ở cả 3 tầng để người đọc tự kiểm |
| 2 | Số config sweep ghi "CB 85" — thực tế **93** (sweep mở rộng thêm 8 config CB sau đó) | Sửa 85 → 93 ở README, `benchmark_tables.py`, `build_final_tables.py`, phase-03 docs; changelog giữ 85 kèm đính chính để làm lịch sử |
| 3 | `results/ablation-*.csv` (144 dòng bằng chứng sweep) bị gitignore → luận điểm "đã cân bằng sweep" **không kiểm chứng được** từ repo | Copy 5 bảng vào `results/final/` để commit |
| 4 | 38 file `*_metrics.csv` thiếu cột `split_id`/`catalog_id` (chạy trước khi thêm cột) → `dev-summary.csv` mất bằng chứng protocol ở dòng CB-4 | Chạy lại 7 config có trong bảng ablation, sinh lại bảng |
| 5 | Thiếu `docs/codebase-summary.md`, `docs/system-architecture.md`, `docs/development-roadmap.md` (chuẩn dự án) | Viết mới 3 file |
| 6 | `scripts/run_shared_benchmark.py` thiếu `best-cb4-hybrid-content.yaml` trong `CONFIGS` → chạy lại sẽ fail `benchmark-cb.csv` | Thêm config + test chặn tách lệch runner/bảng |

### Test chặn hồi quy mới

| Test | Chặn cái gì |
|---|---|
| `test_docs_state_real_sweep_counts` | Đối chiếu số config trong tài liệu với `configs/` thật (không hard-code) |
| `test_final_tables_have_protocol_evidence` | Mọi dòng trong `results/final/*-summary.csv` + `benchmark-*.csv` phải có `split_id`/`catalog_id` khác rỗng |
| `test_ablation_tables_have_protocol_evidence` | Mọi dòng trong `results/final/ablation-*.csv` phải có bằng chứng protocol, không trùng `exp_name` |
| `test_ablation_covers_every_config_of_its_group` | Bảng ablation phải phủ **đúng** mọi config của nhóm — không thiếu (thiếu 33/51 dòng CF) không thừa |
| `test_shared_benchmark_covers_all_table_specs` | `CONFIGS` của runner phải ⊇ mọi config trong `CF_SPECS + CB_SPECS` |
| `test_docs_do_not_claim_cb1_above_cb2` | Cấm tuyên bố thứ hạng CB sai trong README/changelog |

### Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| `pytest tests/` | **234 passed** (219 cũ + 15 test mới), 0 failed |
| Chạy lại 7 config thiếu `split_id` | NDCG@10 **không đổi 1 chữ số** (diff = 0.00e+00 cho cả 7) |
| `benchmark_tables --protocol full` | `benchmark-cf.csv` / `benchmark-cb.csv` / `benchmark-coldstart.csv` **không đổi byte** (md5 khớp) |
| `build_final_tables.py` | Chỉ `dev-summary.csv` thay đổi (vá cột bằng chứng); 26 bảng còn lại giữ nguyên |
| `git add -An --dry-run results/` | 22 bảng final đã track + 5 bảng ablation mới + `dev-summary.csv` sửa |
| Bảng ablation | 111 dòng, **100% có `split_id` + `catalog_id`** |

---

## 2026-11-14 — Viết lại khung báo cáo: bằng chứng sweep CF chỉ phủ 18/51 config

**Phạm vi:** Viết khung báo cáo mới (`notes/report-structure.md`) khớp với hiện trạng thực
tế (25M / 9 mô hình). Khung cũ viết cho MovieLens 1M / 2 methods / 18 config — lệch
hoàn toàn.

### Vấn đề phát hiện khi đối chiếu khung với `results/final/`

Mỗi con số trong khung báo cáo phải **đối chiếu được** với bảng kết quả — nếu không,
khung chỉ là bản chép lại của chính những sai lệch đã sửa. Đối chiếu lộ ra:

| | Tuyên bố | Bằng chứng commit |
|---|---|---|
| Sweep CF | 51 config | `ablation-cf.csv` chỉ **18 dòng** |

**Nguyên nhân:** `collect_results_to_table.write_ablation_tables` lọc bằng
`prefix="cf-"`, bỏ sót họ `rs-*` (Item-CF, BPR — model port từ repo `recommendation-cb`).
33 config `rs-*` đã chạy trên `dev` nhưng không bao giờ vào bảng bằng chứng.

Sửa collector → lộ tiếp: 31 trong 33 file metrics `rs-*` **chạy trước khi thêm cột
`split_id`/`catalog_id`** → không có bằng chứng protocol.

### Đã làm

| Việc | Chi tiết |
|---|---|
| Sửa collector | `ABLATION_PREFIXES` cho phép mỗi nhóm nhiều tiền tố; nhóm CF lấy cả `cf-` và `rs-` |
| Chạy lại | 30 config `rs-*` trên `dev` (30/30 OK, 0 lỗi) |
| Kiểm chứng số liệu | **NDCG@10 không đổi 1 chữ số** cho cả 33 file `rs-*` (diff < 1e-12) — đây là vá bằng chứng, không phải đổi kết quả |
| Sinh lại bảng | `ablation-*.csv` giờ có **144 dòng = 51 CF + 93 CB**, 100% có `split_id` + `catalog_id`, `exp_name` không trùng |
| Test mới | `test_ablation_covers_every_config_of_its_group` — bảng ablation phải phủ **đúng** mọi config của nhóm, không thiếu không thừa |
| Viết khung | `notes/report-structure.md` — 8 chương, 40–45 trang, mọi số liệu đã đối chiếu |

### Vì sao đáng sửa dù không nằm trong plan

Luận điểm trung tâm của báo cáo là "CF thắng CB **đã cân bằng về sweep**". Bằng chứng
thiếu 33/51 dòng CF thì tuyên bố đó **không kiểm chứng được từ repo** — người đọc
mở `ablation-cf.csv` thấy 18 dòng sẽ tự hỏi 33 config kia ở đâu.

---

## 2026-11-14 — Dọn artifact repo: `.gitignore` viết lại + xoá dữ liệu protocol cũ

**Phạm vi:** Dọn file rác. **Không** đụng thuật toán, **không** đụng số liệu.

### Vấn đề phát hiện

`.gitignore` cũ có 2 lỗi làm repo lộ artifact:

- `./notes` và `./plans` — gitignore **không hiểu** tiền tố `./`, nên hai dòng này không có
  tác dụng. Kiểm chứng: `git check-ignore notes/roadmap.md` → không match.
- Không có `venv/`, `__pycache__/`, `.pytest_cache/` → 84 file `.pyc` lộ diễn trong
  `git status`.
- `results/` chưa được ignore → 423 file artifact (507 MB) sẵn sàng bị commit nhầm.

Ngoài ra `data/legacy/` (461 MB) là dữ liệu của protocol cũ (global 70/30) đã bị xoá ở
Phase 04a — grep toàn repo chỉ còn 1 dòng nhắc tên trong mục Phase 00 dưới đây, **không
module nào còn đọc**.

### Đã làm

| Việc | Chi tiết |
|---|---|
| Xoá `data/legacy/` | `train_ml25m.csv` (451 MB), `test_ml25m.csv` (15 MB), `train_implicit_user_item.npz` (17 MB) |
| Viết lại `.gitignore` | thêm `venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`; sửa `./notes` → `notes/`, `./plans` → `plans/` |
| Ignore `results/` | giữ **duy nhất** `results/final/*.csv` (22 bảng) + `results/figures/*.png` (9 hình); 393 file trung gian (`*_topk.npy` 443 MB, `*_peruser.npz` 63 MB, `*_metrics.csv`) bị ignore |
| Xoá file trùng | `results/beyond-accuracy-{full,coldstart}.csv` (trùng **byte-for-byte** với bản trong `results/final/`), `results/wilcoxon.csv` (bản cũ đã bị `results/final/wilcoxon-{cf,cb,coldstart}.csv` thay) |
| Xoá cache | `__pycache__/` (84 file), `.pytest_cache/` |

### Sửa code đi kèm

`results/beyond-accuracy-*.csv` nằm ở gốc `results/` là artifact trung gian (bị ignore), nên
hai chỗ đọc nó phải ưu tiên bản trong `results/final/`:

- `src/paths.py` — thêm `beyond_accuracy_csv(protocol)`: ưu tiên `results/final/`, fallback
  về gốc `results/` để pipeline cũ vẫn chạy được.
- `src/eval/benchmark_tables.py`, `scripts/build_final_tables.py` — dùng hàm đó thay cho
  `paths.RESULTS_DIR / f"beyond-accuracy-{protocol}.csv"`.
- `src/eval/make_result_figures.py` đọc `results/final/beyond-accuracy-full.csv` qua
  `_load()` — không cần sửa.

### Ảnh hưởng tới người clone repo

`results/` trong git chỉ có **bảng báo cáo cuối + hình**. Muốn chạy lại kiểm định thống kê
(`beyond_accuracy_metrics`, `statistical_tests`, `user_bucket_analysis`) thì phải chạy lại
pipeline — chúng đọc `*_peruser.npz` và `*_topk.npy` vốn không có trong git.

### Kiểm chứng

```bash
git check-ignore -v --no-index results/full_cf-als-f064-a005-reg0p1_topk.npy  # phải match
git check-ignore -v --no-index results/final/benchmark-cf.csv                # phải KHÔNG match
git check-ignore -v --no-index notes/roadmap.md                              # phải match
git add -An --dry-run results/    # đúng 31 file: 22 CSV + 9 PNG
venv/bin/python -m pytest tests/  # 219 passed
```

### Ghi chú

- `*_topk.npy` (443 MB) và `*_peruser.npz` (63 MB) **giữ trên đĩa** (chỉ bị gitignore):
  xoá đi thì mỗi lần chạy lại phân tích phải train 9 mô hình × 3 giao thức.
- 10 file `notes/*` + 8 file `plans/*` đã commit từ trước vẫn được track (gitignore không
  untrack file đã có trong index) — không đụng tới.

---

## 2026-11-14 — Phase 04a: Xoá code legacy của protocol cũ

**Phạm vi:** Xoá code + tài liệu cũ. **Không** đụng thuật toán, không đụng số liệu.

### Đã xoá (23 file, `git rm`)

| Nhóm | File |
|---|---|
| Evaluator cũ | `src/evaluation/` — `__init__.py`, `evaluation_export.py`, `evaluation_pipeline.py`, `evaluation_protocols.py`, `evaluation_validation.py`, `recommender_evaluator.py`, `statistical_analysis.py` |
| Recommenders cũ | `src/recommenders/` — `__init__.py`, `als_recommender.py`, `bpr_recommender.py`, `item_cf.py`, `popularity.py` |
| Loader cũ | `src/data_loader.py` (260 dòng) |
| Notebooks | `notebooks/` — 5 notebook + `bpr_experiment_script.py` + `.ipynb_checkpoints/` |
| Script cũ | `scripts/export_models.py` |
| Test cũ | `test_bpr_quick.py`, `test_bpr_import.py` |
| Thư mục rỗng | `models/`, `reports/` (chỉ chứa `reports/figures/` rỗng) |

### Đã làm thêm

| Nhóm | Nội dung |
|---|---|
| Mới | `scripts/add_legacy_banner.py` — gắn banner "tài liệu legacy" đầu 8 file `docs/*.md`. **Idempotent**: chạy lại không nhân bản banner |
| Sửa | `README.md` — mục "Code legacy (sẽ xoá ở Phase 04)" → "Protocol cũ đã bị xoá", kèm dẫn chiếu sang `docs/data-processing-comparison-across-projects.md` |
| Sửa | `plans/.../phase-04-*.md` — đánh dấu 4a complete, cập nhật todo + success criteria |

### Vì sao không cần `docs/legacy-protocol-rs.md`

Phase 04 ban đầu yêu cầu trích protocol cũ từ `src/data_loader.py` + notebooks vào file
riêng trước khi xoá. Kiểm tra thực tế cho thấy **thông tin đó đã nằm đủ** trong
`docs/data-processing-comparison-across-projects.md`:

| Mục | Nội dung |
|---|---|
| 2.1 | Pipeline RS — `ML25MLoader` / `DataPreprocessor`, có số dòng tham chiếu |
| 4.1 | Split global temporal 70/30 (`src/data_loader.py:66`) |
| 6.1 | Dựng ma trận — `build_user_item_matrix` |
| 7.1 | Bốn lỗi đánh giá B1–B4 kèm trạng thái tương ứng ở `src/eval/` |

Viết thêm file riêng sẽ lặp lại nội dung đã có. User cũng chọn **xoá thẳng notebook không
trích lại** — chấp nhận mất tài liệu lịch sử.

### Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| `pytest tests/` | **219 passed**, 0 failed |
| Grep `*.py` (trừ venv) | Không còn import nào tới `src.evaluation` / `src.recommenders` / `data_loader` / `export_models` — chỉ còn trong docstring |
| Smoke test đường chạy | `run_experiment_from_config configs/baseline-mostpop.yaml` → NDCG@10 = **0,0723**, khớp đúng bảng cũ |
| Import nền tảng mới | `model_registry` (11 model), `build_interaction_matrices`, `benchmark_tables` — OK |
| Cấu trúc repo | `src/` còn đúng 3 thư mục: `data/`, `eval/`, `models/` |

---

## 2026-11-14 — Chạy nốt sweep CB (85 config) + đưa CB-4 vào bảng chính

**Phạm vi:** Sweep CB đủ trên `dev`, chạy lại `full`/`coldstart` cho các config mới,
cập nhật registry bảng benchmark, README, changelog, phase-03 docs, tests.

### Vì sao

Audit trước đó phát hiện "CF thắng CB" được đo với **thiên lệch về độ bao phủ sweep**:
CF đã quét 49 config còn CB mới 8 (85 config CB **chưa chạy** — trạng thái trước sweep).
Kết luận đó chỉ là *thận trọng chứ chưa cân bằng*. Để biết kết luận có đổi chiều không
thì phải quét đủ CB.

### Kết quả sweep

| Nhóm | Config | Trạng thái trước | Trạng thái sau |
|---|---|---|---|
| CF (ALS + Item-CF + BPR) | 51 | 49 chạy | **51 chạy** (thêm `cf-itemknn-k050`, `cf-itemknn-k100`) |
| CB (CB-1…CB-4) | 85 | 8 chạy | **85 chạy** |

> **Đính chính (2026-11-14, sau audit báo cáo):** con số 85 là của **thời điểm sweep**.
> Sweep đã được mở rộng thêm 8 config CB (CB-4 late-fusion, CB-3 SVD/biến thể) nên
> hiện tại là **CF 51 / CB 93** — 93/93 đều đã chạy. Con số 85 ở đây giữ nguyên để
> làm lịch sử; xem entry "Sửa sai lệch số liệu trước khi viết báo cáo" ở đầu file.

Chạy 85 config CB mất ~50 phút (tuần tự, mỗi config fit + score 20.000 user + tính
16 metric), **0 lỗi**. Phân bố thời gian trên `dev`:

| Nhóm | Số config | Tổng thời gian | Trung bình |
|---|---|---|---|
| CB-1 TF-IDF | 16 | 182s | 11,4s |
| CB-2 BM25 | 21 | 294s | 14,0s |
| CB-3 Tag Genome | 19 | 93s | 4,9s |
| CB-3 Ridge | 3 | 14s | 4,8s |
| CB-4 Hybrid | 25 | 331s | 13,2s |

### Kết luận đổi chiều — giờ đã cân bằng

Sau khi sweep đủ, **CF thắng rõ ràng** ở giao thức thường, và đây **không còn** là hệ quả
của việc một bên được tinh chỉnh nhiều hơn:

| Mô hình | NDCG@10 (`dev`) | NDCG@10 (`full`) |
|---|---|---|
| ALS | 0,1383 | **0,1375** |
| Item-CF (RS) | 0,1297 | 0,1290 |
| BPR (RS) | 0,1106 | 0,1099 |
| **CB-4 Hybrid** (best CB) | 0,1067 | 0,1054 |
| CB-2 BM25 | 0,0986 | 0,0998 |
| CB-1 TF-IDF | 0,0921 | 0,0937 |
| CB-3 Tag Genome | 0,0703 | 0,0713 |

CF thắng CB-4 khoảng **23%** (`full`). Trước đây (chỉ 8 config CB) là 38%.

### CB-4 vào bảng chính — và bẫy `pop_prior`

CB-4 late-fusion (BM25 + Tag Genome) là mô hình content mạnh nhất nên được thêm vào
`CB_SPECS`. Nhưng phải chọn **bản không có popularity prior**:

| Bản CB-4 | NDCG@10 (`full`) | NDCG@10 (`coldstart`) | `frac_unseen@10` (coldstart) |
|---|---|---|---|
| `cb4-04-late-bm25-w075` (không prior) | 0,1054 | **0,0347** | 0,2223 |
| `cb4-06-late-bm25-pop-lam0040` (λ_pop=0,4) | **0,1081** | 0,0010 | 0,0036 |

`pop_prior` làm **đẹp** số liệu giao thức thường nhưng **giết** cold-start: mọi phim
held-out có popularity = 0 nên bị đẩy xuống đáy danh sách — `frac_unseen@10` rơi từ
0,2223 xuống 0,0036. Bảng chính lấy bản content thuần; bản có prior chỉ để đối chiếu.

### Thay đổi

| Nhóm | Nội dung |
|---|---|
| Chạy | 85 config CB trên `dev` + 6 config trên `full`/`coldstart` (0 lỗi) |
| Sửa | `src/eval/benchmark_tables.py` — thêm `CB-4 Hybrid` vào `CB_SPECS` (bản không pop prior); cập nhật docstring số config |
| Sửa | `README.md` — bảng CB + cold-start có CB-4; mục "Cân bằng sweep" đổi từ "cảnh báo" sang "đã cân bằng"; thêm mục giải thích bẫy `pop_prior` |
| Sửa | `docs/project-changelog.md` — entry này; mục "đính chính" cũ giữ nguyên làm lịch sử |
| Sửa | `plans/.../phase-03-*.md` + `plan.md` — cập nhật số config và kết luận |
| Sửa | `tests/test_phase3_benchmark_tables.py` — bỏ hard-code "8 mô hình", so với registry |
| Sửa | `tests/test_phase3_docs_and_final_tables.py` — kỳ vọng số config mới (CF 51 / CB 85) + test chặn tuyên bố "sweep chưa đủ" |
| Regenerate | `results/final/benchmark-{cf,cb,coldstart}.csv`, `wilcoxon-*`, `results/final/*`, `results/figures/*` |

### Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Sweep CB | 85/85 config, **0 lỗi**, ~50 phút |
| `full` / `coldstart` | 6 config mới chạy, 0 lỗi |
| `pytest tests/` | **216 passed**, 0 failed |
| Bảng cold-start | CB-4 Hybrid 0,0347 (đứng đầu), mọi mô hình CF = 0,0000 |

---

## 2026-11-14 — Phase 00: Port hạ tầng và pipeline của `recommendation-cb`

**Phạm vi:** Hạ tầng + pipeline dữ liệu + evaluator. Không đụng logic thuật toán.

### Thay đổi

| Nhóm | Nội dung |
|---|---|
| Mới | `src/__init__.py` — ép UTF-8 cho stdout/stderr |
| Mới | `src/paths.py` — port từ CB, `RAW_DIR` → `data/raw/ml-25m` |
| Mới | `src/data/` (5 file) — nạp dữ liệu, đặc trưng nội dung, split per-user, ma trận tương tác, EDA |
| Mới | `src/models/` (12 file) — 9 mô hình: random, mostpop, TF-IDF, BM25, Genome, Genome-Ridge, ItemKNN-CF, ALS, Hybrid late-fusion |
| Mới | `src/eval/` (13 file) — metric top-K, evaluator full-ranking theo lô, registry, runner, beyond-accuracy, kiểm định thống kê, bucket theo user, cold-start item split, sampled LOO, phân tầng tag leakage |
| Mới | `tests/` (12 file + `conftest.py`) — 142 test |
| Mới | `configs/` — 22 file YAML (chọn lọc từ 120 của CB) |
| Mới | `scripts/` (9 file) — trừ pipeline DOCX |
| Mới | `pyproject.toml` — cấu hình pytest (`pythonpath = ["."]`, `testpaths = ["tests"]`) |
| Sửa | `requirements.txt` — thêm `pyarrow`, `pytest`, `PyYAML`, `threadpoolctl`, `tqdm`, `tabulate` |
| Di chuyển | `data/processed/{train,test}_ml25m.csv`, `train_implicit_user_item.npz` → `data/legacy/` (cách ly protocol cũ) |
| Xoá | `tests/test_report_docx_builder.py`, `scripts/build_report_docx.py`, `scripts/check_report_consistency.py` (không port `src/report/` theo scope chốt) |

### Giao thức dữ liệu mới (thay thế protocol cũ)

- **Catalog:** lọc còn 13.816 phim có Tag Genome (giữ 98,7% rating).
- **Split:** temporal per-user 80/20, seed 42, `relevant = rating ≥ 4.0`.
- **Leakage:** chống ở tầng cấu trúc — `TrainData` (model thấy) tách khỏi `Dataset` (`test_rel_csr`, chỉ evaluator thấy).
- **Confidence ALS:** `c_ui = 1 + α` chỉ trên tương tác relevant (thay vì `1 + α·rating` trên toàn bộ rating của protocol cũ).

### Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Tiền xử lý | `users=162,540` · `items=13,816` · `ratings=24,674,113` · `user hợp lệ=158,774` — khớp tuyệt đối CB |
| EDA | 25/25 chỉ số `dataset-summary.csv` khớp tuyệt đối CB |
| Test | `pytest tests/` → 142 passed |
| Benchmark dev | 9/9 model khớp CB, max \|Δ NDCG@10\| = 5.98e-06 |
| Phân tầng kỷ nguyên | `tag-leakage-strata.csv` khớp CB (Pearson(tag, pop) = 0,768; Spearman = 0,700) |

### Tương thích ngược

`src/evaluation/`, `src/recommenders/`, `src/data_loader.py` **vẫn còn** (giữ tới Phase 04).
Trong thời gian này có 2 evaluator cùng tồn tại — `src/eval/` là đường chạy chính, `src/evaluation/`
không được dùng để sinh số liệu báo cáo.

### Lưu ý

Chưa commit. Theo quyết định 2026-11-14, không `git add`/`git commit`/`git push` khi chưa được user cho phép.
`.gitignore` giữ nguyên.

---

## 2026-11-14 — Phase 01: Đảm bảo tính đúng của evaluator mới

**Phạm vi:** `src/eval/` (evaluator mới port từ CB). Không sửa `src/evaluation/` cũ.

### Bốn lỗi B1–B4 — đã **loại bỏ khỏi đường chạy**, không phải "đã sửa"

| # | Lỗi của protocol cũ | Kiểm chứng ở `src/eval/` |
|---|---|---|
| B1 | `precision_at_k` = `hits / len(top_k)` | ✅ `hits / k` — có test với case model trả ít item hơn k |
| B2 | `except Exception` → `return []` | ✅ exception nổi lên; thêm chặn `NaN` trong `score_users` |
| B3 | Ghép cặp thống kê theo thứ tự dict | ✅ `load_all` raise khi tập `eval_users` lệch |
| B4 | `validate_dataset_split` mặc định 0,7 | ✅ không còn hàm này; split kiểm bằng test riêng (`TEST_RATIO = 0.2`) |

### Hai hàng rào bổ sung (chưa có ở bản của CB)

**1. Item ngoài catalog → raise** (`src/eval/topk_ranking_metrics.py`)

Không có hàng rào này thì `searchsorted` so sánh luôn `False` và item sai bị tính là "không trúng" —
một lỗi ánh xạ item biến thành điểm số thấp hơn mà không ai biết.

**2. User không có tín hiệu train → metric = 0 có nguyên tắc** (`src/eval/batched_scoring_evaluator.py`)

Đo thực nghiệm: **21/20.000 user** (0,10%) không có tương tác relevant nào trong train. Mô hình nội
dung cho họ điểm 0 trên toàn bộ item, nên `argpartition` trả về item theo thứ tự chỉ số — danh sách do
cách lưu trữ quyết định chứ không phải do thuật toán. Trước guard, user này có cơ hội "trúng" item
relevant chỉ vì item đó tình cờ nằm ở chỉ số thấp.

Guard mới dùng `rows_without_signal()`: hàng mà mọi điểm còn lại đều bằng nhau (kể cả hàng không còn
điểm hữu hạn nào) → metric bằng 0 **một cách có nguyên tắc**. User **vẫn ở lại** trong tập đánh giá —
loại bỏ họ sẽ là thao túng mẫu.

### Kiểm chứng hàng rào leakage (3 tầng)

| Tầng | Nội dung | Kết quả đo |
|---|---|---|
| L1 | `TrainData` / `Dataset` tách cấu trúc | ✅ mô hình không có đường nào tới `test_rel_csr` |
| L2 | `ts_train ≤ ts_test` cho mọi user | ✅ **0** user vi phạm thật |
| L3 | Metadata hậu kỳ (Tag Genome) | ✅ ghi nhận + bảng phân tầng theo kỷ nguyên |

**Phát hiện mới về L2:** 29.826/158.774 user (18,8%) có `ts_train_max == ts_test_min`. Đây **không**
phải rò rỉ: 57,6% cặp `(user, ts)` có hơn một rating (người dùng xem nhiều phim cùng lúc), và khi
timestamp trùng thì không có "trước/sau" để phân biệt. Việc vi phạm thật là `ts_train > ts_test` —
**0 user**. Test được viết để phân biệt hai trường hợp và ghi nhận tỷ lệ 18,8% thay vì giấu đi.

**Kiểm tra bổ sung:** 0 cặp `(user, item)` trùng giữa train và test (trên 4.831.397 dòng test) —
nếu trùng thì item đó bị mask khỏi danh sách gợi ý nên user không bao giờ "trúng" được, làm giảm
mẫu đánh giá âm thầm.

### Test

| File | Số test |
|---|---|
| `tests/test_eval_metric_correctness.py` (mới) | 17 |
| `tests/test_eval_leakage_guards.py` (mới) | 11 |
| Toàn bộ `tests/` | **170 passed** (142 trước đó + 28 mới) |

### Không làm sai lệch số liệu

9/9 model benchmark dev khớp CB, max \|Δ NDCG@10\| = `5.98e-06` — **không đổi** so với trước Phase 01.
Lý do: 21 user không tín hiệu vốn đã "được 0" do may rủi thứ tự chỉ số, nên guard chỉ biến may mắn
thành nguyên tắc mà không đổi con số.

### Đánh dấu legacy

`src/evaluation/__init__.py` giờ in `DeprecationWarning` khi được import, nêu rõ 4 lỗi B1–B4 và
đường chạy chính là `src/eval/`. `README.md` liệt kê bảng B1–B4.

### Lưu ý

Chưa commit. Không `git add`/`git commit`/`git push` khi chưa được user cho phép.

---

## 2026-11-14 — Phase 02: Đưa Item-CF và BPR sang giao thức batch chung

**Phạm vi:** `src/models/` (2 model mới + 1 helper), `src/eval/model_registry.py`,
`configs/` (2 file), `tests/` (1 file mới). **Không sửa `src/recommenders/` cũ.**

### Thay đổi

| Nhóm | Nội dung |
|---|---|
| Mới | `src/models/rs_itemcf_model.py` — `RsItemCfModel` (`rs-itemcf`) |
| Mới | `src/models/rs_bpr_model.py` — `RsBprModel` (`rs-bpr`) |
| Mới | `src/models/seen_item_masking.py` — mask item đã xem, sentinel hữu hạn `−1e9` |
| Sửa | `src/eval/model_registry.py` — đăng ký `rs-itemcf`, `rs-bpr` |
| Mới | `configs/best-rs-itemcf.yaml` (rating), `configs/best-rs-itemcf-binary-pos.yaml` (tốt nhất), `configs/best-rs-bpr.yaml` |
| Mới | `configs/rs-*.yaml` — 33 config sweep (12 Item-CF + 21 BPR) |
| Mới | `tests/test_phase2_rs_models.py` — 19 test |

### Kết quả sweep trên `dev` (20.000 user) — 33 cấu hình

**Item-CF** (k × weight_mode), NDCG@10:

| weight_mode | k=50 | k=100 | k=200 | k=500 |
|---|---|---|---|---|
| `binary_pos` | 0,1263 | 0,1282 | **0,1297** | 0,1280 |
| `rating` | 0,1201 | 0,1215 | 0,1235 | 0,1224 |
| `centered` | 0,1207 | 0,1187 | 0,1126 | 0,1056 |

**Kết quả đảo ngược giả định của plan.** Plan chốt sẵn `weight_mode="rating"` (đặc trưng của
bản gốc RS) — nhưng đo thực tế cho thấy **`binary_pos` tốt hơn `rating` 5,1%** (0,1297 vs
0,1235). Lý do có lý: rating thấp (1–3 sao) là thông tin *về mức độ thích*, nhưng bản thân
item-item cosine không phân biệt được "user thích phim này 2 sao" với "user coi phim này là
tệ" — trọng số rating chỉ làm nhiễu tín hiệu cộng, và phim bị chấm 1 sao vẫn kéo profile
user về phía các phim tệ.

**Điểm kiểm chứng mạnh nhất của phase này:** `rs-itemcf` với `binary_pos` cho NDCG@10 =
**0,129678 — khớp tuyệt đối** với `cf-itemknn` của CB (0,129678), HR@10 = 0,48270 = 0,48270.
Hai mô hình chạy độc lập, khác repo, khác code, chỉ dùng chung catalog + evaluator → khớp
tuyệt đối chứng minh chúng là **cùng một thuật toán**. Test `test_rs_itemcf_binary_weight_mode_matches_cb_itemknn`
chốt lại điều này ở mức unit test.

**BPR** (20 cấu hình), các kết luận:

| trục | kết quả |
|---|---|
| `learning_rate` | 0,1 → 0,0697 · 0,05 → 0,0809 · 0,03 → 0,1012 · 0,02 → 0,1071 · **0,01 → 0,1106** · 0,008 → 0,1101 |
| `regularization` | 0,1 → 0,0364 · 0,01 → 0,0814 · 0,001 → 0,0899 · 0,0001 → 0,1103 · **0,00001 → 0,1106** |
| `iterations` | 10 → **0,1106** · 20 → 0,1103 · 30 → 0,1056 · 100 → 0,0776 |
| `factors` | 64 → 0,1106 · 128 → 0,1076 (không cải thiện) |
| `verify_negative_samples` | `true` → 0,1075 · `false` → 0,0778 (tắt làm mô hình tệ hơn rõ rệt) |

**Ba kết luận đảo ngược tham số của bản gốc RS:**

1. **`learning_rate` phải nhỏ.** Bản gốc dùng 0,05 (chạy trên catalog 62.423 phim). Ở catalog
   13.816 phim với 10,07M tương tác relevant, 0,05 là quá lớn → 0,0809.
2. **`regularization` phải nhỏ.** 0,1 cho 0,0364 — **tệ hơn Random (0,0011)**. Biến thiên
   này từng có trong `configs/` và phải được ghi rõ là điểm âm, không phải baseline hợp lệ.
3. **Nhiều epoch không tốt hơn.** 100 epoch có `train_auc` 93,9% (cao hơn 92,2% của 10 epoch)
   nhưng NDCG@10 **thấp hơn 30%** (0,0776 vs 0,1106) → overfit rõ rệt. Đây là ví dụ điển
   hình cho thấy `train_auc` của `implicit` không phải tiêu chí chọn mô hình.

### Kết quả cuối (config chốt)

| Config | NDCG@10 | HR@10 | P@10 | fit | score |
|---|---|---|---|---|---|
| `best-rs-itemcf-binary-pos.yaml` | **0,1297** | 0,4827 | 0,0966 | 0,35s | 4,24s |
| `best-rs-itemcf.yaml` (rating) | 0,1235 | 0,4694 | 0,0922 | 0,30s | 5,22s |
| `best-rs-bpr.yaml` | **0,1106** | 0,4606 | 0,0828 | 40,87s | 2,77s |

### Vị trí trong bảng CF (dev, NDCG@10)

| Model | NDCG@10 | Nguồn |
|---|---|---|
| ALS (f=64, α=5, reg=0,1) | 0,1383 | CB |
| ItemKNN-CF (k=200) | 0,1297 | CB |
| **Item-CF (binary_pos, k=200)** | **0,1297** | RS — khớp tuyệt đối với dòng trên |
| **BPR (f=64, i=10, lr=0,01, reg=1e-5)** | **0,1106** | RS |
| MostPop | 0,0723 | CB |
| Random | 0,0011 | CB |

BPR thua ItemKNN-CF 14,6% và thua ALS 20,0%. Đây là kết quả **có lý**: BPR ở đây dùng ma
trận nhị phân relevant và SGD trên triplet, còn ItemKNN-CF dùng được trọng số cộng dồn
của cả tập user trong một phép nhân — với catalog nhỏ và user lịch sử dài, phép nhân
Rocchio có tín hiệu mạnh hơn. BPR vẫn được giữ trong bảng vì nó **tối ưu trực tiếp hàm
ranking** (khác ALS tối ưu sai số hồi quy) — đây là luận điểm để thảo luận trong báo cáo,
không phải lý do để tuyên bố BPR mạnh.

### Ba khác biệt giữa `rs-itemcf` và `cf-itemknn` của CB — phải nêu khi đọc bảng

Cùng thuật toán (item-item cosine), cùng hướng `W @ Kᵀ`, khác ở ba điểm:

1. **Trọng số phía user.** `rs-itemcf` dùng `weight_mode="rating"` (rating thật 1–5),
   `cf-itemknn` dùng `"binary_pos"`. Kiểm chứng bằng test: cùng `weight_mode="binary_pos"`
   thì hai model cho **cùng điểm số** (chứng minh phần còn lại là giống nhau thật).
2. **Hướng tính.** Bản gốc RS tính `W @ K` (lấy hàng của item user đã xem); bản port dùng
   `W @ Kᵀ` (lấy hàng của item ứng viên). K bất đối xứng vì top-k cắt theo hàng.
3. **Catalog.** Bản gốc tự dựng `movie_map` từ df rating dương nên lệch 13.816 phim.

### BPR: `implicit` **không reproducible khi đa luồng**

Đo thực nghiệm trên toy data — hai lần chạy cùng `random_state=42`:

| `num_threads` | Hai lần chạy cho cùng kết quả? |
|---|---|
| 1 | ✅ có |
| 2 | ❌ không |
| 4 | ❌ không |

Nguyên nhân: `implicit` lấy mẫu âm bằng RNG dùng chung giữa các luồng → race. Vì vậy
`RsBprModel` đặt **`num_threads=1` mặc định**, và đổi lại 6× thời gian fit. Số liệu trong
báo cáo phải chạy lại được sau này — đây là điều kiện bắt buộc của một pipeline nghiên cứu,
không phải tối ưu tốc độ. Test `test_bpr_default_is_single_threaded_for_reproducibility`
chốt lại điều này.

### Chi phí BPR đo thật (12 core, catalog 13.816 phim, 10,07M tương tác relevant)

| Cấu hình | fit | tái lập được |
|---|---|---|
| 10 epoch, 12 luồng | ~7s | không |
| **10 epoch, 1 luồng (mặc định)** | **41s** | **có** |
| 30 epoch, 1 luồng | 121s | có |
| 100 epoch, 12 luồng | 65s | không |

Rẻ hơn nhiều so với dự đoán "hàng giờ" trong plan — rủi ro "BPR chạy quá lâu" của Phase 02
**không xảy ra**, kể cả ở `full`.

### Sentinel mask item đã xem = `−1e9` (không phải `−∞`)

`minmax_row` (dùng cho hybrid) ném `ValueError` khi thấy điểm không hữu hạn, và `implicit`
không chấp nhận `−∞` trong ma trận trọng số. `−1e9` đủ âm để không bao giờ lọt vào top-k,
vẫn hữu hạn nên phép cộng/minmax chạy được.

### Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Test mới | 19 test trong `tests/test_phase2_rs_models.py` |
| Toàn bộ `tests/` | **189 passed** (170 trước đó + 19 mới) |
| Catalog trong log | **13.816 item** cho cả hai model ✓ (không phải số item từ df rating dương) |
| Khớp tuyệt đối | `rs-itemcf` binary_pos ≡ `cf-itemknn` của CB (NDCG@10 0,129678, HR@10 0,48270) |

### Lưu ý

Chưa commit. Không `git add`/`git commit`/`git push` khi chưa được user cho phép.

---

## 2026-11-14 — Phase 03: Chạy bảng so sánh chung CF/CB

**Phạm vi:** 8 mô hình × 3 giao thức (dev / full / cold-start item). Không đổi thuật toán.

### Thay đổi

| Nhóm | Nội dung |
|---|---|
| Mới | `src/data/catalog_identity.py` — `catalog_id` (mã băm 12 ký tự của tập `movieId` đã sắp xếp + số phim) và `split_id` (`temporal80_20` / `temporal80_20+coldstart_item`) |
| Mới | `src/eval/benchmark_tables.py` — dựng bảng CF (5 dòng) / CB (3 dòng) / cold-start (8 dòng) + Wilcoxon **trong từng bảng** |
| Mới | `scripts/run_shared_benchmark.py` — chạy 8 config × 3 giao thức, `--als-sweep` cho sweep ALS |
| Mới | `tests/test_phase3_benchmark_tables.py` — 11 test |
| Sửa | `src/eval/batched_scoring_evaluator.py` — metrics CSV thêm cột `split_id`, `catalog_id`, `seed` |
| Sửa | `src/eval/run_experiment_from_config.py` — truyền `split_id`/`catalog_id` xuống evaluator |
| Sửa | `src/eval/analyse_tag_leakage.py` — tham số `--protocol dev\|full`; cập nhật danh sách 8 mô hình vào bảng phân tầng |
| Sửa | `scripts/ablation_grids_fusion_and_cf.py` — thêm 3 cấu hình lưới ALS `reg × iterations` (111 config thay vì 108) |

### Vì sao thêm `split_id` / `catalog_id`

Dấu vết duy nhất cho biết một dòng kết quả thuộc giao thức nào là **tiền tố trong tên
file** — mà đổi tên file là mất dấu vết. Nay mỗi dòng `{exp}_metrics.csv` mang định danh
catalog (mã băm tập `movieId`) và split. `benchmark_tables` cũng **raise** nếu cột `model`
trong kết quả không khớp mô hình mà bảng khai báo — trường hợp "config bị sửa sau khi
chạy" trước đây sẽ gộp im lặng hai thuật toán khác nhau.

### Kết quả `full` (158.774 user) — bảng chính

| Bảng CF | NDCG@10 | | Bảng CB | NDCG@10 |
|---|---|---|---|---|
| ALS | **0,1375** | | CB-2 BM25 | **0,0998** |
| Item-CF (RS) | 0,1290 | | CB-1 TF-IDF | 0,0937 |
| BPR (RS) | 0,1099 | | CB-3 Tag Genome | 0,0713 |
| MostPop | 0,0739 | | | |
| Random (sanity) | 0,0012 | | | |

Số `dev` và `full` lệch nhau ít: ALS 0,1383 → 0,1375; Item-CF 0,1297 → 0,1290;
BPR 0,1106 → 0,1099. Không có hiện tượng `full` "nhảy" lên — mẫu 20.000 user đại diện tốt.

### Cân bằng số config — đính chính (số liệu đã kiểm chứng lại)

Bản ghi trước đây ghi "sweep ALS trên `dev` cho biên độ **0,40%** — cùng cỡ biên độ mà CB
đạt sau 103 config, nên kết luận 'CF thắng CB' không phải hậu quả của tinh chỉnh thiếu
công bằng". **Cả hai vế của câu đó đều sai.** Số đo lại trên toàn bộ kết quả `dev`:

| Tuyên bố cũ | Thực tế |
|---|---|
| ALS biên độ 0,40% | Chỉ đúng cho lát cắt `reg` × `iterations` (8 config, f=64, α=5). **Toàn lưới 15 config biên độ 17,94%** (0,1135 → 0,1383) |
| ALS "không nhạy" | ALS **rất nhạy với `α`**: α=40 → 0,1135 (−18% so với α=5 = 0,1379). `factors` cũng ảnh hưởng (f=128 tệ hơn f=64 ở mọi α) |
| CB nhận 103 config, CF chỉ 18 | **Đảo ngược.** Trong repo này CF đã chạy **49 config**, CB mới chạy **8** (85 config CB chưa chạy — trạng thái trước sweep) |
| "CF thắng không bị coi là thiên lệch" | Sai. CF được quét gấp 6 lần CB → kết luận "CF thắng" đang **thiên về CF**, cần sweep CB đủ mới gọi là công bằng |

**Chi tiết lát cắt `reg` × `iterations`** (f=64, α=5), biên độ 0,40%:

| reg \ it | 15 | 30 |
|---|---|---|
| 0,001 | 0,137718 | 0,137845 |
| 0,01 | 0,137905 | 0,137846 |
| 0,1 | **0,138270** | 0,137976 |
| 1,0 | 0,137822 | 0,137824 |

**Toàn lưới ALS đã chạy ở `dev`** (15 config) — biên độ **17,94%**:

| factors | α=1 | α=5 | α=10 | α=40 |
|---|---|---|---|---|
| 64 | 0,1371 | **0,1383** | 0,1333 | 0,1135 |
| 128 | 0,1291 | 0,1331 | 0,1302 | 0,1154 |

**Số config sweep đã chạy thực tế** (đếm từ `results/*_metrics.csv` không tiền tố giao
thức, đối chiếu `configs/`):

| Nhóm | Config trong `configs/` | Đã chạy | Chưa chạy |
|---|---|---|---|
| CF (ALS + Item-CF + BPR) | 51 | **49** | 2 |
| CB (CB-1…CB-4) | 93 | **8** | **85** |

**Hệ quả đã hành động**: bảng CF và bảng CB giữ **riêng**, Wilcoxon chỉ chạy **trong từng
bảng** (không so chéo CF↔CB), và mọi tài liệu nêu rõ kết luận "CF thắng ở giao thức
thường" là kết luận **thận trọng chứ chưa cân bằng**.

Bằng chứng bổ sung (không đưa vào bảng chung vì là số của repo khác): sweep đầy đủ 93
config của `recommendation-cb` trên chính protocol này cho CB tối đa **0,1067** (CB-4 late
fusion + popularity prior), vẫn thấp hơn ALS 0,1375. Biên độ toàn sweep CB là **135,07%**
(0,0030 → 0,1067); trong 18 config tốt nhất biên độ còn **8,85%**. Tức "CB đã hội tụ ở
biên độ 0,4%" là sai — con số 0,4% chỉ là biên độ của 3 config ALS tốt nhất (0,29%).

### Kết quả cold-start item — đảo ngược hoàn toàn

1.381 phim (10%, chọn ngẫu nhiên theo 5 nhóm năm phát hành) bị giữ lại: xoá mọi tương
tác train, test chỉ giữ rating trên chúng. 10.992 user đánh giá được.

| Mô hình | NDCG@10 | Coverage@10 |
|---|---|---|
| CB-1 TF-IDF | 0,0328 | 0,1487 |
| CB-2 BM25 | 0,0306 | 0,1327 |
| CB-3 Tag Genome | 0,0305 | 0,2997 |
| ALS | **0,0000** | 0,1068 |
| Item-CF (RS) | **0,0000** | 0,1153 |
| BPR (RS) | **0,0000** | 0,1399 |
| MostPop | **0,0000** | 0,0140 |

Mô hình CF cho điểm **đúng bằng 0**, không phải "thấp": phim held-out không có tương tác
train nào nên mô hình lọc cộng tác không có gì để học. Đây là luận điểm trung tâm.

### Bảng phân tầng theo kỷ nguyên — thứ hạng không đảo chiều

`results/final/tag-leakage-strata.csv`, NDCG@10 theo thời điểm kết thúc train của user:

| Mô hình | < 2006 | 2006–2012 | ≥ 2012 |
|---|---|---|---|
| CB-1 TF-IDF (tag) | 0,0707 | 0,1256 | 0,1083 |
| CB-2 BM25 | 0,0824 | 0,1263 | 0,1094 |
| CB-3 Genome | 0,0516 | 0,0836 | 0,0920 |
| ALS | 0,1283 | 0,1507 | 0,1430 |
| Item-CF (RS) | 0,1299 | 0,1379 | 0,1229 |
| BPR (RS) | 0,1038 | 0,1186 | 0,1137 |
| MostPop | 0,0640 | 0,1046 | 0,0708 |
| Chỉ genres (không leak) | 0,0026 | 0,0041 | 0,0060 |

Thứ hạng CB-2 ≥ CB-1 > CB-3 giữ nguyên ở cả ba tầng; CF không bao giờ bị CB vượt ở tầng
nào. CB-1 dẫn CB-3 **1,18x–1,50x** ở mọi tầng — phần lớn lợi thế của tag là do tính chất
nội dung chứ không phải do rò rỉ thời gian.

### Coverage đảo ngược với độ chính xác

`results/beyond-accuracy-full.csv`: ALS chính xác nhất (NDCG@10 0,1375) nhưng chỉ chạm
**14,94%** catalog; CB-3 kém nhất trong CB (0,0713) nhưng chạm **67,20%**. Gini đi cùng
hướng ngược (CB-3 0,9407 < ALS 0,9732) — CF dồn suất vào ít phim hơn, CB trải rộng hơn.

### Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Test mới | 11 test trong `tests/test_phase3_benchmark_tables.py` |
| Toàn bộ `tests/` | **200 passed** (189 trước đó + 11 mới) |
| Tái lập được | Chạy lại Item-CF `full` 2 lần → NDCG@10 **0,1290143** cả hai lần (bitwise) |
| Đối chiếu CB cũ | CB-2 BM25: CB cũ 0,0997608 vs mới 0,0998 · ALS: 0,1374754 vs 0,1375 · CB-3: 0,0712931 vs 0,0713 — khớp 4 chữ số |
| Wilcoxon (bảng CF, mốc ALS) | ALS > Item-CF (p = 2,9e-182), ALS > BPR (p < 1e-300) — mọi cặp đều có ý nghĩa |
| Thời gian | `full` 8 model ≈ 9,5 phút · `coldstart` ≈ 3,7 phút · sweep ALS `dev` ≈ 4,6 phút |

### So sánh với bảng CB cũ — vì sao lệch (lệch rất ít)

Bảng `full` của CB cũ (`recommendation-cb/results/final/full-summary.csv`) và bảng mới
**khớp 4 chữ số** ở mọi mô hình dùng chung (Random, MostPop, CB-1, CB-2, CB-3, ALS).
Khác biệt nằm ở:

- **Item-CF**: bản RS với `weight_mode="rating"` → 0,1235; bản CB với `binary_pos` → 0,1290.
  Bảng chung lấy bản RS **binary_pos** (khớp tuyệt đối với CB) và ghi rõ trong README.
- **BPR**: không có ở CB.
- **Số user**: giống nhau (158.774).

### Lưu ý

Chưa commit. Không `git add`/`git commit`/`git push` khi chưa được user cho phép.

---

## 2026-11-14 — Đính chính tài liệu + sửa `build_final_tables.py`

**Phạm vi:** `README.md`, `docs/project-changelog.md`, `plans/.../phase-03-*.md`,
`plans/.../plan.md`, `scripts/build_final_tables.py`, `src/eval/benchmark_tables.py`,
`src/eval/make_result_figures.py`, `configs/`, `tests/`. **Không đổi thuật toán, không
đổi số liệu báo cáo.**

### Vì sao

Audit phát hiện ba lỗi trong chính tài liệu của phase 03 và một script nguy hiểm:

1. **"ALS không nhạy, biên độ 0,40%"** — con số đó chỉ đúng cho lát cắt `reg × iterations`
   (8 config, `f=64, α=5`). Toàn lưới đã chạy (15 config, thêm `α` và `factors`) cho biên
   độ **17,94%**. `α` là tham số gây sụp lớn nhất: α=40 → 0,1135 (−18% so với α=5).
2. **"CB nhận 103 config, CF chỉ 18"** — đảo ngược. Đó là số của repo `recommendation-cb`.
   Trong repo này CF đã chạy **49 config**, CB mới **8** (85 config CB chưa chạy — trạng
   thái trước sweep).
3. **Kết luận "CF thắng CB không bị coi là thiên lệch"** — suy luận sai: CF được quét
   gấp 6 lần CB nên kết luận đó **thiên về CF**.
4. **`scripts/build_final_tables.py`** crash với `FileNotFoundError` khi thiếu
   `results/dev-summary.csv` và ghi đè `results/final/benchmark-*.csv` — xoá mất bằng
   chứng protocol (`split_id`, `catalog_id`) mà phase 03 tạo ra.

### Thay đổi

| Nhóm | Nội dung |
|---|---|
| Sửa | `README.md` — mục "Cảnh báo cân bằng sweep — CF vs CB" (49 vs 8) và "Độ nhạy tham số của ALS"; kết luận "CF thắng" viết lại thành *thận trọng, chưa cân bằng* |
| Sửa | `docs/project-changelog.md` — mục "Cân bằng số config — đính chính" với bảng biên độ 0,40% vs 17,94% và bảng số config thực tế |
| Sửa | `plans/.../phase-03-run-shared-benchmark-tables.md` + `plan.md` — đính chính cùng nội dung; ghi rõ sweep CB 85 config **đã chốt tạm dừng** |
| Viết lại | `scripts/build_final_tables.py` — lấy nguồn từ `src.eval.benchmark_tables` (giữ `split_id`/`catalog_id`/`model`), **không ghi đè** `benchmark-*`/`wilcoxon-*`, mọi lỗi thiếu dữ liệu thành `_Missing` thay vì crash |
| Sửa | `src/eval/benchmark_tables.py` — thêm cột `model` vào bảng; sửa docstring về số config |
| Sửa | `src/eval/make_result_figures.py` — `_family()` nhận cả nhãn hiển thị, khoá registry và tên thí nghiệm; ghép bảng theo `exp_name` (không theo nhãn, tránh nhân bản dòng) |
| Mới | `configs/rs-bpr-f64-i30.yaml` — bổ sung config cho kết quả `rs-bpr-f64-i30` đã đo (trước đó là "kết quả mồ côi") |
| Xoá | `configs/rs-itemcf-k200.yaml`, `results/rs-itemcf-k200_{metrics.csv,peruser.npz}` — bản trùng của `rs-itemcf-k200-rating` (metric per-user giống hệt) |
| Mới | `tests/test_phase3_docs_and_final_tables.py` — 16 test chặn hồi quy tài liệu + hành vi script |
| Sửa | `pyproject.toml` — đăng ký marker `slow` |

### Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Toàn bộ `tests/` | **216 passed** (200 trước đó + 16 mới) |
| `build_final_tables.py` | chạy end-to-end không crash; `benchmark-cf.csv` **không đổi** (so byte) |
| Kết quả mồ côi | 0 (trước: 2) |
| `make_result_figures.py` | sinh đủ 5 hình, màu đúng (CF đỏ / CB xanh / baseline xám) |

### Ghi chú về sweep CB (trạng thái tại thời điểm ghi entry này)

Sweep 85 config CB **chưa chạy** (user chốt tạm dừng để trao đổi). Bằng chứng từ repo
`recommendation-cb` (93 config, cùng protocol) cho CB tối đa **0,1067** — thấp hơn ALS
0,1375 — nên hướng "CF thắng ở giao thức thường" có cơ sở, nhưng chưa được chứng minh
**công bằng** cho tới khi sweep CB chạy đủ.

> **Cập nhật 2026-11-14 (sau sweep):** đã chạy đủ 85/85 config CB. Xem entry
> "Chạy nốt sweep CB (85 config) + đưa CB-4 vào bảng chính" ở đầu file.

### Lưu ý

Chưa commit. Không `git add`/`git commit`/`git push` khi chưa được user cho phép.
