# Hướng dẫn chạy giao thức `full` (thay cho `dev`)

Tài liệu này dành cho việc **chạy lại toàn bộ thí nghiệm trên 158.774 user** thay vì
20.000 user mẫu của `dev`.

Mặc định mọi config trong `configs/` đều đặt `users: dev` — đó là chủ ý: **tham số phải
được chọn trên `dev`**, rồi mới chạy `full` **một lần** với config đã khoá. Chạy sweep
toàn bộ trên `full` là một việc khác hẳn, tốn nhiều giờ hơn và sinh ra bảng số khác
đáng kể. Tài liệu này nói rõ cả hai cách.

Đọc kèm: [`README.md`](../README.md) (đường chạy chính),
[`codebase-summary.md`](codebase-summary.md) (cấu trúc repo),
[`../notes/report-structure.md`](../notes/report-structure.md) (đề mục báo cáo).

---

## 1. Tại sao "thay cho dev" là câu hỏi dễ hiểu sai

`dev` và `full` **không chỉ khác ở số user** — chúng là hai giao thức khác nhau về mặt
pháp lý trong báo cáo:

| | `dev` | `full` |
|---|---|---|
| Số user đánh giá | 20.000 | 158.774 |
| Vai trò | **chọn tham số** (sweep 144 config) | **bảng chính** (chạy 1 lần sau khi khoá config) |
| Dùng để lập luận | biên độ sweep, đối chiếu độ nhạy tham số | mọi bảng so sánh, kiểm định thống kê |
| `split_id` | `temporal80_20` | `temporal80_20` (giống nhau) |
| `catalog_id` | `ml25m-n13816-de09a1b710f6` | (giống nhau) |

Cùng `split_id` + `catalog_id` nghĩa là hai giao thức **chấm trên cùng tập ứng viên**
(13.816 phim) và cùng cách chia — chỉ khác ở *ai được chấm điểm*. `dev` là tập con của
`full`.

**Hệ quả thực tế:** NDCG@10 lệch giữa hai giao thức rất nhỏ (ALS 0,1383 → 0,1375,
chênh −0,6%), và **thứ hạng khớp tuyệt đối ở cả 9 mô hình** — không cặp nào đảo. Nên
kết luận từ `dev` **có đại diện** cho `full` — nhưng bảng chính trong báo cáo vẫn phải
lấy từ `full`, vì `full` mới có đủ 158.774 user cho kiểm định bootstrap và Wilcoxon.

> **Số liệu tham khảo (đã đo, `results/final/`):**
> | Mô hình | `dev` | `full` | Lệch |
> |---|---:|---:|---:|
> | ALS | 0,1383 | 0,1375 | −0,6% |
> | Item-CF (RS) | 0,1297 | 0,1290 | −0,5% |
> | BPR (RS) | 0,1106 | 0,1099 | −0,6% |
> | CB-4 Hybrid | 0,1043 | 0,1054 | +1,1% |
> | CB-2 BM25 | 0,0986 | 0,0998 | +1,1% |
> | CB-1 TF-IDF | 0,0921 | 0,0937 | +1,7% |
> | MostPop | 0,0723 | 0,0739 | +2,2% |
> | CB-3 Tag Genome | 0,0703 | 0,0713 | +1,3% |
> | Random (sanity) | 0,0011 | 0,0012 | +7,6% |
>
> Nguồn: `benchmark-{cf,cb}-dev.csv`, `benchmark-{cf,cb}.csv`.

---

## 2. Trước khi bắt đầu

### 2.1. Kiểm tra môi trường

```bash
cd /home/huyquang/Workspace/Personal/PyCharmProjects/RecommendationSystem
venv/bin/python -c "import implicit, scipy, pandas, numpy; print('OK')"
```

### 2.2. Kiểm tra dữ liệu đã tiền xử lý

`full` dùng **chung** `train.parquet` / `test.parquet` với `dev` — không cần split riêng.

```bash
venv/bin/python - <<'PY'
import pandas as pd, numpy as np
tr = pd.read_parquet('data/processed/train.parquet')
te = pd.read_parquet('data/processed/test.parquet')
print('train rows :', f'{len(tr):,}')
print('test rows  :', f'{len(te):,}')
print('users      :', f'{tr.user_idx.nunique():,}')
print('dev users  :', f'{len(np.load("data/processed/dev_users.npy")):,}')
PY
```

Kỳ vọng: `train` 19.801.784 dòng, `test` 4.831.397 dòng, 162.540 user,
`dev_users.npy` có 20.000 phần tử.

### 2.3. Chỗ trống đĩa

Artifact của `full` **lớn hơn nhiều** lần vì ma trận top-k lưu theo số user:

| Nhóm artifact | `dev` (20k user) | `full` (158k user) |
|---|---:|---:|
| `{exp}_metrics.csv` | ~0,8 KB | ~0,8 KB |
| `{exp}_peruser.npz` | ~0,27 MB | ~1,9 MB |
| `{exp}_topk.npy` | ~4,0 MB | ~31,8 MB |

> `topk.npy` là mảng `[n_eval_users, 50]` kiểu `int32` — `158.774 × 50 × 4 byte ≈ 30 MB`,
> khớp với 31,8 MB đo được. **Chạy sweep `full` cho 144 config sẽ sinh khoảng 5 GB
> artifact** (`144 × ~33,7 MB`), cộng kết quả `dev` hiện có (~0,5 GB).

Kiểm tra trước:

```bash
df -h . | tail -1
```

Cần **≥ 10 GB trống** nếu định chạy cả sweep, chỉ cần **~1 GB** nếu chỉ chạy 9 config
bảng chính (xem §3.1).

### 2.4. Bảo toàn kết quả `dev`

Kết quả `dev` là **nguồn sự thật** cho bảng sweep trong báo cáo
(`results/final/ablation-*.csv`). Xoá nhầm là mất bằng chứng.

```bash
# Sao lưu trước khi thử nghiệm
cp -r results/final results/final.bak-dev
```

> **Không** xoá `results/*_metrics.csv` không có tiền tố — đó là kết quả `dev`.

---

## 3. Cách A — Chạy `full` cho 9 mô hình bảng chính (khuyến nghị)

Đây là cách **đúng về mặt phương pháp** và là cách repo đã dùng để sinh ra
`results/final/benchmark-{cf,cb}.csv`.

### 3.1. Chạy thí nghiệm

```bash
venv/bin/python scripts/run_shared_benchmark.py --protocol full
```

Lệnh này chạy 9 config đã khoá tham số trên 158.774 user, mỗi lần bật `--save-topk`
(cần cho coverage / novelty / ILD).

**Thời gian ước tính (đo thực tế từ `fit_s` + `score_s`):**

| Mô hình | `fit` | `score` | Tổng |
|---|---:|---:|---:|
| ALS | 15,2s | 24,9s | 40,1s |
| Item-CF (RS) | 0,4s | 34,9s | 35,3s |
| BPR (RS) | 39,8s | 21,7s | 61,5s |
| CB-4 Hybrid | 0,8s | 140,2s | 141,0s |
| CB-2 BM25 | 0,6s | 115,1s | 115,8s |
| CB-1 TF-IDF | 2,0s | 105,5s | 107,5s |
| CB-3 Tag Genome | 0,3s | 28,2s | 28,5s |
| MostPop | 0,1s | 20,4s | 20,4s |
| Random | 0,0s | 23,2s | 23,2s |
| | | | **≈ 9,6 phút** |

BPR là mô hình chậm nhất về `fit` (~40s) vì `num_threads=1` — đây là **đánh đổi có chủ ý**
để kết quả tái lập được tuyệt đối (xem `README.md` §"Giới hạn cần nêu khi bảo vệ").

**Chạy lại config đã có kết quả:** thêm `--force`.

```bash
venv/bin/python scripts/run_shared_benchmark.py --protocol full --force
```

### 3.2. Sinh chỉ số ngoài độ chính xác

```bash
venv/bin/python -m src.eval.beyond_accuracy_metrics --prefix full_ \
    --output results/final/beyond-accuracy-full.csv
```

### 3.3. Gom bảng benchmark + Wilcoxon

```bash
venv/bin/python -m src.eval.benchmark_tables --protocol full
```

Sinh ra `results/final/benchmark-cf.csv`, `benchmark-cb.csv`, `wilcoxon-cf.csv`,
`wilcoxon-cb.csv`. Bảng này **ghi đè** bản cũ — cần thiết khi chạy lại từ đầu.

### 3.4. Bootstrap CI

```bash
venv/bin/python -m src.eval.statistical_tests --prefix full_ --metric NDCG@10
```

> Ghi ra `results/bootstrap-ci.csv` và `results/wilcoxon.csv` (ở **gốc** `results/`, không
> phải `results/final/`). `scripts/build_final_tables.py` ở §3.7 sẽ copy bootstrap vào
> `results/final/bootstrap-ci.csv`.

### 3.5. Bucket theo độ dài lịch sử user

```bash
venv/bin/python -m src.eval.user_bucket_analysis --prefix full_ --metric NDCG@10 \
    --output coldstart-user-buckets.csv
```

> Ghi ra `results/coldstart-user-buckets.csv` (gốc `results/`); `build_final_tables.py`
> copy vào `results/final/`.

### 3.6. Bảng phân tầng chống rò rỉ (3 kỷ nguyên)

```bash
venv/bin/python -m src.eval.analyse_tag_leakage --protocol full
```

### 3.7. Gom bảng rút gọn + sinh hình

```bash
venv/bin/python scripts/build_final_tables.py
venv/bin/python -m src.eval.make_result_figures
```

> `build_final_tables.py` **không** ghi đè `results/final/benchmark-*.csv` và
> `wilcoxon-*.csv` (đó là sản phẩm của §3.3, có bằng chứng protocol). Script này chỉ
> rút gọn thành `full-summary.csv` / `coldstart-item.csv` / `dev-summary.csv` và **bỏ
> qua phần thiếu dữ liệu thay vì crash**.

### 3.8. Kiểm chứng

```bash
venv/bin/python scripts/check_report_numbers.py
venv/bin/python -m pytest tests/
```

`check_report_numbers.py` phải báo **`✓ 0 lệch`** — nếu báo lệch, số trong
`docs/report/05-experimental-results.md` và `06-leakage-analysis-and-discussion.md` đã
lệch so với CSV mới, cần cập nhật hoặc rollback.

---

## 4. Cách B — Chạy *sweep* toàn bộ trên `full`

Dùng khi cần biết **độ nhạy tham số trên toàn bộ user** thay vì trên mẫu 20k, hoặc
khi muốn đối chiếu xem thứ hạng ở `dev` có phản ánh `full` hay không.

> ⚠️ **Không dùng kết quả sweep `full` để chọn tham số.** Chọn tham số trên tập mà
> sau đó dùng để báo cáo trên chính tập đó là **rò rỉ tham số** — sai về mặt phương
> pháp. Nếu cần sweep trên `full`, hãy dùng nó **chỉ để kiểm tra độ nhạy**, và giữ
> `dev` làm nơi chọn tham số chính thức.

### 4.1. Thời gian và chỗ trống

| Phạm vi | Thời gian ước tính | Chỗ trống |
|---|---:|---:|
| 9 config bảng chính | ≈ 10 phút | ~0,5 GB |
| 144 config sweep | **≈ 3,1 giờ** | **~5 GB** |

Ước tính `full` = `sum(fit_s)` + `158.774/20.000 × sum(score_s)`:
`1.988s + 7,94 × 1.171s ≈ 11.284s ≈ 3,13 giờ`.
`fit` không scale theo số user (chỉ phụ thuộc ma trận train), `score` thì có.

### 4.2. Cách chạy

Mỗi config sweep đều có `users: dev` trong YAML. Để chạy ở `full`, dùng cờ
`--users full`:

```bash
# Chạy TỪNG config một — an toàn, dễ dừng giữa chừng
venv/bin/python -m src.eval.run_experiment_from_config configs/cf-als-f064-a005.yaml --users full
```

Hoặc chạy cả lưới qua runner (tự sinh config trước khi chạy):

```bash
venv/bin/python scripts/run_all_experiments.py --force
```

> ⚠️ `run_all_experiments.py` **không** có cờ `--users`. Muốn chạy sweep ở `full` thì
> phải gọi `run_experiment_from_config --users full` cho từng config, **không** dùng
> runner này.

### 4.3. ⚠️ Bẫy: bảng ablation sẽ rỗng nếu bạn xoá kết quả `dev`

`src/eval/collect_results_to_table.py` lọc bảng ablation theo **tiền tố tên thí nghiệm**:

```python
ABLATION_PREFIXES = {"cb1": ("cb1-",), ..., "cf": ("cf-", "rs-")}
df["exp_name"].str.startswith(prefixes)
```

Khi sweep chạy ở `full`, tên kết quả là `full_cf-als-...` — **không** bắt đầu bằng
`cf-`, nên:

| Tình huống | `ablation-cf.csv` sinh ra |
|---|---|
| Chỉ có kết quả `dev` (hiện tại) | 51 dòng ✅ |
| Chỉ có kết quả `full` (xoá sạch `dev`) | **0 dòng** ❌ |
| Có cả hai | 51 dòng `dev` (phần `full` bị bỏ qua) ⚠️ |

**Cách an toàn:** giữ nguyên kết quả `dev` và để sweep `full` ghi vào tên tiền tố
`full_` (đã tự động nhờ `experiment_name()` trong
`src/eval/run_experiment_from_config.py`). **Đừng xoá `results/*_metrics.csv` không
tiền tố.**

Nếu bạn *buộc phải* có bảng ablation trên `full`, cần sửa filter ở
`src/eval/collect_results_to_table.py` để so trên tên đã bóc tiền tố:

```python
df["base"] = [strip_protocol_prefix(n) for n in df["exp_name"]]
df = df[df["base"].str.startswith(prefixes)]
```

(Đây là thay đổi trong *mã*, không phải trong báo cáo — cần test hồi quy đi kèm.)

### 4.4. Tên kết quả sau khi chạy `full`

`experiment_name()` gắn tiền tố theo thứ tự cố định:

```
<split_dir>_ <users>_ <exp_name gốc>
```

| Giao thức | Cờ | Tên file kết quả |
|---|---|---|
| `dev` | (mặc định) | `cb2-01-k1-05-b000_metrics.csv` |
| `full` | `--users full` | `full_cb2-01-k1-05-b000_metrics.csv` |
| `coldstart` | `--split-dir data/processed/coldstart` | `coldstart_cb2-01-k1-05-b000_metrics.csv` |

→ Nhờ tiền tố, **chạy `full` không ghi đè kết quả `dev`**.

### 4.5. So sánh độ lệch `dev` ↔ `full` sau khi sweep

```bash
venv/bin/python - <<'PY'
import pandas as pd
from src.eval.collect_results_to_table import collect, strip_protocol_prefix

df = collect(sort_by=None)
df["base"] = [strip_protocol_prefix(n) for n in df["exp_name"]]
dev = df[df.exp_name == df.base].set_index("base")["NDCG@10"]
full = df[df.exp_name.str.startswith("full_")].set_index("base")["NDCG@10"]

cmp = pd.DataFrame({"dev": dev, "full": full}).dropna()
cmp["lệch %"] = (cmp["full"] - cmp["dev"]) / cmp["dev"] * 100
print(cmp.sort_values("lệch %").to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\nLệch tuyệt đối lớn nhất: {cmp['lệch %'].abs().max():.2f}%")
PY
```

> **Đọc kết quả:** lệch lớn nhất **không nằm ở mô hình chính** mà ở config yếu
> (`cb1-00-genres-binary-profile`, +12,1% — NDCG@10 chỉ 0,0036 nên nhiễu tương đối lớn).
> Với 9 mô hình bảng chính, lệch nằm trong khoảng **−0,7% → +2,2%** và thứ hạng
> **không đảo** (xem §1).

---

## 5. Giao thức `coldstart` (đi kèm, để tham khảo)

`coldstart` là giao thức thứ ba: giữ lại **1.381 phim** (10% catalog, phân tầng theo 5
nhóm năm phát hành), xoá mọi tương tác train của các phim đó.

```bash
# 1. Tạo split (chỉ làm 1 lần)
venv/bin/python -m src.eval.cold_start_item_split

# 2. Chạy 9 config
venv/bin/python scripts/run_shared_benchmark.py --protocol coldstart

# 3. Chỉ số ngoài độ chính xác + bảng
venv/bin/python -m src.eval.beyond_accuracy_metrics --prefix coldstart_ \
    --output results/final/beyond-accuracy-coldstart.csv
venv/bin/python -m src.eval.benchmark_tables --protocol coldstart
```

**Lưu ý:** `coldstart` dùng **`users: dev`** (20.000 user) chứ không phải `full` —
vì tập user hợp lệ của split cold-start là tập con, và `load_dataset` tự lọc theo
`has_relevant`. `n_eval_users` trong `benchmark-coldstart.csv` là **10.992** (khác
`beyond-accuracy-coldstart.csv` là 10.988 vì 4 user không có tín hiệu bị loại).

---

## 6. Những chỗ dễ sai

| Sai lầm | Hệ quả | Cách đúng |
|---|---|---|
| Chạy sweep ở `full` rồi dùng để **chọn tham số** | Rò rỉ tham số — số báo cáo lạc quan | Chọn tham số ở `dev`; sweep `full` chỉ để kiểm tra độ nhạy |
| Xoá `results/*_metrics.csv` không tiền tố để "dọn cho gọn" | Mất toàn bộ bảng ablation `dev` (bằng chứng sweep) | Giữ lại; `full` tự có tiền tố riêng |
| Ghi đè `results/final/benchmark-*.csv` bằng bản tự rút gọn | Mất bằng chứng protocol | Chỉ `src.eval.benchmark_tables` được ghi đè; `build_final_tables.py` **không** ghi đè |
| Bỏ `--save-topk` khi chạy `full` | `beyond_accuracy_metrics` nổ vì thiếu top-k | Luôn kèm `--save-topk` |
| Chạy `full` mà không `--force` sau khi đổi config | Kết quả cũ được giữ lại âm thầm | `--force`, hoặc xoá `results/full_*_metrics.csv` |
| Đổi `users: full` trong YAML rồi chạy `run_all_experiments.py` | Config trong repo bị bẩn; `select_best_configs.py` đọc sai | Không sửa YAML; dùng cờ `--users full` |
| Bỏ qua cảnh báo `OpenBLAS is configured to use 12 threads` | Với BPR: thời gian không ổn định, số có thể lệch nhẹ giữa 2 lần chạy | `OPENBLAS_NUM_THREADS=1` (xem §6.1) |
| Tin bảng `benchmark-*-dev.csv` đã commit mà không kiểm tra lại | Bảng có thể **cũ hơn** code (thiếu dòng, thiếu cột) — xem §6.2 | Chạy lại `benchmark_tables --protocol dev` trước khi trích số |
| Chạy `full` rồi chạy `select_best_configs.py` | `best-*.yaml` bị ghi đè bằng config chọn trên `full` | `select_best_configs.py` đã lọc `protocol="dev"` — nhưng **đừng** chạy nó sau khi đổi kết quả dev |

> **Lưu ý cuối:** `scripts/select_best_configs.py` chọn best **trên `dev`** (`collect(protocol="dev")`)
> và ghi đè `configs/best-*.yaml`. Nếu bạn chạy sweep `full` rồi chạy script này, nó
> vẫn chọn trên `dev` — an toàn. Nhưng nếu bạn **xoá** kết quả `dev`, script sẽ
> không tìm thấy gì và **giữ nguyên** `best-*.yaml` cũ (không hỏng, nhưng cũng không
> cập nhật).

### 6.1. Cảnh báo `OpenBLAS is configured to use 12 threads`

Khi chạy ALS hoặc BPR, `implicit` in cảnh báo:

```
RuntimeWarning: OpenBLAS is configured to use 12 threads. It is highly recommended
to disable its internal threadpool by setting the environment variable
'OPENBLAS_NUM_THREADS=1' ...
```

**Đây là cảnh báo, không phải lỗi** — script vẫn chạy và vẫn ghi kết quả.

Repo đã xử lý **một nửa** vấn đề này:

| Mô hình | Xử lý BLAS trong code | Trạng thái |
|---|---|---|
| ALS (`src/models/als_cf_model.py`) | `_single_threaded_blas()` bọc `fit` bằng `threadpool_limits(1, "blas")` | ✅ Đã ép 1 luồng |
| BPR (`src/models/rs_bpr_model.py`) | **Không** ép — chỉ đặt `num_threads=1` cho chính `implicit` | ⚠️ BLAS vẫn có thể mở threadpool riêng |

Với BPR, threadpool OpenBLAS cạnh tranh với luồng chính nên thời gian chạy **không ổn
định** giữa các lần (khó so sánh `fit_s`), và thứ tự cộng dồn floating-point có thể
làm số lệch nhẹ giữa hai lần chạy.

Để có số tái lập được tuyệt đối (đúng như bảng chính trong báo cáo):

```bash
OPENBLAS_NUM_THREADS=1 venv/bin/python scripts/run_shared_benchmark.py --protocol full
```

> BPR đã cố ý đặt `num_threads=1` trong config vì lý do tương tự (RNG dùng chung giữa
> các luồng nên >1 luồng cho kết quả khác nhau — đo được `num_threads=2` đã cho hai lần
> chạy khác nhau). `OPENBLAS_NUM_THREADS=1` là lớp bảo vệ tương ứng cho phép toán tuyến
> tính, và cũng làm cảnh báo biến mất.
>
> Lưu ý: `threadpoolctl` là phụ thuộc **tùy chọn** — `als_cf_model.py` bắt
> `ImportError` và chạy tiếp nếu thiếu. Trong `venv/` hiện tại đã có sẵn
> (`threadpoolctl 3.7.0`), nên ALS đang được ép 1 luồng đúng như thiết kế.

### 6.2. Bảng `benchmark-*-dev.csv` đã commit có thể cũ hơn code

`results/final/benchmark-cf-dev.csv` và `benchmark-cb-dev.csv` được commit từ lâu. Nếu
`configs/best-*.yaml` thay đổi sau đó (ví dụ thêm config CB-4 content thuần), các bảng
này **không tự cập nhật** — chúng chỉ được sinh lại khi chạy
`benchmark_tables --protocol dev`.

**Dấu hiệu nhận biết:** số dòng của bảng `dev` không khớp số dòng của bảng `full`
(cùng bộ mô hình), hoặc bảng thiếu cột `model`.

**Cách kiểm tra (nhanh, không tốn thời gian chạy mô hình):**

```bash
venv/bin/python - <<'PY'
import pandas as pd
for suffix in ('cf', 'cb'):
    dev = pd.read_csv(f'results/final/benchmark-{suffix}-dev.csv')
    full = pd.read_csv(f'results/final/benchmark-{suffix}.csv')
    print(f'{suffix}: dev {len(dev)} dòng, full {len(full)} dòng',
          '-> OK' if len(dev) == len(full) else '-> LỆCH, cần sinh lại')
    print('   thiếu cột model?', 'model' not in dev.columns)
PY
```

**Cách sửa:**

```bash
venv/bin/python -m src.eval.benchmark_tables --protocol dev
```

Lệnh này chỉ **đọc** kết quả `dev` đã có trong `results/` (không chạy lại mô hình), nên
chạy trong vài giây. Giá trị của các dòng đã có **không đổi** — chỉ bổ sung dòng còn
thiếu và cột `model`.

> **Đã xảy ra thật trong repo này:** `benchmark-cb-dev.csv` bị commit khi chỉ có 3 dòng
> (thiếu CB-4 Hybrid), trong khi `benchmark-cb.csv` có 4 dòng. Báo cáo §5.2.3 trích số
> `dev` của CB-4 từ `dev-summary.csv` nên **không bị ảnh hưởng** — nhưng bảng CSV mà
> báo cáo dẫn nguồn thì thiếu dòng. Đã sinh lại và kiểm chứng: 3 dòng cũ giữ nguyên
> giá trị, chỉ thêm dòng CB-4; `check_report_numbers.py` vẫn báo **0 lệch**.

---

## 7. Checklist

- [ ] `data/processed/train.parquet` + `test.parquet` tồn tại
- [ ] `df -h .` còn ≥ 10 GB (sweep) hoặc ≥ 1 GB (chỉ 9 config)
- [ ] Đã sao lưu `results/final/` → `results/final.bak-dev`
- [ ] Chạy `run_shared_benchmark.py --protocol full` (hoặc cặp `--users full` cho sweep)
- [ ] Mọi lần chạy đều kèm `--save-topk`
- [ ] `beyond_accuracy_metrics` + `benchmark_tables` + `statistical_tests` đã chạy lại
- [ ] `build_final_tables.py` + `make_result_figures` đã chạy lại
- [ ] `scripts/check_report_numbers.py` báo **`✓ 0 lệch`**
- [ ] `pytest tests/` xanh
- [ ] Kết quả `dev` **còn nguyên** (bảng ablation 144 dòng)

---

## 8. Xem nhanh

```bash
# Bảng chính
column -s, -t < results/final/benchmark-cf.csv | cut -c1-120
column -s, -t < results/final/benchmark-cb.csv | cut -c1-120

# So dev vs full cho 9 mô hình bảng chính
venv/bin/python - <<'PY'
import pandas as pd
d = pd.read_csv('results/final/dev-summary.csv')[['Thuật toán', 'NDCG@10']]
f = pd.read_csv('results/final/full-summary.csv')[['Thuật toán', 'NDCG@10']]
m = d.merge(f, on='Thuật toán', suffixes=('_dev', '_full'))
m['lệch %'] = (m['NDCG@10_full'] - m['NDCG@10_dev']) / m['NDCG@10_dev'] * 100
print(m.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
PY
```
