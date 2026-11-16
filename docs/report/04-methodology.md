# Chương 4 — Phương pháp

> **Phục vụ RQ1–RQ3:** chương này mô tả **cách** 9 mô hình được huấn luyện và chấm
> điểm trên một giao thức thống nhất. Ba giao thức ở §4.5 là nền để so sánh được
> CF với content-based (RQ1), để đo đảo chiều ở cold-start (RQ2), và để kiểm tra
> tín hiệu content theo kỷ nguyên (RQ3).
>
> **Nguồn số liệu:** `results/final/full-summary.csv` (cột `fit_s`, `score_s`),
> `results/final/ablation-*.csv` (144 dòng sweep).

---

## 4.1 Thiết lập thực nghiệm

### 4.1.1 Môi trường

| Thành phần | Giá trị |
|---|---|
| Ngôn ngữ | Python 3.12 |
| Tính toán ma trận | `numpy` 2.5.2, `scipy` 1.18.1 |
| Phân rã ma trận / BPR | `implicit` 0.7.3 (CPU) |
| Xử lý dữ liệu | `pandas` 3.0.5, `pyarrow` |
| Vector hoá TF-IDF | `scikit-learn` 1.9.0 |
| Lạc quan hóa | `scipy.sparse.linalg` |
| Kết xuất hình | `matplotlib` 3.9.2, `seaborn` 0.13.2 |
| **Random seed** | **42** (`paths.SEED`) |

Toàn bộ thí nghiệm chạy trên CPU, không dùng GPU. Đây là quyết định có chi phí:
mô hình BPR bị giới hạn `num_threads=1` (§4.4.3) nên `fit` mất 39,84s thay vì ~7s —
đánh đổi lấy tính tái lập được tuyệt đối (§7.3).

### 4.1.2 Cấu trúc repo và cách tái lập

Cấu trúc được thiết kế để **một mô hình không thể vô tình nhìn thấy nhãn test**:

```
src/
├── data/        # tiền xử lý: nạp, dựng đặc trưng, chia tập
├── models/      # 9 mô hình + lớp cơ sở + tiện ích dùng chung
├── eval/        # evaluator, registry, bảng kết quả, phân tích
├── api/         # tầng serving (demo) — không import src.eval.*
└── paths.py     # đường dẫn, hằng số, ngưỡng relevant
scripts/         # CLI: sinh bảng kết quả, vẽ hình, đối chiếu số báo cáo
configs/         # 159 file YAML: 144 sweep + 13 best-* + 2 baseline-*
tests/           # 288+ test
```

Hướng dẫn chạy lại đầy đủ: [`docs/codebase-summary.md`](../codebase-summary.md).

### 4.1.3 Vì sao `TrainData` và `Dataset` phải tách

Đây là quyết định thiết kế quan trọng nhất của đồ án, và nó **không** phải chuyện
sắp xếp code cho đẹp.

```python
@dataclass
class TrainData:      # mô hình chỉ được thấy cái này
    train_csr: sparse.csr_matrix
    train_df: pd.DataFrame
    n_users: int
    n_items: int

@dataclass
class Dataset:        # chỉ evaluator được thấy cái này
    train: TrainData
    test_rel_csr: sparse.csr_matrix   # nhãn relevant
    eval_users: np.ndarray
```

Mọi mô hình đều nhận `fit(train: TrainData, item_features: dict)` — kiểu dữ liệu đó
**không chứa** nhãn test, nên mô hình về mặt cấu trúc không thể đọc nó. Lỗi "gõ nhầm
`train_csr` thành `test_rel_csr`" sẽ không biên dịch được chứ không chỉ chạy sai âm
thầm.

Vấn đề này không giả định: nó xảy ra thật trong quá trình phát triển đồ án. Trước
đây có hai evaluator song song với protocol khác nhau (global 70/30 và per-user 80/20),
dẫn tới kết quả không thể so sánh. Việc hợp nhất về một đường chạy — 9 mô hình ×
3 giao thức, một evaluator — là đóng góp thứ nhất của đồ án (§8.1).

---

## 4.2 Baseline

Hai baseline không phải để cạnh tranh mà để **kiểm tra evaluator**: nếu evaluator
đúng, Random phải cho metric ≈ 0 và MostPop phải cho một mức sàn hợp lý.

| Baseline | Cách chấm | Vai trò |
|---|---|---|
| **Random** | Xếp hạng ngẫu nhiên 13.816 phim | **Kiểm tra evaluator** — mọi metric phải ≈ 0 (NDCG@10 = 0,0012) |
| **MostPop** | Xếp hạng theo số lượt rating ≥ 4,0 trong `train` | Sàn sống của mọi mô hình (NDCG@10 = 0,0739) |

MostPop dùng **cùng ngưỡng relevant** (≥ 4,0) và **cùng tập train** như mọi mô hình
khác — không có ngoại lệ nào. Ở giao thức `full`, MostPop đạt NDCG@10 = 0,0739, tức
bằng 54% NDCG của ALS (0,1375). Con số này đặt ngữ cảnh: **một mô hình không học gì
ngoài độ phổ biến** vẫn đạt hơn một nửa mức của mô hình tốt nhất — hệ thống gợi ý
thực tế đang cạnh tranh với một heuristic rất đơn giản.

Ở giao thức cold-start, MostPop rơi về **0,0000** (§5.3) vì mọi phim held-out đều
có popularity bằng 0 trong tập train. Đây là tiền lệ cho kết luận ở Chương 5: mọi
mô hình dựa trên độ phổ biến cũng chết ở cold-start, không chỉ CF.

---

## 4.3 Mô hình Content-based

### 4.3.1 Kiến trúc chung: hồ sơ user trên không gian đặc trưng item

Cả bốn mô hình content trong đồ án dùng **cùng một kiến trúc**, chỉ khác ở cách dựng
ma trận đặc trưng `X` [n_items, D]:

```
X  (đặc trưng item, L2-normalised)
W  (trọng số lịch sử user)              P = normalize(W_batch @ X)   (hồ sơ user)
                                          S = X @ P.T               (điểm ứng viên)
```

Hai quyết định kỹ thuật trong kiến trúc này:

- **Hồ sơ tính theo lô, không precompute toàn bộ.** Với 158.774 user và D lớn, tính
  trước P cho mọi user sẽ tốn hàng chẻm bộ nhớ; tính theo lô 2.000 user giữ mức
  dùng RAM không đổi theo số user.
- **Ma trận `X` giữ dạng thưa nếu đầu vào thưa.** TF-IDF trên tag có mật độ ~0,14%;
  bản dense 13.816 × 20.000 chiếm 2,06 GB (peak 8,28 GB khi dựng) trong khi CSR chỉ
  ~6 MB. Phép nhân cũng nhanh hơn vì chi phí tỉ lệ với số phần tử khác 0.

### 4.3.2 Bốn mô hình và tham số quét

| Mô hình | Cấu hình | Số config sweep | Ghi chú |
|---|---|---:|---|
| **CB-1 TF-IDF** | TF-IDF trên tag document + genres | 15 | `min_df`, `max_df`, `max_features`, `sublinear_tf` |
| **CB-2 BM25** | BM25 trên tag, lưới `k1` × `b` | 21 | `k1`, `b`, `min_df`, `max_df` |
| **CB-3 Tag Genome** | 1.128 chiều genome, chuẩn hoá, threshold, SVD | 22 | 19 `cb3-genome` + 3 `cb3-ridge` |
| **CB-4 Hybrid** | Late fusion BM25 + Genome | 35 | 32 `cb4-hybrid` + 3 `cb1-tfidf` time-decay |

> **Phân biệt hai con số dễ nhầm:** **144** là số dòng sweep thực sự đã chạy
> (51 CF + 93 CB, xem `results/final/ablation-*.csv`). **159** là số file trong
> `configs/` — gồm 144 sweep + 13 file `best-*.yaml` (bản config đã khoá) + 2 file
> `baseline-*.yaml`. Báo cáo dùng **144** khi nói về sweep và **159** khi nói về số
> file config.

**CB-1 — TF-IDF.** Với mỗi phim i, `tfidf(t, i) = tf(t,i) · idf(t) · L2-normalise`,
trong đó `idf(t) = ln((1+N)/(1+df(t))) + 1`. Tham số `sublinear_tf` quyết định dùng
`1 + ln(tf)` hay `tf` thẳng — quyết định này ảnh hưởng mạnh tới việc một tag bị
gán 500 lần có "áp đảo" tag được 5 lần hay không.

**CB-2 — BM25.** Công thức Robertson & Zaragoza (2009), với N = số phim,
`df_t` = số phim chứa tag t:

\[
idf_t = \ln\frac{N - df_t + 0{,}5}{df_t + 0{,}5} + 1
\qquad
w_{it} = idf_t \cdot \frac{c_{it}(k_1+1)}{c_{it} + k_1(1 - b + b\,\ell_i/\bar{\ell})}
\]

BM25 khác TF-IDF ở **đúng hai chỗ**, và cả hai đều cần thiết cho tag document:

- **Bão hoà tần suất** `c/(c+k1)`: một tag được 500 người gán không được áp đảo
  tag được 5 người gán gấp 100 lần. `sublinear_tf` của TF-IDF cũng làm việc này
  nhưng bằng `log`, không điều chỉnh được.
- **Chuẩn hoá độ dài tài liệu** qua `b`: phim blockbuster có hàng nghìn tag không
  tự động giống mọi thứ. Ở đây điều này đáng chú ý vì **số tag mỗi phim tương quan
  0,768 với độ phổ biến** (§3.2.3) — tức độ dài tài liệu chính là popularity đội lốt
  nội dung.

**CB-3 — Tag Genome.** Ma trận genome thô (13.816 × 1.128) được xử lý theo thứ tự
**bắt buộc**: threshold → standardize → SVD → L2-normalise. Threshold phải áp lên
*relevance thô* mới đúng nghĩa "tag yếu"; nếu chạy sau z-score thì ngưỡng không còn
là ngưỡng relevance — đo trên dữ liệu thật, nó xoá 76,7% ô thay vì 66,2%, và 15,1%
tổng số ô bị xoá oan dù relevance thô ≥ 0,1. Tệ hơn, sau z-score giá trị 0 nghĩa là
"đúng bằng trung bình của tag đó" chứ không phải "không liên quan", nên clamp về 0 là
đi mà phim về mức trung bình. SVD được cache theo vân tay của ma trận đầu vào, không
chỉ theo tham số.

**CB-4 — Hybrid late fusion.**

\[
S = \sum_m w_m \cdot \mathrm{minmax\_row}(S_m) + \lambda_{pop}\cdot \mathrm{pop\_norm}
\]

Chuẩn hoá min-max là bắt buộc trước khi cộng: thang điểm của BM25 (cosine trong
[0,1]) và Genome (tương tự) không so được với nhau nếu không chuẩn hoá. Điểm tinh tế:
min-max phải tính **chỉ trên tập ứng viên** (đã loại phim user xem), vì item user đã
xem chính là item điểm cao nhất của mô hình nội dung — nếu lấy min/max trên toàn bộ
13.816 phim, thang điểm của phần ứng viên sẽ phụ thuộc vào khoảng cách "đã xem/chưa
xem" riêng của từng mô hình, và trọng số `w = 0,75` sẽ không còn nghĩa là "75% mô hình
A".

### 4.3.3 Bẫy `pop_prior` — điều phải nêu trước khi trích dẫn bảng chính

CB-4 có hai biến thể: content thuần (`λ_pop = 0`) và có popularity prior
(`λ_pop = 0,4`). Bản có prior **tốt hơn** ở giao thức thường:

| Bản CB-4 | NDCG@10 (`full`) | NDCG@10 (`coldstart`) | `frac_unseen@10` (`coldstart`) |
|---|---:|---:|---:|
| Content thuần (`λ_pop = 0`) | 0,1054 | **0,0347** | 0,2223 |
| Có prior (`λ_pop = 0,4`) | **0,1081** | 0,0010 | 0,0036 |

*(Nguồn: `results/final/benchmark-coldstart.csv`, `results/final/beyond-accuracy-coldstart.csv`.)*

Nguyên nhân rất rõ: ở split cold-start, mọi phim held-out có popularity bằng 0 trong
tập train, nên prior **không phân biệt được** chúng và kéo điểm của mọi phim held-out
xuống đáy bảng xếp hạng.

**Vì vậy bảng chính của báo cáo lấy bản content thuần**, và bản có prior chỉ được
trích dẫn ở đây làm đối chứng. Đây là ví dụ điển hình của bài học ở §7.3: một siêu
tham số có vẻ chỉ "cải thiện" ở giao thức thường lại có thể phá hỏng hoàn toàn ở giao
thức khác — nên phải đánh giá trên **cả hai** trước khi chọn.

### 4.3.4 Chế độ trọng số hồ sơ

Ma trận trọng số `W` được dựng từ `train_csr` theo ba chế độ:

| Chế độ | Công thức | Ý nghĩa |
|---|---|---|
| `binary_pos` | 1 nếu rating ≥ 4,0, 0 ngược lại | Chỉ quan tâm user **có thích** hay không |
| `rating` | Giữ nguyên giá trị rating | Phim 5 sao đóng góp nhiều hơn 4 sao |
| `centered` | `r − ū_train` | Phim không thích kéo hồ sơ ra xa thể loại đó |

Ngoài ra có tham số `half_life_days` áp hệ số suy giảm `0,5^(Δt/half_life)` cho
mỗi tương tác — phim xem lâu rồi đóng góp ít hơn vào hồ sơ. Hiệu ứng của ba chế độ
được đo ở Chương 6.3.

---

## 4.4 Mô hình Collaborative Filtering

| Mô hình | Thư viện / cài đặt | Ghi chú |
|---|---|---|
| **ALS** | `implicit.als.AlternatingLeastSquares` | `c_ui = 1 + α`, chỉ tương tác relevant |
| **Item-CF (RS)** | Tự cài, trọng số = **rating thật** | Quét `k_neighbors`, `weight_mode` |
| **BPR-MF** | `implicit.bpr` | `num_threads=1` — tái lập được tuyệt đối |
| **ItemKNN-CF** | Tự cài, trọng số **nhị phân** | Trong registry, không vào bảng chính |

### 4.4.1 ALS cho implicit feedback

Hu, Koren & Volinsky (2008) mở rộng ALS sang phản hồi ẩn bằng cách gán mỗi tương tác
một mức tin cậy `c_ui` và tối ưu:

\[
\min_{P,Q} \sum_{u,i} c_{ui}\,(p_u^\top q_i - r_{ui})^2 + \lambda(\|P\|^2 + \|Q\|^2)
\]

**Ngữ nghĩa confidence — chỗ dễ sai nhất của toàn bộ đồ án.** Thư viện `implicit` coi
**giá trị lưu trong ma trận chính là `c_ui`**, rồi dùng `(c−1)` để tích luỹ `YᵀC_uY`
và `c` cho vế phải. Do đó:

- phải lưu **`1 + α`**, không phải `α`;
- **chỉ được lưu tương tác relevant** (rating ≥ 4,0). Nếu lưu cả rating thấp với
  `c = 1` thì số hạng `(c−1)` bằng 0 nhưng `b += 1·factor` vẫn cộng vào — tức mô
  hình học "user thích phim này với độ tin cậy 1" cho đúng những phim user đã chấm
  điểm thấp.

Đây chính là lý do `paths.RELEVANCE_THRESHOLD` được dùng thống nhất ở ALS, MostPop,
ItemKNN-CF và nhãn test.

Một chi tiết kỹ thuật khác: trong lúc `fit`, BLAS bị ép về 1 luồng qua
`threadpoolctl`. ALS đã tự song song hoá theo user; để OpenBLAS mở thêm threadpool
lồng nhau gây tranh chấp và làm chậm đi nhiều lần.

### 4.4.2 Item-based CF

Hai biến thể dùng cùng một nguyên lý ("người xem phim này cũng xem phim kia") nhưng
khác trọng số:

- **`rs-itemcf`** — trọng số = **rating thật** của user lên item, cộng trọng số của
  các hàng xóm có rating cao. Cho phim user thích điểm cao đóng góp nhiều hơn.
- **`cf-itemknn`** — trọng số **nhị phân** (có tương tác hay không). Bản này nằm
  trong registry để thí nghiệm nhưng **không vào bảng chính**.

Item-based CF scale tốt hơn user-based vì số phim (13.816) ít hơn nhiều lần số user
đánh giá (158.774), và ma trận tương đồng item×item thưa hơn ma trận user×user.

### 4.4.3 BPR-MF và lý do `num_threads=1`

BPR (Rendle et al., 2009) tối ưu mục tiêu pairwise ranking:

\[
\mathcal{L} = \sum_{(u,i,j)} -\ln \sigma(\hat{r}_{ui} - \hat{r}_{uj}) + \lambda\|\Theta\|^2
\]

với `j` lấy bằng **negative sampling** trên các item user chưa tương tác.

`num_threads=1` là quyết định **cố ý**: với nhiều luồng, thứ tự cập nhật tham số
phụ thuộc lịch trình, khiến kết quả **không tái lập được tuyệt đối** giữa các lần
chạy. Đánh đổi: `fit_s = 39,84` cho 158.774 user ở giao thức `full`
(*nguồn: cột `fit_s` của `results/final/full-summary.csv`*) thay vì ~7s khi dùng
nhiều luồng. Trong một đồ án đánh giá, tính tái lập được quan trọng hơn tốc độ — nhưng
đây là một đánh đổi đã ghi nhận, không phải lựa chọn miễn phí (§7.3).

---

## 4.5 Giao thức đánh giá

### 4.5.1 Ba giao thức

| Giao thức | Số user | `split_id` | Dùng để |
|---|---:|---|---|
| `dev` | 20.000 | `temporal80_20` | **chọn tham số** — sweep 144 config (51 CF + 93 CB) |
| `full` | 158.774 | `temporal80_20` | **bảng chính** — chạy **một lần** sau khi khoá config |
| `coldstart` | 10.992 | `temporal80_20+coldstart_item` | cold-start item (§5.3) |

> **Số user của `coldstart` phụ thuộc bảng:** `coldstart-item.csv` ghi
> `n_eval_users = 10.992`, còn `beyond-accuracy-coldstart.csv` dùng **10.988** user
> (đã loại 4 user không có tín hiệu nào để chấm). Khi trích dẫn phải nói rõ đang
> nói về bảng nào.

**`dev` ⊂ `full`**, cùng `train.parquet`/`test.parquet` — khác biệt duy nhất là ai
được chấm điểm. Median train/user = 58 ở cả hai; NDCG@10 lệch ≤ 1,7% và **thứ hạng
không đảo** giữa `dev` và `full`. Đây là cơ sở để dùng biên độ sweep đo ở `dev`
(20.000 user, rẻ hơn nhiều) trong báo cáo, với điều kiện phải **ghi rõ mức user của
từng bảng**.

**Giao thức cold-start item:** giữ lại **1.381 phim** (10% catalog, phân tầng theo
5 nhóm năm phát hành) và **xoá toàn bộ tương tác train** của các phim đó. Mô hình
được huấn luyện lại trên split này rồi chấm điểm trên tập relevant của các phim
held-out. Đây là nơi kết luận đảo chiều (§5.3).

### 4.5.2 Vì sao full ranking, không sampled

Mỗi user được chấm điểm trên **toàn bộ 13.816 ứng viên**. Lý do:

1. **Sampled metrics có thể đảo thứ hạng.** Krichene & Rendle (2020) chỉ ra rằng với
   một mẫu negative nhỏ, thứ hạng giữa hai thuật toán có thể bị đảo — tức phép đo
   không còn bảo toàn thứ hạng mà ta muốn so sánh.
2. **Ở đây phép đo còn đặc biệt rẻ.** Với NDCG@10, một item có relevance nhị phân,
   toàn bộ đóng góp vào DCG nằm ở 10 vị trí đầu — nên chỉ cần sắp xếp một phần
   nhỏ đầu danh sách cũng đủ dùng làm ứng viên cho điểm số, miễn là bảo đảm không
   bỏ sót ứng viên thật sự có thể lọt vào top-10.
3. **Chi phí vẫn chấp nhận được.** Chấm theo lô 2.000 user để tránh dựng ma trận
   8,9 GB; toàn bộ giao thức `full` mất 21,67s (BPR) đến 140,19s (CB-4) cho phần
   chấm điểm.

> **Giới hạn phải nói thẳng:** giao thức thay thế (LOO + 100 negative) **đã được cài
> đặt** (`scripts/run_sampled_loo_comparison.py`) nhưng **chưa chạy** — `results/`
> không có file kết quả LOO/Kendall τ nào. Vì vậy lựa chọn full ranking ở đây
> **dựa trên trích dẫn, chưa có bằng chứng thực nghiệm trong repo**. Đây là hướng
> phát triển ưu tiên cao nhất (§8.2) và là một giới hạn được nêu ở §7.4.

### 4.5.3 Bằng chứng protocol trên từng dòng kết quả

Mỗi dòng trong mọi bảng kết quả mang ba cột:

| Cột | Ví dụ | Ý nghĩa |
|---|---|---|
| `split_id` | `temporal80_20` | Cách chia tập đã dùng |
| `catalog_id` | `ml25m-n13816-de09a1b710f6` | Mã băm của tập ứng viên (13.816 phim + mã cấu hình) |
| `seed` | `42` | Hạt random toàn cục |

Cơ chế này biến câu "các mô hình này được đánh giá trên cùng một giao thức" từ **lời
tuyên bố** thành **thứ kiểm chứng được**: đối chiếu `split_id` + `catalog_id` của
hai dòng là kiểm tra ba trường hợp ký tự. Người đọc không phải tin lời.

`scripts/check_report_numbers.py` dùng chính bộ bằng chứng này để đối chiếu mọi
con số trong báo cáo với 27 bảng CSV — và phải báo **0 lệch**.

### 4.5.4 Metric đánh giá

Bài toán là **xếp hạng top-N**, không phải dự đoán rating, nên không dùng RMSE/MAE
(định nghĩa và lý do ở §2.4). Bốn metric chính:

| Metric | Định nghĩa | Ý nghĩa |
|---|---|---|
| `P@k` | `hits / k` | Tỉ lệ người dùng mà đúng `k` người dùng khác sẽ thích |
| `R@k` | `hits / \|relevant\|` | Độ phủ của tập relevant |
| `HR@k` | Tỉ lệ user có ≥ 1 hit trong top-k | User nào tìm được thứ muốn |
| `NDCG@k` | DCG chuẩn hoá theo IDCG | Có trọng số theo vị trí |

> **Định nghĩa đúng của `P@k` là `hits / k`** — phân biệt với `hits / len(top_k)`.
> Với `k = 10` và danh sách luôn đủ 10 phim thì hai cách trùng nhau, nhưng định
> nghĩa đúng phải là `hits/k` và evaluator hiện dùng đúng cách đó.

Ngoài độ chính xác còn bốn metric mô tả **hành vi** của hệ thống: `coverage@10`
(tỉ lệ phim từng được gợi ý), `gini@10` (mức dồn suất), `novelty@10` (độ mới lạ),
`ILD@10` (độ đa dạng nội dung trong danh sách). Kết quả ở Chương 6.4 cho thấy
coverage **đảo ngược** với độ chính xác.

Kiểm định thống kê: **bootstrap CI 95%** (`bootstrap-ci.csv`) và **Wilcoxon
signed-rank ghép cặp** (`wilcoxon-{cf,cb,coldstart}.csv`).

---

## 4.6 Tóm tắt: thiết kế nào đảm bảo so sánh được

| Yếu tố thiết kế | Đảm bảo điều gì |
|---|---|
| Một evaluator, một đường chạy | Không còn so sánh giữa hai protocol khác nhau |
| `TrainData` tách `Dataset` | Mô hình không thể chạm nhãn test (ở tầng cấu trúc) |
| Cùng catalog 13.816 phim cho mọi mô hình | Bảng so sánh là so thuật toán, không so bài toán |
| Cùng ngưỡng relevant ≥ 4,0 | "Relevant" nghĩa giống nhau cho mọi mô hình |
| Split thời gian per-user | Mô hình bị hỏi về tương tác về sau, không phải quá khứ |
| Sweep cân bằng 51 CF / 93 CB | Kết luận "CF thắng" không mang nghi ngờ thiên lệch |
| `split_id` + `catalog_id` + `seed` | "Cùng giao thức" kiểm chứng được, không phải lời tuyên bố |
| Đánh giá trên cả `full` lẫn `coldstart` | Không bỏ sót điều kiện mà phương pháp thắng |
| Bằng chứng sweep commit trong repo | Người đọc tự đếm, không phải tin |

---

**Tài liệu tham khảo chương này:** Hu, Koren & Volinsky (2008); Koren, Bell &
Volinsky (2009); Rendle et al. (2009); Robertson & Zaragoza (2009); Järvelin &
Kekäläinen (2002); Krichene & Rendle (2020); Harper & Konstan (2015).
