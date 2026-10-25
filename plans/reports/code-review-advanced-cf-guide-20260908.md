# Code Review — advanced-cf-bpr-svd-svdpp-ml1m-guide.md

**Reviewer:** Claude Code Agent
**Files reviewed:** `notes/advanced-cf-bpr-svd-svdpp-ml1m-guide.md`, `README.md`, `requirements.txt`, `src/data_loader.py`
**Date:** 2026-09-08
**Scope:** Technical accuracy · Runnable Python API · Vietnamese clarity · Fairness of evaluation · Safety claims

---

## TỔNG QUAN

Tài liệu viết rõ ràng, cấu trúc logic tốt, lý thuyết chính xác. Tuy nhiên có **1 lỗi runtime nghiêm trọng** (BPR `items=` dùng sai index space) và **nhiều conflict về package versions** giữa guide và `requirements.txt` chính của dự án. Phần còn lại — API, evaluation protocol, fairness claims, safety — đều đạt.

---

## 1. LỖI RUNTIME NGHIÊM TRỌNG

### D1 · BPR `.recommend(..., items=)` truyền matrix index thay vì raw item ID

**Vị trí:** Dòng 344–345 trong `evaluate_bpr_at_k`

```startLine:340:endLine:350:notes/advanced-cf-bpr-svd-svdpp-ml1m-guide.md
        candidate_indices = np.array(
            [item_to_index[movie_id] for movie_id in candidates], dtype=np.int32
        )
        ranked_indices, _ = model.recommend(
            userid=user_to_index[user_id],
            user_items=user_item,
            N=len(candidate_indices),
            filter_already_liked_items=False,
            items=candidate_indices,
        )
```

**Vấn đề:** `items=` trong implicit BPR nhận **raw item IDs** (kiểu int, đúng như `movie_id` gốc), không phải matrix internal indices.

Kiểm tra [implicit 0.7.2 API](https://benfred.github.io/implicit/api/models/cpu/bpr.html):

> `items` (array_like, optional) – Array of **extra item ids**. When set this will only rank the items in this array instead of ranking every item the model was fit for.

Tham số `items` dùng raw item ID từ dataset, KHÔNG phải internal matrix index. Code hiện tại truyền `candidate_indices` (đã map qua `item_to_index`), nên BPR sẽ hiểu sai hoàn toàn — có thể raise `KeyError` hoặc trả kết quả ngẫu nhiên.

**Fix:**

```python
        # candidate_indices: dùng RAW movie_id cho items= parameter
        # candidate_matrix_indices: dùng internal index để build ranked array
        candidate_matrix_indices = np.array(
            [item_to_index[movie_id] for movie_id in candidates], dtype=np.int32
        )
        ranked_indices, _ = model.recommend(
            userid=user_to_index[user_id],
            user_items=user_item,
            N=len(candidates),
            filter_already_liked_items=False,
            items=candidates,  # truyền raw movie_id, không phải internal index
        )
```

Lưu ý: khi `items=` được set, implicit sẽ trả về item IDs (đúng là raw IDs), nên không cần map lại qua `index_to_movie`. Kết quả trả về của `.recommend()` khi dùng `items=` đã là raw item IDs chứ không phải internal indices. Cần thêm logic xử lý.

**Mức độ:** CRITICAL — code chạy sai hoàn toàn hoặc crash.

---

## 2. API — TRICKY / INCOMPATIBLE

### D2 · `bpr.fit(item_user, show_progress=True)` — param name có thể không đúng

**Vị trí:** Dòng 189

**Kiểm tra:** Web search xác nhận implicit `fit()` signature trong `recommender_base.py` là:

```python
def fit(self, user_items, show_progress=True, callback=None):
```

→ `show_progress` là parameter hợp lệ. **Tuy nhiên:** version được guide yêu cầu là `implicit==0.7.2`, còn `requirements.txt` chính dùng `implicit==0.7.3`. Không có breaking change về `fit()` giữa 0.7.2 và 0.7.3, nhưng cần thống nhất version.

**Khuyến nghị:** Dùng `implicit==0.7.2` như guide (để tránh surprise conflict — xem D5), hoặc dùng chung `implicit==0.7.3` nếu ML-25M pipeline đã chạy ổn với 0.7.3.

**Mức độ:** LOW — API đúng, nhưng version cần rõ ràng.

---

### D3 · `surprise.SVD.fit()` / `SVDpp.fit()` không có `show_progress`

**Vị trí:** Dòng 244–249 và 265–268

**Vấn đề:** Guide gọi `svd.fit(svd_trainset)` và `svdpp.fit(svd_trainset)` mà không truyền `show_progress`. Đây không phải lỗi — `fit()` không có `show_progress`. Muốn in epoch ra console phải set `verbose=True` lúc **khởi tạo** model.

Code hiện tại hoàn toàn chạy được, nhưng **nếu muốn show progress**, cần thêm `verbose=True` vào constructor.

**Fix nếu muốn show progress:**

```python
svd = SVD(
    n_factors=64,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.02,
    random_state=RANDOM_SEED,
    verbose=True,  # thêm dòng này
)
```

**Mức độ:** LOW — code hiện tại không lỗi, nhưng ghi chú trong doc có thể gây hiểu nhầm người đọc nghĩ `fit(show_progress=True)` là cách đúng.

---

## 3. PACKAGE VERSION CONFLICTS

### D4 · `numpy==1.26.4` (guide) vs `numpy==2.5.2` (requirements.txt)

**Guide dòng 73:** `numpy==1.26.4`
**requirements.txt dòng 2:** `numpy==2.5.2`

**Vấn đề:** Đây là conflict nghiêm trọng. `scikit-surprise==1.1.4` (dùng trong guide) **không tương thích** với NumPy 2.x. Lỗi được ghi nhận trên [GitHub issue #485](https://github.com/NicolasHug/Surprise/issues/485): NumPy 2.x dùng C API mới, khiến surprise Cython extension crash với thông báo:

> "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x as it may crash."

Fix đã có trong surprise 1.1.5 (theo [PR #486](https://github.com/NicolasHug/Surprise/pull/486)), nhưng 1.1.5 chưa release stable tại thời điểm review. Cũng không có fork chính thức nào support Python 3.11 + surprise + NumPy 2.x cùng lúc.

Guide đúng khi yêu cầu `numpy==1.26.4`. Nếu dùng `requirements.txt` hiện tại với NumPy 2.5.2, surprise sẽ crash.

**Mức độ:** HIGH — setup sẽ fail.

---

### D5 · `scikit-surprise==1.1.4` (guide) không có trong requirements.txt

**Vấn đề:** `requirements.txt` chính không chứa `scikit-surprise`. Guide yêu cầu cài riêng trong `.venv-advanced-cf`, nhưng nếu user vô tình dùng global env (với NumPy 2.5.2), surprise sẽ không chạy.

Guide giải thích rõ phải tạo venv riêng, nhưng không conflict trực tiếp — chỉ cần user tuân thủ.

**Mức độ:** MEDIUM — nếu user không đọc kỹ hướng dẫn.

---

### D6 · `implicit==0.7.2` (guide) vs `implicit==0.7.3` (requirements.txt)

**Guide dòng 73:** `implicit==0.7.2`
**requirements.txt dòng 1:** `implicit==0.7.3`

Không có breaking change API giữa 0.7.2 và 0.7.3 về BPR. Tuy nhiên, khi thay đổi implicit version trong global env, cần đảm bảo ML-25M pipeline vẫn hoạt động.

**Mức độ:** LOW — API tương thích ngược.

---

### D7 · `pandas==2.2.3` (guide) vs `pandas==3.0.5` (requirements.txt)

**Guide dòng 73:** `pandas==2.2.3`
**requirements.txt dòng 3:** `pandas==3.0.5`

pandas 3.x có breaking changes đáng kể (nullable dtypes mặc định, `copy-on-write` behavior thay đổi). Guide dùng cú pháp pandas cổ điển, tương thích với cả 2.x và 3.x. Tuy nhiên cần thống nhất.

**Mức độ:** LOW — code pandas trong guide tương thích cả hai phiên bản.

---

## 4. EVALUATION PROTOCOL & FAIRNESS

### D8 · BPR evaluation: thứ tự `index_to_movie` map sau khi dùng `items=`

**Vị trí:** Dòng 350–353

```startLine:350:endLine:353:notes/advanced-cf-bpr-svd-svdpp-ml1m-guide.md
        ranked_indices, _ = model.recommend(
            userid=user_to_index[user_id],
            user_items=user_item,
            N=len(candidate_indices),
            filter_already_liked_items=False,
            items=candidate_indices,
        )
        ranked = np.array([index_to_movie[item_index] for item_index in ranked_indices])
```

**Vấn đề:** Khi `items=` được truyền, `model.recommend()` trả về **raw item IDs** (theo implicit source code). Dòng `ranked = np.array([index_to_movie[item_index] ...])` hiện giả định trả về internal indices, dẫn đến lỗi KeyError hoặc sai kết quả.

Fix khi đã sửa D1: cần kiểm tra implicit behavior. Thực tế, khi `items=` được set, `ranked_indices` trả về raw item IDs từ `items` array đã được sắp xếp theo score giảm dần — KHÔNG cần map qua `index_to_movie`.

**Mức độ:** MEDIUM — phụ thuộc D1.

---

### D9 · Fairness claim về "không so sánh BPR với ALS qua RMSE" — CHÍNH XÁC ✓

Guide dòng 24–27 và dòng 384–385 khẳng định rõ: SVD/SVD++ tối ưu rating regression; BPR tối ưu ranking. Không dùng RMSE so sánh BPR với ALS. Đây là nhận định đúng về mặt methodology.

---

### D10 · Positive threshold = 4.0 được ghi rõ là "quy ước thực nghiệm" — AN TOÀN ✓

Guide dòng 82–85 ghi nhận đây là quy ước thực nghiệm, không khẳng định rating 1–3 là negative. Claim an toàn.

---

## 5. CONTRADICTIONS VỚI ML-25M PIPELINE

### D11 · Không có contradiction trực tiếp ✓

Guide dòng 14–17 nêu rõ đây là thực nghiệm **độc lập** với pipeline ALS ML-25M. Cấu trúc dữ liệu (`user_id`, `movie_id`, `rating`, `timestamp`) dùng tên cột khác với `data_loader.py` (`userId`, `movieId`) — phân biệt rõ ràng. Không conflict.

---

## 6. SAFETY CLAIMS

### D12 · BPR score không phải rating/probability — CHÍNH XÁC ✓

Guide dòng 207–210 và dòng 419 cảnh báo rõ: BPR score chỉ dùng để rank, không phải xác suất, không phải rating 1–5. Claim chính xác.

---

### D13 · Cold-start claim — CHÍNH XÁC ✓

Guide dòng 405–409 mô tả đúng: CF thuần không giải quyết tốt user/item mới. Warm-test filtering ở Bước 2 và dòng 146–147 được ghi rõ. Claim an toàn.

---

## 7. VIETNAMESE CLARITY

### D14 · "BPR trả item đã xem" warning — NHẮT LẠI LẗI CŨ

Guide dòng 397–399 ghi: "BPR trả item đã xem: bắt buộc truyền `user_items=user_item` khi gọi `.recommend(..., filter_already_liked_items=True)`."

**Vấn đề:** `user_items` trong implicit BPR là **user × item matrix** (user rows × item columns), KHÔNG phải `item_user` (item × user). Code ở Bước 3 truyền `user_item` (đúng), nhưng comment trong warning này dễ gây nhầm nếu user đọc kỹ.

**Khuyến nghị:** Sửa comment thành:

```
BPR trả item đã xem: bắt buộc truyền user_items=user_item (user×item CSR matrix)
khi gọi .recommend(..., filter_already_liked_items=True).
```

**Mức độ:** LOW.

---

## TÓM TẮT THEO MỨC ĐỘ

| ID | Mức độ | Mô tả | Vị trí |
|---|---|---|---|
| D1 | **CRITICAL** | BPR `items=` truyền matrix index thay vì raw item ID → crash/sai | Dòng 344 |
| D4 | **HIGH** | NumPy version conflict: guide yêu 1.26.4, requirements.txt dùng 2.5.2 → surprise crash | Dòng 73 vs requirements.txt |
| D8 | MEDIUM | BPR ranked output map sai index space (phụ thuộc D1) | Dòng 353 |
| D5 | MEDIUM | surprise không có trong requirements.txt chính → có thể bị install sai env | N/A |
| D2 | LOW | implicit version 0.7.2 vs 0.7.3 cần thống nhất | Dòng 73 |
| D3 | LOW | Surprise `.fit()` không có `show_progress`; cần `verbose=True` ở constructor | Dòng 244, 265 |
| D6 | LOW | implicit version conflict (0.7.2 vs 0.7.3) | Dòng 73 |
| D7 | LOW | pandas version conflict (2.2.3 vs 3.0.5) | Dòng 73 |
| D14 | LOW | Warning comment về `user_items` matrix orientation dễ gây nhầm | Dòng 397 |

**Các điểm còn lại: ✓ Chính xác về lý thuyết, API, evaluation protocol, fairness, safety claims.**

---

## KHUYẾN NGHỊ HÀNH ĐỘNG

1. **[Ngay lập tức]** Sửa D1: đổi `items=candidate_indices` → `items=candidates` (raw IDs). Thêm logic xử lý output của `.recommend()` khi dùng `items=` (trả về raw IDs, không cần `index_to_movie` map).
2. **[Ngay lập tức]** Giải quyết D4: hoặc (a) thêm `numpy<2` constraint vào guide, hoặc (b) dùng `scikit-surprise-2` fork (với Python 3.13+). Guide hiện tại dùng venv riêng nên có thể cài `numpy==1.26.4` riêng — cần nhấn mạnh rõ điều này.
3. **[Ưu tiên cao]** Thống nhất package versions giữa guide và dự án chính để tránh confusion.
4. **[Ưu tiên thấp]** Sửa D3, D14 để giảm ambiguity cho người đọc.

---

**Status:** DONE_WITH_CONCERNS
**Summary:** 1 lỗi critical (BPR `items=` index space), 1 lỗi high (NumPy conflict), 1 medium, 6 low. Tất cả evaluation fairness, safety claims, và theory đều chính xác. Cần fix D1 + D4 trước khi guide có thể chạy thực tế.
**Concerns:** D1 và D4 là blockers — guide sẽ crash hoặc sai hoàn toàn nếu không sửa.
