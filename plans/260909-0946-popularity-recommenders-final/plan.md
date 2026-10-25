# Kế hoạch: Hoàn thiện Popularity Recommenders (Bayesian + Wilson + Popularity cơ bản)

## 1. Bối cảnh & phạm vi

Kế hoạch này là bản **rút gọn, bằng tiếng Việt**, dựa trên thiết kế đã được phê duyệt tại `plans/popularity-recommenders-plan.md` và bổ sung thêm **hoàn thiện `PopularityRecommender` hiện đang ở trạng thái stub** trong `src/baselines.py`. Giai đoạn lập kế hoạch **không sửa source code hay test** — chỉ mô tả TODO phases, file chính xác, tiêu chí chấp nhận và lệnh validation.

**Trong phạm vi:**
1. `PopularityRecommender` — hoàn thiện (đổi tên thuộc tính, ổn định thứ tự, docstring, edge cases).
2. `BayesianPopularityRecommender` — IMDb weighted rating.
3. `WilsonLowerBoundRecommender` — Wilson 95 % lower bound trên tỉ lệ rating ≥ 4.
4. `tests/test_baselines_popularity.py` — pytest đơn vị thuần tuý.

**Ngoài phạm vi:**
- `src/evaluation.py` (giữ nguyên trạng thái rỗng).
- UserCF, ItemCF, ALS, Content-based, BPR/SVD/SVD++.
- `notebooks/`, `notes/`, `docs/`, `requirements.txt`.

## 2. File liên quan

| File | Trạng thái | Hành động |
|---|---|---|
| `src/baselines.py` | Có sẵn | **Đã sửa**: 3 lớp + `_RankedRecommender` base + `_validate_ratings` |
| `tests/test_baselines_popularity.py` | Mới | **Đã tạo**: 14 hàm test (parametrize mở rộng → 28 case) |

Không file nào khác bị đụng.

## 3. Chuỗi phụ thuộc

```
[1] Hoàn thiện PopularityRecommender   (src/baselines.py)   ✅ DONE
   ▼
[2] Thêm BayesianPopularityRecommender  (src/baselines.py) ✅ DONE — default m đổi median→mean (xem § 9)
   ▼
[3] Thêm WilsonLowerBoundRecommender    (src/baselines.py) ✅ DONE
   ▼
[4] Tạo tests/test_baselines_popularity.py                 ✅ DONE (28 passed, xem § 5)
   ▼
[5] Validation: import smoke + pytest                       ✅ DONE
```

## 4. Chi tiết từng phase

### Phase 1 — Hoàn thiện `PopularityRecommender` ✅ DONE
- Đổi tên thuộc tính: `popular_items` → `ranked_items_`, `score_by_item` → `scores_` (thống nhất với 2 lớp mới).
- Dùng `sort_values(..., kind="mergesort")` để xác định khi bằng điểm.
- Thêm docstring ≤ 3 dòng; giữ nguyên hợp đồng `fit(train_df) → recommend(uid, n, seen) -> list[int]`.
- **Triển khai thực tế:** dùng `_RankedRecommender` base chung (`ranked_items_` + `scores_` + `popular_items`/`score_by_item` alias cho backward-compat) và `_validate_ratings` helper.

### Phase 2 — `BayesianPopularityRecommender` ✅ DONE (default m đã đổi median→mean)
- Công thức: `WR = (v/(v+m))·R + (m/(v+m))·C`; `m` mặc định = `mean(v_i)` (đổi từ `median(v_i)` sau review F1, xem § 9).
- Dùng `groupby("movieId")["rating"].agg(["count", "mean"])` để tính `v`, `R`; `C = train_df["rating"].mean()`.
- `scores_` lưu `dict[int, float]`; `ranked_items_` lưu `list[int]` đã sắp xếp.

### Phase 3 — `WilsonLowerBoundRecommender` ✅ DONE
- Fast-path: `positive = (train_df["rating"] >= 4).groupby(train_df["movieId"]).sum()` (tránh `groupby.apply` chậm trên ML-25M).
- `lower = (p̂ + z²/(2v) − z·√(p̂(1−p̂)/v + z²/(4v²))) / (1 + z²/v)` với `z = 1.96`.
- `.clip(0, 1)` chống underflow; `p_hat.fillna(0)` khi `v == 0`.

### Phase 4 — `tests/test_baselines_popularity.py` ✅ DONE
- Fixture `toy_df` 10 dòng, 3 user, 3 phim, rating 1–5.
- 14 hàm test, mở rộng parametrize → **28 passed** theo report targeted validation.
- Mapping AC (xem § 5) + edge cases (`n=0`/`n=-1`/`n > len`, refit-replaces-state, `seen=set()` vs `seen=None`, `ValueError` cho input thiếu cột / `train_df` rỗng).

### Phase 5 — Validation ✅ DONE
```bash
python -c "from src.baselines import PopularityRecommender, BayesianPopularityRecommender, WilsonLowerBoundRecommender; print('imports OK')"
pytest -q tests/test_baselines_popularity.py   # 28 passed
```

## 5. Tiêu chí chấp nhận (AC)

| AC | Mô tả | Test mapping |
|---|---|---|
| AC-1 Hợp đồng | `fit` trả `self`; `recommend` trước `fit` → `RuntimeError`; trả `list[int]` độ dài ≤ `n` | `test_*_fit_returns_self`, `test_*_recommend_before_fit_raises`, `test_*_recommend_length` |
| AC-2 Bayes | Item `v → ∞` → `score ≈ R`; item `v → 0` → `score ≈ C`; top-1 ưu tiên vừa nhiều vote vừa rating cao | `test_bayes_shrinkage_pulls_low_vote`, `test_bayes_top1_higher_mean` |
| AC-3 Wilson | `0 ≤ lower ≤ 1`; `lower == 0` khi `positive == 0`; xếp hạng không tăng | `test_wilson_lower_in_unit_interval`, `test_wilson_ranking_non_increasing` |
| AC-4 Determinism | Hai lần `fit` → cùng `ranked_items_` | `test_*_deterministic` |
| AC-5 Seen filter | `set(recs) & seen == ∅` | `test_*_excludes_seen` |
| AC-6 Backward compat | `PopularityRecommender` cũ vẫn import và hoạt động | `test_existing_popularity_contract` |

## 6. Done Definition

- [x] 3 lớp đúng công thức, cùng hợp đồng, đặt trong `src/baselines.py`.
- [x] `tests/test_baselines_popularity.py` chạy xanh toàn bộ (28 passed).
- [x] Smoke-test import in ra "imports OK".
- [x] `src/evaluation.py` chưa bị sửa.
- [x] Không thay đổi `requirements.txt`.

## 7. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| `groupby.apply` chậm trên 25M dòng | Fast-path `(rating >= 4).groupby(movieId).sum()` trong Phase 3 |
| Sort không ổn định | `kind="mergesort"` cho cả 3 lớp |
| Underflow Wilson | `.clip(0, 1)` |
| `pytest` chưa cài | `pip install pytest` (không sửa `requirements.txt`) |

## 8. Ước lượng

- Triển khai: ~30 phút.
- Test: ~25 phút.
- Validation: < 1 phút.

## 9. Câu hỏi chưa giải quyết

- ~~Ngưỡng `m` Bayesian mặc định: chốt `median(v_i)`.~~ ✅ **Đã chốt `mean(v_i)`** theo review `plans/reports/reviewer-260909-popularity-baselines.md` (F1-HIGH). Lý do: trên ML-25M `median(v_i) ≈ 1` → co về gần item-mean, vô hiệu hóa shrinkage; `mean(v_i) = total_ratings / n_items` là heuristic IMDb/BR chuẩn. Đã thay đổi tại `src/baselines.py:85` (`float(votes.mean())`) và cập nhật docstring (`BayesianPopularityRecommender` line 70-72). Test `test_bayesian_default_prior_is_mean_votes_per_item` pin hành vi mới.
- `recommend` trả `list[int]` hay `list[tuple[int, float]]`? — Kế hoạch giữ `list[int]` để tương thích hợp đồng hiện hành và `src/evaluation.py` (khi được populate). Nếu sau này cần score, đổi cả 3 lớp cùng lúc.
- ⚠ **Scope note chưa giải quyết:** `src/evaluation.py` vẫn rỗng — AC-6 chỉ kiểm tra importability của `PopularityRecommender`, chưa chạy end-to-end `evaluate_model` được. Cần phase riêng để populate evaluator trước khi benchmark trên ML-25M.

---

**Status:** DONE (đồng bộ với work đã hoàn thành)
**Summary:** Đã tạo `plans/260909-0946-popularity-recommenders-final/plan.md` — bản kế hoạch tiếng Việt, rút gọn, liệt kê 5 phase (hoàn thiện Popularity + 2 lớp mới + test + validation), file chính xác (`src/baselines.py`, `tests/test_baselines_popularity.py`), 6 nhóm AC, 2 lệnh validation. Giữ `src/evaluation.py` ngoài phạm vi như yêu cầu. **Đồng bộ lần 2 (260909):** tất cả 5 phase chuyển sang ✅ DONE, default Bayesian `m` đổi `median(v_i) → mean(v_i)` theo review F1 (evidence: `src/baselines.py:85`, test `test_bayesian_default_prior_is_mean_votes_per_item`); targeted validation `pytest tests/test_baselines_popularity.py` = 28 passed; scope note `src/evaluation.py` còn rỗng được giữ nguyên.
**Concerns/Blockers:**
- `src/evaluation.py` vẫn rỗng → chưa end-to-end benchmark được; cần phase populate evaluator trước khi chạy ML-25M evaluation (xem § 9 scope note).
- Reviewer nêu F2 (test ordering vs exclusion nên tách) và F3 (edge-case coverage) — code đã bổ sung `n=0`/`n=-1`/`seen=set()`/refit-replaces-state vào test file, F2 rename/fixture split có thể xử lý ở phase polish sau nếu cần.
- `PopularityRecommender` skeleton ban đầu không có `recommend` body — đã được implementer hoist lên `_RankedRecommender` base chung cho cả 3 lớp.
