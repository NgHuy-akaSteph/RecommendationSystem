# Tổng hợp triển khai hướng A & C — Non-Personalized Popularity Recommenders

**Phạm vi:** `src/baselines.py` (PopularityRecommender + BayesianPopularityRecommender + WilsonLowerBoundRecommender) và `tests/test_baselines_popularity.py` (28 case pytest).
**Ngày:** 2026-09-09
**Validation:** `pytest -q tests/test_baselines_popularity.py` = **28 passed** (chạy qua `.venv/bin/python3` trên host này).

Tài liệu này tổng hợp lại hai hướng đã duyệt:
- **A — Bayesian popularity** (IMDb-style weighted rating).
- **C — Wilson popularity** (Wilson 95% lower bound trên tỉ lệ rating dương).

Kèm theo việc hoàn thiện `PopularityRecommender` (count) hiện đang ở dạng stub, đảm bảo cả ba lớp dùng chung một hợp đồng `fit/recommend` và một bộ test thuần tuý.

---

## 1. Bối cảnh & mục tiêu

Triển khai ba baseline **không cá nhân hoá** trên dữ liệu rating MovieLens, phục vụ đối chiếu trước khi đưa vào mô hình collaborative filtering ở phase sau:

1. **PopularityRecommender** — đếm số lượt rating, hoàn thiện skeleton đang stub.
2. **BayesianPopularityRecommender** — IMDb weighted rating, co về mean vì có quá ít vote.
3. **WilsonLowerBoundRecommender** — khoảng tin cậy dưới 95% cho tỉ lệ rating dương.

Yêu cầu cứng:
- Giữ hợp đồng `fit(train_df) -> recommend(user_id, n, seen=None) -> list[int]` như `notes/person1-implementation-guide.md` § Phần 1 đã mô tả.
- Tie-break tất định, không phụ thuộc thứ tự hash.
- Validate input thiếu cột / DataFrame rỗng; ném `RuntimeError` khi gọi `recommend` trước `fit`.
- Out-of-scope: `src/evaluation.py` (giữ nguyên rỗng); không đụng `requirements.txt`, `notebooks/`, `notes/`.

---

## 2. Công thức & quyết định thiết kế

### 2.1. Popularity baseline (count)

```text
score_i = #rating của item i
```

- Đếm theo `groupby("movieId").size()`.
- Chỉ phản ánh cường độ tiếp cận, không phản ánh chất lượng — đây là baseline tham chiếu để cho thấy tại sao cần Bayesian / Wilson.

### 2.2. A — Bayesian weighted rating

```text
WR_i = (v_i / (v_i + m)) * R_i + (m / (v_i + m)) * C

v_i : #rating của item i
R_i : mean rating của item i
C   : global_mean = mean(rating) trên toàn tập train
m   : prior strength (>= 0)
```

Quyết định:
- **Default `m = mean(v_i)`** (tức `total_ratings / n_items`).
  - Lý do: review phát hiện `m = median(v_i)` (đề xuất đầu) cho `m ≈ 1` trên MovieLens 25M vì phân bố long-tail → co về gần `R_i` cho item phổ biến, vô hiệu hoá co.
  - `mean(v_i)` duy trì tỉ lệ shrinkage tương xứng với mật độ dữ liệu, và là heuristic IMDb/BR chuẩn.
- Có thể override `m` qua `BayesianPopularityRecommender(m=...)` (phải `> 0`).
- Tính `v_i` và `R_i` bằng `groupby("movieId")["rating"].agg(["count","mean"])` — vector hoá trên pandas.

### 2.3. C — Wilson lower bound cho tỉ lệ rating dương

```text
p̂ = positives / votes
positive = (rating >= 4)
lower = (p̂ + z²/(2v) − z·√(p̂(1−p̂)/v + z²/(4v²))) / (1 + z²/v)
lower ← clip(lower, 0, 1)
```

Quyết định:
- `z = 1.96` (khoảng tin cậy 95%).
- `positive = (rating >= 4)` — group theo `movieId` qua fast-path `(rating >= 4).groupby(movieId).sum()`, tránh `groupby.apply` chậm trên ML-25M.
- Clip `[0, 1]` chống underflow khi `v → 0`.
- Item không có rating dương → score `= 0`.

### 2.4. Hợp đồng & tie-break chung cho cả 3 lớp

- Base `_RankedRecommender` lưu `ranked_items_: list[int]`, `scores_: dict[int, float]`, đồng thời giữ alias `popular_items`, `score_by_item` để tương thích với guide triển khai cũ.
- `fit(df)` validate cột `movieId` + `rating` và `df.empty`; trả `self`.
- `recommend(uid, n=10, seen=None)`:
  - Gọi trước `fit` → `RuntimeError("Call fit() first")`.
  - `n <= 0` → `[]`; `n > len(ranked_items_)` → trả toàn bộ.
  - `seen` falsy (`None` hoặc `set()`) → trả `ranked_items_[:n]`.
  - Có `seen` → lọc item đã xem rồi cắt top-`n`.
- **Tie-break tất định**: `sort_values(["score", "movieId"], ascending=[False, True], kind="mergesort")`. `mergesort` ổn định; tie-break theo `movieId` tăng dần để ranking giống nhau qua các lần chạy.

---

## 3. Snippet mã (tham chiếu, không phải commit)

Các đoạn dưới đây **chính là nội dung thực tế** trong `src/baselines.py`, sắp xếp theo thứ tự file.

### 3.1. Base `_RankedRecommender` + validation

```python
import pandas as pd


class _RankedRecommender:
    """Shared ranking state and seen-item filtering for popularity models."""

    def __init__(self) -> None:
        self.ranked_items_: list[int] | None = None
        self.scores_: dict[int, float] | None = None
        # Giữ tên cũ để tương thích hướng dẫn triển khai trước đó.
        self.popular_items: list[int] | None = None
        self.score_by_item: dict[int, float] | None = None

    def _set_ranking(self, scores: pd.Series) -> None:
        ranking = (
            scores.rename("score")
            .rename_axis("movieId")
            .reset_index()
            .sort_values(
                ["score", "movieId"],
                ascending=[False, True],
                kind="mergesort",
            )
        )
        self.ranked_items_ = [
            int(movie_id) for movie_id in ranking["movieId"]
        ]
        self.scores_ = {
            int(movie_id): float(score)
            for movie_id, score in scores.items()
        }
        self.popular_items = self.ranked_items_
        self.score_by_item = self.scores_

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        seen: set[int] | None = None,
    ) -> list[int]:
        """Trả tối đa ``n`` item xếp hạng toàn cục, loại bỏ item đã xem."""
        if self.ranked_items_ is None:
            raise RuntimeError("Call fit() first")
        if n <= 0:
            return []
        if not seen:
            return self.ranked_items_[:n]
        return [
            movie_id
            for movie_id in self.ranked_items_
            if movie_id not in seen
        ][:n]


def _validate_ratings(train_df: pd.DataFrame) -> None:
    required_columns = {"movieId", "rating"}
    missing_columns = required_columns.difference(train_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"train_df is missing required columns: {missing}"
        )
    if train_df.empty:
        raise ValueError("train_df must not be empty")
```

**Vì sao cần base chung:** cả ba lớp dùng cùng cách sort, cùng logic lọc `seen`, cùng lưu alias `popular_items`/`score_by_item`. Tránh lặp code và đảm bảo hành vi giống nhau.

### 3.2. A — Bayesian weighted rating

```python
class BayesianPopularityRecommender(_RankedRecommender):
    """IMDb-style weighted rating, co về global mean.

    Prior mặc định là mean số rating trên mỗi item (trong tập train).
    """

    def __init__(self, m: float | None = None) -> None:
        super().__init__()
        if m is not None and m <= 0:
            raise ValueError("m must be greater than zero")
        self.m = m

    def fit(self, train_df: pd.DataFrame) -> "BayesianPopularityRecommender":
        _validate_ratings(train_df)
        item_stats = train_df.groupby("movieId")["rating"].agg(
            ["count", "mean"]
        )
        votes = item_stats["count"].astype(float)
        item_mean = item_stats["mean"].astype(float)
        global_mean = float(train_df["rating"].mean())
        prior_strength = (
            float(self.m) if self.m is not None else float(votes.mean())
        )
        scores = (votes / (votes + prior_strength)) * item_mean
        scores += (prior_strength / (votes + prior_strength)) * global_mean
        self._set_ranking(scores)
        return self
```

**Đọc nhanh:** `votes` đếm số rating; `item_mean` là trung bình rating cụm theo item; `global_mean` là C; `prior_strength` ưu tiên `self.m` nếu người dùng truyền vào, ngược lại dùng `mean(votes)`. Công thức chính xác là `WR = (v/(v+m))·R + (m/(v+m))·C`.

### 3.3. C — Wilson lower bound

```python
class WilsonLowerBoundRecommender(_RankedRecommender):
    """Wilson 95% lower bound của tỉ lệ rating dương (`rating >= 4`)."""

    def __init__(self, z: float = 1.96) -> None:
        super().__init__()
        if z <= 0:
            raise ValueError("z must be greater than zero")
        self.z = float(z)

    def fit(self, train_df: pd.DataFrame) -> "WilsonLowerBoundRecommender":
        _validate_ratings(train_df)
        votes = train_df.groupby("movieId").size().astype(float)
        positives = (
            (train_df["rating"] >= 4)
            .groupby(train_df["movieId"])
            .sum()
        )
        positive_rate = (
            positives.reindex(votes.index, fill_value=0).astype(float)
            / votes
        )
        z_squared = self.z ** 2
        denominator = 1.0 + z_squared / votes
        centre = positive_rate + z_squared / (2.0 * votes)
        margin = self.z * (
            positive_rate * (1.0 - positive_rate) / votes
            + z_squared / (4.0 * votes ** 2)
        ).pow(0.5)
        scores = (
            (centre - margin) / denominator
        ).clip(lower=0.0, upper=1.0)
        self._set_ranking(scores)
        return self
```

**Đọc nhanh:**
- `votes` đếm trên toàn DataFrame.
- `positives` đếm nhanh qua mặt nạ boolean rồi `groupby(movieId).sum()` — nhanh hơn `groupby.apply` trên ML-25M.
- `positive_rate` được `reindex` theo `votes` để tránh thiếu item chỉ có rating âm.
- `clip(0, 1)` chống underflow khi `(centre - margin)` âm nhỏ.

### 3.4. PopularityRecommender (count) — hoàn thiện

```python
class PopularityRecommender(_RankedRecommender):
    """Xếp hạng item theo số lượt rating, loại bỏ item đã xem."""

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        _validate_ratings(train_df)
        counts = train_df.groupby("movieId").size().astype(float)
        self._set_ranking(counts)
        return self
```

**Khác biệt so với stub trước:** thêm validation, dùng `groupby("movieId").size()` rõ ràng, kế thừa sort + lọc `seen` từ base.

---

## 4. Snippet test (`tests/test_baselines_popularity.py`)

Bộ test thuần pandas + pytest, không phụ thuộc dữ liệu thật.

```python
import math

import pandas as pd
import pytest

from src.popularity_baselines import (
    BayesianPopularityRecommender,
    PopularityRecommender,
    WilsonLowerBoundRecommender,
)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "movieId": [10, 20, 10, 30, 10, 20, 20, 30, 10, 30],
            "rating": [5, 3, 4, 5, 2, 4, 5, 1, 5, 4],
        }
    )


_models = [PopularityRecommender(), BayesianPopularityRecommender(), WilsonLowerBoundRecommender()]
_model_classes = [PopularityRecommender, BayesianPopularityRecommender, WilsonLowerBoundRecommender]


@pytest.mark.parametrize("model", _models)
def test_recommend_before_fit_raises(model):
    with pytest.raises(RuntimeError, match=r"Call fit\(\) first"):
        model.recommend(1)


@pytest.mark.parametrize("model", _models)
def test_fit_returns_self_and_recommendations_are_movie_ids(model, toy_df):
    assert model.fit(toy_df) is model
    recs = model.recommend(1, n=2)
    assert len(recs) == 2
    assert all(isinstance(m, int) for m in recs)


@pytest.mark.parametrize("model", _models)
def test_recommend_excludes_seen_items(model, toy_df):
    recs = model.fit(toy_df).recommend(99, n=5, seen={10, 20})
    assert not (set(recs) & {10, 20})


@pytest.mark.parametrize("model", _models)
def test_recommend_handles_non_positive_and_oversized_n(model, toy_df):
    model.fit(toy_df)
    assert model.recommend(1, n=0) == []
    assert model.recommend(1, n=-1) == []
    assert model.recommend(1, n=10) == model.ranked_items_
    assert model.recommend(1, n=2, seen=None) == model.recommend(1, n=2, seen=set())


@pytest.mark.parametrize("model_class", _model_classes)
def test_refit_replaces_existing_ranking(model_class, toy_df):
    model = model_class().fit(toy_df)
    refit = pd.DataFrame({"movieId": [99, 99, 98], "rating": [5.0, 5.0, 1.0]})
    model.fit(refit)
    assert model.ranked_items_ == [99, 98]


def test_popularity_ranks_by_count_and_keeps_compatibility_attributes(toy_df):
    model = PopularityRecommender().fit(toy_df)
    assert model.recommend(1, n=3) == [10, 20, 30]
    assert model.popular_items == model.ranked_items_
    assert model.score_by_item == model.scores_


def test_bayesian_score_matches_weighted_rating():
    train_df = pd.DataFrame({"movieId": [1, 1, 2], "rating": [4.0, 4.0, 5.0]})
    model = BayesianPopularityRecommender(m=2).fit(train_df)
    global_mean = 13 / 3
    expected = (2 / 4) * 4.0 + (2 / 4) * global_mean
    assert math.isclose(model.scores_[1], expected)


def test_bayesian_shrinkage_preserves_high_quality_high_volume_item():
    train_df = pd.DataFrame(
        {"movieId": [100] * 100 + [200], "rating": [4.9] * 100 + [3.0]}
    )
    model = BayesianPopularityRecommender(m=50).fit(train_df)
    assert model.recommend(1, n=1) == [100]


def test_bayesian_default_prior_is_mean_votes_per_item():
    train_df = pd.DataFrame(
        {"movieId": [1, 1, 1, 1, 2], "rating": [4.0, 4.0, 4.0, 4.0, 5.0]}
    )
    model = BayesianPopularityRecommender().fit(train_df)
    global_mean = 21 / 5
    prior_strength = 5 / 2  # mean votes per item
    expected = (1 / (1 + prior_strength)) * 5.0
    expected += (prior_strength / (1 + prior_strength)) * global_mean
    assert math.isclose(model.scores_[2], expected)


def test_bayesian_tie_breaks_by_movie_id():
    train_df = pd.DataFrame({"movieId": [20, 10], "rating": [4.0, 4.0]})
    assert (
            BayesianPopularityRecommender().fit(train_df).recommend(1, n=2)
            == [10, 20]
    )


def test_wilson_scores_are_bounded_and_zero_without_positive_ratings():
    train_df = pd.DataFrame(
        {"movieId": [1, 1, 2, 2], "rating": [1.0, 2.0, 4.0, 5.0]}
    )
    model = WilsonLowerBoundRecommender().fit(train_df)
    assert model.scores_[1] == 0.0
    assert all(0.0 <= s <= 1.0 for s in model.scores_.values())


def test_wilson_prefers_more_evidence_at_equal_positive_rate():
    train_df = pd.DataFrame(
        {"movieId": [1] * 10 + [2], "rating": [5.0] * 11}
    )
    model = WilsonLowerBoundRecommender().fit(train_df)
    assert model.recommend(1, n=2) == [1, 2]


@pytest.mark.parametrize("model_class", _model_classes)
def test_fit_is_deterministic(model_class, toy_df):
    first = model_class().fit(toy_df).ranked_items_
    second = model_class().fit(toy_df).ranked_items_
    assert first == second


@pytest.mark.parametrize("model_class", _model_classes)
def test_invalid_input_is_rejected(model_class):
    with pytest.raises(ValueError, match="movieId"):
        model_class().fit(pd.DataFrame({"rating": [5.0]}))
    with pytest.raises(ValueError, match="must not be empty"):
        model_class().fit(pd.DataFrame({"movieId": [], "rating": []}))
```

**Ý nghĩa từng nhóm test:**
- **Contract chung** (parametrize): `RuntimeError` trước `fit`, `fit → self`, `recommend` trả `list[int]`, lọc `seen`, xử lý `n <= 0` / `n` vượt kích thước / `seen=None` ≡ `seen=set()`, refit thay thế ranking cũ.
- **Popularity**: xếp hạng đúng theo count; alias `popular_items`/`score_by_item` trùng với `ranked_items_`/`scores_`.
- **Bayesian**: công thức match closed-form (`m=2`); item chất lượng cao + nhiều vote thắng item ít vote dù mean cao; default `m = votes.mean()` pin đúng closed-form; tie-break theo `movieId`.
- **Wilson**: score bounded `[0,1]`; item không có rating dương → `0`; cùng tỉ lệ positive thì nhiều vote hơn thắng.
- **Determinism + validate**: hai lần `fit` cùng kết quả; thiếu cột hoặc DataFrame rỗng → `ValueError`.

---

## 5. Câu lệnh chạy & kiểm tra nhanh

```bash
# 1) Smoke-test import
python -c "from src.baselines import PopularityRecommender, BayesianPopularityRecommender, WilsonLowerBoundRecommender; print('imports OK')"

# 2) Targeted unit tests
pytest -q tests/test_baselines_popularity.py
# Kết quả: 28 passed (trên .venv/bin/python3 host này)
```

---

## 6. Bảng chỉ số chất lượng đạt được

| AC | Mô tả | Bằng chứng kiểm thử |
|---|---|---|
| AC-1 | Hợp đồng `fit`/`recommend`, `RuntimeError` trước `fit`, độ dài `≤ n` | `test_recommend_before_fit_raises`, `test_fit_returns_self_and_recommendations_are_movie_ids`, `test_recommend_handles_non_positive_and_oversized_n` |
| AC-2 | Bayesian: giới hạn `v → ∞` → `R`, `v → 0` → `C`; chất lượng cao + nhiều vote thắng | `test_bayesian_score_matches_weighted_rating`, `test_bayesian_shrinkage_preserves_high_quality_high_volume_item`, `test_bayesian_default_prior_is_mean_votes_per_item`, `test_bayesian_tie_breaks_by_movie_id` |
| AC-3 | Wilson: bounded `[0,1]`, `= 0` khi không có rating dương, ranking không tăng | `test_wilson_scores_are_bounded_and_zero_without_positive_ratings`, `test_wilson_prefers_more_evidence_at_equal_positive_rate` |
| AC-4 | Determinism: hai `fit` cho cùng `ranked_items_` | `test_fit_is_deterministic` (parametrize 3 lớp) |
| AC-5 | `seen` lọc đúng các item đã xem | `test_recommend_excludes_seen_items`, `test_recommend_handles_non_positive_and_oversized_n` (kèm refit) |
| AC-6 | Backward-compat với `PopularityRecommender` cũ (qua alias) | `test_popularity_ranks_by_count_and_keeps_compatibility_attributes` |
| AC-7 | Validate input thiếu cột / DataFrame rỗng → `ValueError` | `test_invalid_input_is_rejected` |

---

## 7. Hạn chế & bước tiếp theo

- `src/evaluation.py` vẫn rỗng — chưa có pipeline HR@K / NDCG@K để chấm 3 baseline trên split chuẩn.
- Chưa benchmark end-to-end trên MovieLens 1M/25M; test hiện dùng fixture toy để xác nhận contract thuật toán, chưa đo chất lượng ranking thực tế.
- PopularityRecommender (count) vẫn phản ánh popularity bias — đây là baseline tham chiếu; Bayesian và Wilson bổ sung signal chất lượng để giảm thiểu hiện tượng này.
- Cả ba baseline non-personalized; cần kết hợp signal cá nhân hoá (collaborative filtering, content) ở phase sau để cải thiện recall người dùng cold-start.

---

**Status:** DONE
**Summary:** Tổng hợp A (Bayesian IMDb WR với default `m = mean(votes)`) và C (Wilson lower bound `z = 1.96` cho `rating ≥ 4`) cùng baseline count đã hoàn thiện trên hợp đồng `fit`/`recommend` chung; 28 pytest passed.
**Concerns/Blockers:** `src/evaluation.py` rỗng — chưa đo chất lượng ranking trên split chuẩn; chưa có kết quả benchmark end-to-end trên MovieLens thật.
