# Hướng Dẫn Triển Khai CF – Baseline → Neighborhood → ALS

Tài liệu này trình bày một lộ trình Collaborative Filtering tăng dần độ khó cho **MovieLens 25M** (dataset đang có trong `data/raw/ml-25m/`):

1. Phần 0 — Chuẩn bị dữ liệu dùng chung.
2. Phần 1 — Popularity baseline (mốc so sánh tối thiểu).
3. Phần 2 — User-based Collaborative Filtering.
4. Phần 3 — Item-based Collaborative Filtering.
5. Phần 4 — Implicit ALS (advanced CF).
6. Phần 5 — So sánh, chọn mô hình và case study.
7. Phụ lục ngắn — Cheatsheet, lỗi thường gặp, tham khảo.

Mọi thuật toán dùng chung một train/test split, một giao diện `fit/recommend` và cùng bộ metric Precision@K, Recall@K, NDCG@K. Không ghi sẵn số liệu mẫu: mọi ô trong bảng kết quả phải điền từ thực nghiệm.

## Nguyên tắc đánh giá

- Ba baseline neighborhood dùng **explicit rating** (giá trị 1–5) của train set.
- Implicit ALS dùng **positive interactions** (rating ≥ 4) và confidence weighting, đánh giá trên cùng metric ranking.
- Test = warm users/items đã thấy trong train; cold-start được lọc trước khi đánh giá.
- Khi tính Top-N phải loại phim user đã xem trong train.
- Báo cáo luôn ghi seed, version package và thời gian train.

---



# Phần 0 — Chuẩn bị dữ liệu dùng chung



## 0.1 Yêu cầu môi trường

Tệp `requirements.txt` đã ghim các phiên bản tương thích với `implicit==0.7.3` (xem repo). Nếu tạo venv riêng cho bài này:

```bash
python3 -m venv .venv-cf
source .venv-cf/bin/activate
pip install -U pip
pip install -r requirements.txt
```



## 0.2 Load và EDA tối thiểu

Một data loader phục vụ cho tất cả phần sau, đặt tại `src/data_loader.py` (chỉ giữ phần cần cho CF; Person 2 sẽ dùng loader riêng cho Content-based).

```python
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class ML25MLoader:
    def __init__(self, data_dir="data/raw/ml-25m"):
        self.data_dir = Path(data_dir)

    def load_ratings(self, sample_frac: float | None = None) -> pd.DataFrame:
        dtype = {"userId": "int32", "movieId": "int32",
                 "rating": "float16", "timestamp": "int32"}
        path = self.data_dir / "ratings.csv"
        if sample_frac is None:
            return pd.read_csv(path, dtype=dtype)
        n = sum(1 for _ in open(path)) - 1
        skip = np.random.choice(range(1, n + 1),
                                size=int(n * (1 - sample_frac)),
                                replace=False)
        return pd.read_csv(path, dtype=dtype, skiprows=skip)

    def load_movies(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "movies.csv")


class DataPreprocessor:
    @staticmethod
    def temporal_split(ratings: pd.DataFrame, train_ratio: float = 0.7):
        ordered = ratings.sort_values("timestamp").reset_index(drop=True)
        idx = int(len(ordered) * train_ratio)
        return ordered.iloc[:idx].copy(), ordered.iloc[idx:].copy()

    @staticmethod
    def filter_cold_start(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        users = set(train["userId"].unique())
        movies = set(train["movieId"].unique())
        warm = test[test["userId"].isin(users) & test["movieId"].isin(movies)].copy()
        removed = len(test) - len(warm)
        print(f"Filtered cold-start: {removed:,} interactions removed")
        return warm

    @staticmethod
    def to_implicit(ratings: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
        implicit = ratings[ratings["rating"] >= threshold].copy()
        print(f"Implicit conversion: {len(implicit):,} / {len(ratings):,} "
              f"({len(implicit) / len(ratings) * 100:.1f}%) positive")
        return implicit

    @staticmethod
    def build_user_item_matrix(interactions: pd.DataFrame,
                               value_column: str | None = None,
                               confidence_alpha: float | None = None):
        """Build a user-by-item CSR matrix.

        Exactly ONE of ``value_column`` / ``confidence_alpha`` may be supplied
        (passing both raises ``ValueError``).

        Three branches:

        * ``value_column="rating"``     — preserve explicit values (1.0–5.0),
          used for Popularity, User-CF (weighted cosine) and Item-CF
          (adjusted cosine requires mean-centering by rating).
        * ``confidence_alpha=1.0``      — cell = ``1 + α · rating`` (Hu/Koren/Volinsky
          2008), used for Implicit ALS.
        * Both ``None``                — every cell = 1.0 (binary "seen"), used for
          any model that only needs presence (e.g. BPR).
        """
        user_ids = np.sort(interactions["userId"].unique())
        movie_ids = np.sort(interactions["movieId"].unique())
        user_to_idx = {u: i for i, u in enumerate(user_ids)}
        movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

        df = interactions.assign(
            u_idx=interactions["userId"].map(user_to_idx),
            m_idx=interactions["movieId"].map(movie_to_idx),
        )
        if value_column is not None and confidence_alpha is not None:
            raise ValueError("Use value_column OR confidence_alpha, not both")
        if confidence_alpha is not None:
            data = (1.0 + confidence_alpha * df["rating"]).astype(np.float32)
        elif value_column is not None:
            data = df[value_column].astype(np.float32)
        else:
            data = np.ones(len(df), dtype=np.float32)
        matrix = csr_matrix((data, (df["u_idx"], df["m_idx"])),
                            shape=(len(user_ids), len(movie_ids)))
        sparsity = 100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        print(f"Matrix {matrix.shape}; nnz={matrix.nnz:,}; sparsity={sparsity:.4f}%")
        return matrix, dict(user_to_idx), dict(movie_to_idx)
```

EDA tối thiểu (giữ để chèn vào báo cáo) chỉ in 4 số liệu: tổng rating, số user, số phim, khoảng thời gian, tỉ lệ rating ≥ 4.

```python
loader = ML25MLoader()
ratings = loader.load_ratings()
print(len(ratings), ratings["userId"].nunique(), ratings["movieId"].nunique())
print(pd.to_datetime(ratings["timestamp"], unit="s").min(),
      pd.to_datetime(ratings["timestamp"], unit="s").max())
print((ratings["rating"] >= 4).mean())
```



## 0.3 Temporal split, cold-start filter, ID mapping

```python
from scipy.sparse import save_npz
from src.data_loader import save_processed_data

pre = DataPreprocessor()
train_raw, test_raw = pre.temporal_split(ratings, train_ratio=0.7)
test_warm = pre.filter_cold_start(train_raw, test_raw)

# Ma trận explicit cho Popularity, User-CF và Item-CF.
explicit_user_item_csr, explicit_user_map, explicit_movie_map = (
    pre.build_user_item_matrix(train_raw, value_column="rating")
)

# Ma trận implicit/confidence riêng cho ALS.
train_implicit = pre.to_implicit(train_raw, threshold=4.0)
test_implicit = pre.to_implicit(test_warm, threshold=4.0)
implicit_user_item_csr, implicit_user_map, implicit_movie_map = (
    pre.build_user_item_matrix(train_implicit, confidence_alpha=1.0)
)

save_processed_data(train_raw, test_warm)
save_npz("data/processed/train_implicit_user_item.npz", implicit_user_item_csr)
```

`user_item_csr` có hàng = user, cột = movie. Trong CSR: `data`/`indices`/`indptr`; `indptr[u]` là vị trí bắt đầu dòng `u` trong `data`/`indices`, và `indptr[n_rows] == len(data)` là vị trí **sau** phần tử cuối (không phải trỏ tới `data[len(data)]`). Đây là điểm hay gây nhầm: nên luôn cắt `data[indptr[u]:indptr[u + 1]]` để lấy dòng `u`.

## 0.4 Kiểm tra sau Phần 0

- Có `data/processed/train_ml25m.csv`, `test_ml25m.csv` và `train_implicit_user_item.npz`.
- Có hai matrix riêng: `explicit_user_item_csr` cho baseline/neighborhood và `implicit_user_item_csr` cho ALS.
- Không có cold-start user/item trong `test_warm`.
- `implicit_user_item_csr` có sparsity khoảng 99.7% (theo kinh nghiệm ML-25M sau khi lọc positive).

---



# Phần 1 — Popularity baseline

Baseline không cá nhân hoá; mục đích là chứng minh rằng personalization có giá trị. Lưu ở `src/baselines.py`.

```python
import pandas as pd


class PopularityRecommender:
    """Same top-N for everyone, excluding items each user has already interacted with."""

    def __init__(self):
        self.popular_items: list[int] | None = None
        self.score_by_item: dict[int, float] | None = None

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        counts = train_df["movieId"].value_counts()
        self.popular_items = counts.index.tolist()
        self.score_by_item = counts.to_dict()
        return self

    def recommend(self, user_id: int, n: int = 10,
                  seen: set[int] | None = None) -> list[int]:
        if self.popular_items is None:
            raise RuntimeError("Call fit() first")
        if not seen:
            return self.popular_items[:n]
        return [m for m in self.popular_items if m not in seen][:n]
```

Kiểm tra tối thiểu:

```python
seen_by_user = train_raw.groupby("userId")["movieId"].agg(set).to_dict()
pop = PopularityRecommender().fit(train_raw)
sample_user = next(iter(seen_by_user))
recs = pop.recommend(sample_user, n=10, seen=seen_by_user[sample_user])
assert not (set(recs) & seen_by_user[sample_user]), "Overlap with history"
```

---



# Phần 2 — User-based Collaborative Filtering



## 2.1 Ý tưởng

Với mỗi user, tìm K láng giềng có rating giống nhau nhất (cosine trên rating vector thưa), rồi dự đoán rating cho phim chưa xem bằng trung bình trọng số similarity. Trade-off: trực quan nhưng tốn chi phí khi số user lớn; với ML-25M cần giới hạn computation/sample.

## 2.2 Chuẩn bị user–user similarity (cosine trên sparse)

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def normalize_user_vectors(user_item_csr: csr_matrix) -> csr_matrix:
    """Return L2-normalized explicit user-rating vectors without an N×N matrix."""
    return normalize(user_item_csr, norm="l2", axis=1).tocsr()
```



## 2.3 Lớp `UserCFRecommender`

```python
from dataclasses import dataclass


@dataclass
class _ModelState:
    normalized_user_item: csr_matrix
    user_item: csr_matrix
    user_map: dict[int, int]
    movie_map: dict[int, int]
    user_map_inv: dict[int, int]
    movie_map_inv: dict[int, int]
    top_k: int


class UserCFRecommender:
    """User-CF using cosine similarity between explicit user rating vectors."""

    def __init__(self, top_k: int = 50):
        self.top_k = top_k
        self.state: _ModelState | None = None

    def fit(self, user_item_csr: csr_matrix, user_map: dict[int, int],
            movie_map: dict[int, int]) -> "UserCFRecommender":
        self.state = _ModelState(
            normalized_user_item=normalize_user_vectors(user_item_csr),
            user_item=user_item_csr,
            user_map=user_map,
            movie_map=movie_map,
            user_map_inv={v: k for k, v in user_map.items()},
            movie_map_inv={v: k for k, v in movie_map.items()},
            top_k=self.top_k,
        )
        return self

    def recommend(self, user_id: int, n: int = 10,
                  seen: set[int] | None = None) -> list[tuple[int, float]]:
        if self.state is None:
            raise RuntimeError("Call fit() first")
        if user_id not in self.state.user_map:
            return []
        u_idx = self.state.user_map[user_id]
        similarities = (self.state.normalized_user_item @
                        self.state.normalized_user_item[u_idx].T).toarray().ravel()
        similarities[u_idx] = 0.0
        n_neighbours = min(self.state.top_k, len(similarities) - 1)
        if n_neighbours <= 0:
            return []
        neighbour_idx = np.argpartition(similarities, -n_neighbours)[-n_neighbours:]
        weights = similarities[neighbour_idx]
        positive = weights > 0
        neighbour_idx, weights = neighbour_idx[positive], weights[positive]
        if len(neighbour_idx) == 0:
            return []
        score = np.asarray(weights @ self.state.user_item[neighbour_idx]).ravel()
        score /= np.abs(weights).sum() + 1e-9
        if seen:
            mask = np.ones(score.shape, dtype=bool)
            for m in seen:
                if m in self.state.movie_map:
                    mask[self.state.movie_map[m]] = False
            score = np.where(mask, score, -np.inf)
        top = np.argsort(-score)[:n]
        return [(self.state.movie_map_inv[int(j)], float(score[j])) for j in top]
```

Kiểm tra nhanh:

```python
user_cf = UserCFRecommender(top_k=50).fit(
    explicit_user_item_csr, explicit_user_map, explicit_movie_map
)
sample_user = next(iter(seen_by_user))
print(user_cf.recommend(sample_user, n=10, seen=seen_by_user[sample_user]))
```



## 2.4 Lưu ý thực nghiệm

- Cosine không chuẩn hoá lệch rating trung bình của từng user; nếu cần, dùng Pearson bằng cách trừ `rating - user_mean` trước khi `normalize`.
- Với mỗi user test, lấy `seen` từ train_raw (không lẫn test) để tránh leak.
- Với ML-25M, phép nhân `normalized @ normalized[u_idx].T` trên toàn ma trận yêu cầu RAM đủ; nếu vượt ngưỡng, có thể subsample top-active users hoặc dùng approximate nearest neighbour.

---



# Phần 3 — Item-based Collaborative Filtering



## 3.1 Ý tưởng

Item similarity ổn định hơn user similarity vì thuộc tính phim thay đổi chậm; có thể precompute. Item-CF tập trung vào việc tổng hợp điểm từ các phim user đã thích (theo trọng số cosine giữa item vectors).

## 3.2 `ItemCFRecommender`

```python
class ItemCFRecommender:
    """Item-CF score items on demand; never materialize a full item×item matrix."""

    def __init__(self):
        self.normalized_item_user: csr_matrix | None = None
        self.user_item: csr_matrix | None = None
        self.user_map: dict[int, int] | None = None
        self.movie_map: dict[int, int] | None = None
        self.movie_map_inv: dict[int, int] | None = None

    def fit(self, user_item_csr: csr_matrix, user_map: dict[int, int],
            movie_map: dict[int, int]) -> "ItemCFRecommender":
        self.normalized_item_user = normalize(user_item_csr.T, norm="l2", axis=1).tocsr()
        self.user_item = user_item_csr
        self.user_map, self.movie_map = user_map, movie_map
        self.movie_map_inv = {v: k for k, v in movie_map.items()}
        return self

    def recommend(self, user_id: int, n: int = 10,
                  seen: set[int] | None = None) -> list[tuple[int, float]]:
        if self.normalized_item_user is None or user_id not in self.user_map:
            return []
        seen_indices = [self.movie_map[m] for m in (seen or []) if m in self.movie_map]
        if not seen_indices:
            return []
        profile = self.user_item[self.user_map[user_id], seen_indices].toarray().ravel()
        similarities = self.normalized_item_user @ self.normalized_item_user[seen_indices].T
        scores = np.asarray(similarities @ profile).ravel()
        scores[seen_indices] = -np.inf
        top = np.argsort(-scores)[:n]
        return [(self.movie_map_inv[int(i)], float(scores[i])) for i in top]
```

Kiểm tra nhanh:

```python
item_cf = ItemCFRecommender().fit(
    explicit_user_item_csr, explicit_user_map, explicit_movie_map
)
print(item_cf.recommend(sample_user, n=10, seen=seen_by_user[sample_user]))
```

Lớp này chỉ giữ `item × user` đã normalize. Khi recommend, nó tính block similarity từ các phim trong lịch sử user đến toàn catalogue; không tạo hay lưu ma trận `item × item` đầy đủ.

---



# Phần 4 — Implicit ALS



## 4.1 Ý tưởng

ALS huấn luyện trên **positive interactions** (rating ≥ 4) với confidence `c_ui = 1 + α * rating`. Khi gọi `fit`, thư viện `implicit` nhận ma trận **user × item** (CSR); phương thức `recommend` chấp nhận `user_items` là dòng thưa của user để biết phim đã thích.

## 4.2 Một lớp duy nhất

```python
from implicit.als import AlternatingLeastSquares


class ALSRecommender:
    def __init__(self, factors: int = 64, regularization: float = 0.01,
                 iterations: int = 15, alpha: float = 1.0,
                 random_state: int = 42):
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
            calculate_training_loss=True,
        )
        self.alpha = alpha
        self.user_map: dict[int, int] | None = None
        self.movie_map: dict[int, int] | None = None
        self.user_map_inv: dict[int, int] | None = None
        self.movie_map_inv: dict[int, int] | None = None
        self.train_matrix: csr_matrix | None = None
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, train_implicit: pd.DataFrame) -> "ALSRecommender":
        matrix, user_map, movie_map = DataPreprocessor.build_user_item_matrix(
            train_implicit, confidence_alpha=self.alpha
        )
        self.train_matrix = matrix
        self.user_map = user_map
        self.movie_map = movie_map
        self.user_map_inv = {v: k for k, v in user_map.items()}
        self.movie_map_inv = {v: k for k, v in movie_map.items()}
        self.model.fit(matrix)
        return self

    def recommend(self, user_id: int, n: int = 10,
                  seen: set[int] | None = None) -> list[tuple[int, float]]:
        """`seen` is accepted for the shared evaluator; filtering uses train_matrix."""
        if self.train_matrix is None:
            raise RuntimeError("Call fit() first")
        if user_id not in self.user_map:
            return []
        u_idx = self.user_map[user_id]
        ids, scores = self.model.recommend(
            userid=u_idx,
            user_items=self.train_matrix[u_idx],
            N=n,
            filter_already_liked_items=True,
        )
        return [(self.movie_map_inv[int(i)], float(s)) for i, s in zip(ids, scores)]

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "alpha": self.alpha,
                "factors": self.factors,
                "regularization": self.regularization,
                "iterations": self.iterations,
                "user_map": self.user_map,
                "movie_map": self.movie_map,
                "train_matrix": self.train_matrix,
            }, f)

    @classmethod
    def load(cls, path: str) -> "ALSRecommender":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(factors=data["factors"], regularization=data["regularization"],
                  iterations=data["iterations"], alpha=data["alpha"])
        obj.model = data["model"]
        obj.user_map = data["user_map"]
        obj.movie_map = data["movie_map"]
        obj.train_matrix = data.get("train_matrix")
        obj.user_map_inv = {v: k for k, v in obj.user_map.items()}
        obj.movie_map_inv = {v: k for k, v in obj.movie_map.items()}
        return obj
```



## 4.3 Tuning tối thiểu

- Bắt đầu với `factors ∈ {32, 64, 128}`, `regularization ∈ {0.01, 0.1}`, `iterations ∈ {10, 15, 25}`, `alpha = 1.0`.
- Sau khi pipeline đánh giá chạy đúng, sweep từng biến một; luôn ghi seed, thời gian train và metric ở cùng K.
- Không tăng `iterations` vô hạn — diminishing return sau ~20 vòng trên ML-25M.

---



# Phần 4b — Bayesian Personalized Ranking (BPR)

**Mục tiêu:** Bổ sung BPR như một hướng tiếp cận Matrix Factorization khác cho implicit feedback, để so sánh với ALS.

## 4b.1 Động lực

ALS và BPR đều là Matrix Factorization cho implicit feedback, nhưng optimize objectives khác nhau:

- **ALS:** Tối ưu **weighted reconstruction** - minimize weighted squared error trên toàn bộ ma trận
  ```
  min Σ c_ui × (p_ui - x_ui)²
  ```
  
- **BPR:** Tối ưu **pairwise ranking** - item positive phải có score cao hơn item unseen
  ```
  max Σ log σ(x_ui - x_uj)  với i = positive, j = negative
  ```

**Trade-offs:**
- BPR optimize trực tiếp cho ranking → phù hợp khi metric chính là HitRate/NDCG
- ALS converge nhanh hơn, dễ parallelize → phù hợp cho production scale
- BPR không dùng confidence weighting → phù hợp khi chỉ có binary interactions

## 4b.2 Lý thuyết ngắn gọn

**Core Idea:** Với user `u`, nếu item `i` là positive (đã tương tác) và item `j` là negative (chưa tương tác), BPR muốn:

```
score(u, i) > score(u, j)
```

BPR học bằng cách:
1. Sample triplets `(u, i, j)` trong đó `i` ∈ positive items, `j` ∈ unseen items
2. Update embeddings để maximize log likelihood của `x_ui > x_uj`
3. Sử dụng stochastic gradient descent với negative sampling

**Assumptions:**
- Items chưa tương tác không nhất thiết là "dislike" - chỉ là "chưa biết"
- Training optimize thứ tự tương đối, không phải absolute scores
- Evaluation dùng ranking metrics (HitRate, NDCG), không phải rating prediction (RMSE)

**Chi tiết lý thuyết:** Xem `notes/advanced-cf-bpr-svd-svdpp-ml1m-guide.md` Section 3.

## 4b.3 Implementation

### Notebook: `notebooks/03-bpr-experiments.ipynb`

**Full implementation** bao gồm:
1. Temporal leave-one-positive-out split (khác với temporal split của ALS)
2. Binary sparse matrix (không dùng confidence weighting)
3. BPR training với `implicit.bpr.BayesianPersonalizedRanking`
4. Sampled ranking evaluation (1 positive + 99 negatives)
5. ALS retrain với cùng protocol để fair comparison
6. Comparison visualization

**Key differences từ ALS pipeline:**

| Aspect | ALS (Phần 4) | BPR (Phần 4b) |
|--------|--------------|---------------|
| **Split** | Temporal 70/30 trên all ratings | Leave-one-positive-out per user |
| **Matrix** | Confidence-weighted (`c = 1 + α×rating`) | Binary (1.0 if interaction exists) |
| **Evaluation** | Precision/Recall/NDCG trên full test set | HitRate/NDCG trên sampled candidates (1+99) |
| **Iterations** | 15 (ALS converges fast) | 100 (BPR needs more iterations) |

### Quick Start

```python
# Load and split data (temporal leave-one-positive-out)
from src.data_loader import ML25MLoader
loader = ML25MLoader()
ratings = loader.load_ratings(sample_frac=0.1)  # 10% sample for testing

train_bpr, test_positive = temporal_leave_one_positive_out(
    ratings, threshold=4.0
)

# Build binary matrix
user_item_bpr = build_binary_matrix(train_bpr)  # No confidence weighting

# Train BPR
from implicit.bpr import BayesianPersonalizedRanking
bpr = BayesianPersonalizedRanking(
    factors=64, learning_rate=0.05, regularization=0.01,
    iterations=100, random_state=42
)
bpr.fit(user_item_bpr)

# Evaluate with sampled ranking
candidates = build_ranking_candidates(train_bpr, test_positive, item_ids, negatives=99)
results = evaluate_bpr_at_k(bpr, candidates, k=10)
print(f"HitRate@10: {results['hit_rate']:.4f}")
print(f"NDCG@10: {results['ndcg']:.4f}")
```

**Full code:** Xem notebook để có implementation chi tiết các functions.

## 4b.4 Kết quả (ML-25M)

**Protocol:**
- Dataset: MovieLens 25M (10% sample cho testing, full run overnight)
- Split: Temporal leave-one-positive-out
- Evaluation: Sampled ranking (1 positive + 99 negatives)
- Metrics: HitRate@K, NDCG@K (K=5,10,20)
- Random seed: 42

**Results:** Xem `notes/bpr-ml25m-report.md` để có:
- Bảng comparison đầy đủ BPR vs ALS
- Visualization (bar charts)
- Training time comparison
- Detailed analysis

**Expected observations:**
- BPR thường có NDCG cao hơn (optimize trực tiếp cho ranking)
- ALS train nhanh hơn (~7x với 100 vs 15 iterations)
- Trade-off: ranking quality vs training efficiency

## 4b.5 Khi nào dùng BPR vs ALS?

### Chọn BPR khi:
- ✅ Priority là **ranking quality** (NDCG, HitRate)
- ✅ Có đủ thời gian train (BPR chậm hơn ALS)
- ✅ Chỉ có **binary interactions** (viewed/not viewed, clicked/not clicked)
- ✅ Evaluation metric là ranking-based

### Chọn ALS khi:
- ✅ Priority là **training speed** (production deployment)
- ✅ Có **confidence signal** (rating magnitude, view count)
- ✅ Cần **scale lớn** (ALS dễ parallelize hơn)
- ✅ Cần **deterministic convergence** (ALS ổn định hơn SGD)

### Recommendation cho Production:
1. **Development:** Thử cả BPR và ALS trên validation set
2. **Comparison:** So sánh cả ranking metrics VÀ training/inference time
3. **A/B testing:** Deploy cả hai, đo online metrics (CTR, conversion)
4. **Hybrid:** Có thể ensemble predictions của BPR và ALS

## 4b.6 Limitations & Future Work

**Current limitations:**
- Sampled ranking chỉ evaluate 99 negatives, không phải full catalogue
- Chưa hyperparameter tuning comprehensive (chỉ dùng default configs)
- Cold-start vẫn là vấn đề (giống ALS)

**Future improvements:**
1. **Full-catalogue ranking:** Evaluate trên toàn bộ items (chậm nhưng chính xác hơn)
2. **Hyperparameter tuning:** Grid search learning_rate, iterations, factors
3. **Alternative objectives:** Thử WARP (Weighted Approximate-Rank Pairwise)
4. **Hybrid with content:** Kết hợp BPR embeddings với content features
5. **Inference optimization:** ANN (Approximate Nearest Neighbors) cho fast retrieval

---



# Phần 5 — Đánh giá, so sánh và case study



## 5.1 Evaluator dùng chung

Đặt tại `src/evaluation.py`. Giao diện thống nhất `evaluate_model(model, test_df, train_df, k_values)`.

```python
import numpy as np
import pandas as pd


def _recommend(model, user_id, n, seen):
    if hasattr(model, "recommend"):
        result = model.recommend(user_id, n=n, seen=seen)
        if isinstance(result, list) and result and isinstance(result[0], tuple):
            return [item_id for item_id, _ in result]
        return list(result)
    raise TypeError(f"Model {type(model).__name__} chưa có phương thức recommend")


def precision_recall_ndcg(predictions, ground_truth, k):
    if not predictions or not ground_truth:
        return 0.0, 0.0, 0.0
    top = predictions[:k]
    hits = len(set(top) & set(ground_truth))
    precision = hits / k
    recall = hits / len(ground_truth)
    dcg = 0.0
    for rank, item in enumerate(top):
        if item in ground_truth:
            dcg += 1.0 / np.log2(rank + 2)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(ground_truth))))
    ndcg = dcg / idcg if idcg else 0.0
    return precision, recall, ndcg


def evaluate_model(model, test_df: pd.DataFrame, train_df: pd.DataFrame,
                  k_values=(5, 10, 20), n_rec: int = 50) -> dict[int, dict[str, float]]:
    seen_by_user = train_df.groupby("userId")["movieId"].agg(set).to_dict()
    truth_by_user = test_df.groupby("userId")["movieId"].agg(list).to_dict()
    common_users = [u for u in truth_by_user if u in seen_by_user]
    metrics = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    for user_id in common_users:
        seen = seen_by_user[user_id]
        preds = _recommend(model, user_id, n_rec, seen)
        truth = truth_by_user[user_id]
        for k in k_values:
            p, r, n = precision_recall_ndcg(preds, truth, k)
            metrics[k]["precision"].append(p)
            metrics[k]["recall"].append(r)
            metrics[k]["ndcg"].append(n)
    return {
        k: {m: float(np.mean(v)) for m, v in mvals.items()}
        for k, mvals in metrics.items()
    }
```



## 5.2 Chạy đánh giá

```python
from src.evaluation import evaluate_model

models = {
    "Popularity": pop,
    "UserCF": user_cf,
    "ItemCF": item_cf,
    "ALS": als,
}

results = {name: evaluate_model(model, test_warm, train_raw, k_values=(5, 10, 20))
           for name, model in models.items()}
```

Bảng kết quả (điền sau khi chạy):


| Model        | Precision@10 | Recall@10 | NDCG@10 | Train time (s) |
| ------------ | ------------ | --------- | ------- | -------------- |
| Popularity   |              |           |         |                |
| UserCF       |              |           |         |                |
| ItemCF       |              |           |         |                |
| Implicit ALS |              |           |         |                |




## 5.3 Case study

Với mỗi user, in top-5 phim đã rate (train) và top-5 khuyến nghị của từng model. Nhận xét: personalization, diversity, cold-start.

```python
movies = loader.load_movies()


def case_study(model, user_id, seen, k=5):
    history = (train_raw[train_raw["userId"] == user_id]
               .merge(movies[["movieId", "title"]], on="movieId")
               .nlargest(k, "rating"))
    preds = _recommend(model, user_id, k, seen)
    recs = movies[movies["movieId"].isin(preds)][["title"]]
    return history["title"].tolist(), recs["title"].tolist()
```

Quan sát cần nêu trong báo cáo:

- UserCF/ItemCF có xu hướng trùng popular items khi history ngắn.
- ALS ít trùng popularity baseline hơn, thể hiện personalization mạnh hơn.
- User/item hoàn toàn mới (cold-start) trả về danh sách rỗng; cần popularity fallback.

---



# Phụ lục ngắn



## A. Cheatsheet lệnh

```bash
# 1. Tải MovieLens 25M (xem link trong data/raw/ml-25m/README.txt)
# 2. Tiền xử lý + EDA
jupyter notebook notebooks/01-data-exploration.ipynb
# 3. Train và đánh giá
jupyter notebook notebooks/02-cf-experiments.ipynb
```



## B. Lỗi thường gặp

1. `MemoryError` khi tạo full item-item similarity — giới hạn `top_k` hoặc tính theo từng user thay vì toàn ma trận.
2. Kết quả Precision/Recall = 0 — kiểm tra user có trong cả `train_raw` và `test_warm`; kiểm tra `filter_already_liked_items=True` ở ALS.
3. `KeyError` ở `seen_by_user` — user test cold-start; bỏ qua trong evaluation (xử lý trong `evaluate_model`).
4. `implicit` import lỗi — cài `pip install implicit==0.7.3` đúng phiên bản đã pin trong `requirements.txt`.
5. Kết quả khác giữa các lần chạy — set `random_state=42` cho mọi thành phần; document seed trong báo cáo.



## C. Tham khảo

- Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*.
- Resnick, P. et al. (1994). *GroupLens: An Open Architecture for Collaborative Filtering of Netnews*.
- Sarwar, B. et al. (2001). *Item-based Collaborative Filtering Recommendation Algorithms*.
- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*.
- Tài liệu `implicit`: [https://benfred.github.io/implicit/](https://benfred.github.io/implicit/).

