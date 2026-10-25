# Hướng Dẫn Triển Khai Chi Tiết - Người 2 (Content-Based Filtering)

**Vai trò:** Content-Based Filtering + Evaluation Module + Comparison Analysis
**Timeline:** 6 tuần (~30 giờ)
**Output:** Working CB recommender + Evaluation framework + ~18 trang báo cáo
**Phối hợp:** Người 1 (Data + CF), Người Phụ A/B/C (Intro, Conclusion, Appendix)

---

## Mục Lục

1. [Mục Tiêu Tổng Quan](#1-mục-tiêu-tổng-quan)
2. [File Structure - Quyền Sở Hữu Của Người 2](#2-file-structure---quyền-sở-hữu-của-người-2)
3. [Tuần 1-2: Research + Data Understanding](#3-tuần-1-2-research--data-understanding)
4. [Tuần 3: Core Implementation - Content-Based](#4-tuần-3-core-implementation---content-based)
5. [Tuần 3: Evaluation Module](#5-tuần-3-evaluation-module)
6. [Tuần 4: Experiments Notebook](#6-tuần-4-experiments-notebook)
7. [Tuần 5: Report Writing](#7-tuần-5-report-writing)
8. [Tuần 6: Integration với Người 1](#8-tuần-6-integration-với-người-1)
9. [Expected Results](#9-expected-results)
10. [Key Insights - Strengths & Weaknesses CB](#10-key-insights---strengths--weaknesses-cb)
11. [To-Do List Summary](#11-to-do-list-summary)

---

## 1. Mục Tiêu Tổng Quan

### 1.1 Vai Trò Của Người 2

Người 2 chịu trách nhiệm xây dựng **Content-Based Filtering (CB)** recommendation system - phương pháp dựa trên đặc trưng nội tại của item (genres, tags, descriptions) để tìm ra những item tương tự mà người dùng có thể quan tâm.

**Tại sao cần Content-Based?**
- ✅ Giải quyết **cold-start problem** cho items mới (CF không làm được vì cần interaction history)
- ✅ **Explainable recommendations** - có thể giải thích "vì sao recommend movie X" (cùng genre, cùng tag)
- ✅ **Diversify** recommendations - tránh filter bubble của CF
- ✅ **Independent** của user count - chỉ cần features của item

### 1.2 Deliverables Chính

| Tuần | Deliverable | Mô Tả |
|------|-------------|-------|
| 1-2 | Theory notes | Lý thuyết CB, TF-IDF, cosine similarity, EDA genres |
| 3 | `src/content_based.py` | Class `ContentBasedRecommender` hoàn chỉnh |
| 3 | `src/evaluation.py` | Class `RecommenderEvaluator` với 6+ metrics |
| 4 | `04-content-based-experiments.ipynb` | Training + evaluation + visualizations |
| 5 | Report chapters | 2.3, 2.4, 4.4, 5.1, 5.4, lead Ch.6 (~18 trang) |
| 6 | Integration | Hybrid experiment với Người 1 |

### 1.3 Phần Báo Cáo Phụ Trách

| Chapter | Nội dung | Trang |
|---------|----------|-------|
| 2.3 | Content-based Filtering (theory) | 2 trang |
| 2.4 | Evaluation Metrics for Ranking | 1 trang |
| 4.4 | Content-based Methodology | 2 trang |
| 5.1 | Evaluation Protocol | 1 trang |
| 5.4 | Content-based Results | 2 trang |
| 6 | Comparison & Analysis (lead) | 4 trang |
| 7 | Discussion (collaborate) | ~1.5 trang |
| **Total** | | **~13.5 trang** |

---

## 2. File Structure - Quyền Sở Hữu Của Người 2

### 2.1 Files Người 2 Sở Hữu (đọc + sửa)

```
RecommendationSystem/
├── src/
│   ├── content_based.py           # ✅ NGƯỜI 2 - Core CB implementation
│   └── evaluation.py              # ✅ NGƯỜI 2 - Metrics & evaluator
├── notebooks/
│   └── 04-content-based-experiments.ipynb   # ✅ NGƯỜI 2 - Experiments
├── figures/
│   ├── cb_genre_distribution.png            # ✅ NGƯỜI 2
│   ├── cb_similar_movies.png                # ✅ NGƯỜI 2
│   ├── cb_metrics_vs_baseline.png           # ✅ NGƯỜI 2
│   ├── cb_cold_start_analysis.png           # ✅ NGƯỜI 2
│   ├── cb_metrics_vs_k.png                  # ✅ NGƯỜI 2
│   └── all_methods_comparison.png           # ✅ NGƯỜI 2 (lead)
├── results/
│   └── cb_metrics.json                      # ✅ NGƯỜI 2 - Final metrics
├── notes/
│   ├── person2-content-based-guide.md       # ✅ NGƯỜI 2 (this file)
│   └── cb-insights.md                       # ✅ NGƯỜI 2 - Personal notes
└── docs/
    └── hybrid-methods.md                    # ✅ NGƯỜI 2 (Week 6)
```

### 2.2 Files Đọc (KHÔNG sửa) - Của Người 1

```
src/data_loader.py                # 🔒 NGƯỜI 1 - Reuse ML25MLoader, DataPreprocessor
src/baselines.py                  # 🔒 NGƯỜI 1 - PopularityBaseline for comparison
src/collaborative_filtering.py    # 🔒 NGƯỜI 1 - ALSRecommender for comparison
notebooks/01-data_preprocessing.ipynb     # 🔒 NGƯỜI 1
notebooks/02-cf-experiments.ipynb         # 🔒 NGƯỜI 1
```

### 2.3 Quy Tắc Collaboration

- ✅ **Đọc thoải mái** files của Người 1 để hiểu data structure & API
- ✅ **Import** từ `src.data_loader`, `src.collaborative_filtering` trong code của Người 2
- ❌ **KHÔNG sửa** bất kỳ file nào của Người 1
- ❌ **KHÔNG tạo** file mới trong `src/` ngoài 2 file được phân công
- ✅ Nếu cần thêm utility → tạo trong `src/content_based_helpers.py` (file phụ của Người 2)
- ✅ Khi cần Người 1 thêm field/API → ping qua message (không tự sửa)

---

## 3. Tuần 1-2: Research + Data Understanding

**Timeline:** 6 giờ
**Output:** Theory notes + EDA genres + sample TF-IDF demo

### 3.1 Lý Thuyết Content-Based Filtering

#### 3.1.1 Khái Niệm Cốt Lõi

Content-Based Filtering recommend items dựa trên **đặc trưng nội tại (intrinsic features)** của item và **profile sở thích** của user được xây dựng từ lịch sử tương tác.

**Workflow tổng quan:**

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Item Features  │         │  User History    │         │  Recommendation │
│  (genres, tags) │────────▶│  (watched items) │────────▶│  (top-N similar)│
└─────────────────┘         └──────────────────┘         └─────────────────┘
        │                            │                            │
        ▼                            ▼                            ▼
   TF-IDF vector              Aggregate scores            Exclude seen
   (n_movies × n_genres)      (sum of similarities)       & return top-K
```

#### 3.1.2 TF-IDF (Term Frequency - Inverse Document Frequency)

**Intuition:** Genre nào xuất hiện nhiều trong 1 movie nhưng hiếm trong dataset → đó là đặc trưng quan trọng để phân biệt.

**Công thức:**

```
TF(t, d) = (số lần term t xuất hiện trong document d) / (tổng số terms trong d)

IDF(t) = log(N / df(t))
  - N = tổng số documents (movies)
  - df(t) = số documents chứa term t

TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Ví dụ cụ thể với MovieLens:**

```
Movie "Toy Story (1995)" → genres: "Adventure|Animation|Children|Comedy|Fantasy"

Term "Children" xuất hiện ở 1/59,047 movies → IDF cao
Term "Drama" xuất hiện ở ~20,000/59,047 movies → IDF thấp

→ Children là đặc trưng phân biệt tốt hơn Drama
```

**Implementation trong scikit-learn:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# genres là string "Adventure|Animation|Children|Comedy|Fantasy"
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')  # Split by '|'
tfidf_matrix = tfidf.fit_transform(movies['genres'])
# Shape: (n_movies, n_genres) - ví dụ (59047, 24)
```

#### 3.1.3 Cosine Similarity

**Intuition:** Đo góc giữa 2 vectors trong không gian nhiều chiều. Hai movies cùng genre → vectors gần nhau → similarity cao.

**Công thức:**

```
            A · B
cos(θ) = ─────────
         ||A|| × ||B||

A · B = tổng tích từng phần tử: Σ A[i] × B[i]
||A|| = sqrt(Σ A[i]²)
```

**Range:** [-1, 1] nhưng với TF-IDF non-negative → [0, 1]
- 1 = giống hệt nhau
- 0 = không liên quan

**Implementation:**

```python
from sklearn.metrics.pairwise import cosine_similarity

# Sparse matrix similarity - efficient với 59K movies
similarity_matrix = cosine_similarity(tfidf_matrix)
# Shape: (59047, 59047) - dense nhưng OK với 59K
# Memory: 59047² × 4 bytes ≈ 13.3 GB ❌ QUÁ LỚN

# GIẢI PHÁP: Chỉ compute khi cần (top-K retrieval)
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix[query_idx:query_idx+1])
# Shape: (n_movies, 1) - chỉ tính 1 vector vs all
```

**Hoặc dùng sparse từ đầu:**

```python
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF matrix đã là sparse CSR
tfidf_sparse = csr_matrix(tfidf_matrix)
# Cosine similarity giữa 2 sparse matrices → vẫn dense output
# Workaround: dùng sklearn.linear_kernel (faster cho sparse)
from sklearn.metrics.pairwise import linear_kernel

# linear_kernel(A, B) = A × B^T (treat như dense dot product)
# Nếu vectors đã L2-normalized → linear_kernel = cosine_similarity
```

#### 3.1.4 So Sánh CB với CF

| Tiêu chí | Content-Based | Collaborative Filtering |
|----------|---------------|-------------------------|
| **Input** | Item features (genres, tags) | User-item interactions |
| **Cold-start item** | ✅ OK (chỉ cần features) | ❌ Fail (không có interactions) |
| **Cold-start user** | ⚠️ Cần >=1 interaction để build profile | ❌ Fail |
| **Explainability** | ✅ "Vì cùng genre X" | ❌ Latent factors khó giải thích |
| **Diversity** | ✅ Thường diverse hơn | ⚠️ Filter bubble |
| **Serendipity** | ⚠️ Thấp (chỉ similar items) | ✅ Cao (discover patterns ẩn) |
| **Scalability** | ✅ Item features ít thay đổi | ⚠️ User-item matrix phình to |

### 3.2 Data Exploration Cho Content-Based

**Notebook:** Tạo `notebooks/03-data-exploration-cb.ipynb` (optional, cho practice)

#### 3.2.1 EDA Movies & Genres

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load movies (từ Người 1's ML25MLoader)
from src.data_loader import ML25MLoader

loader = ML25MLoader('data/raw/ml-25m')
movies = loader.load_movies()

print(f"Total movies: {len(movies):,}")
print(f"Columns: {movies.columns.tolist()}")
print(movies.head())

# Movies có genres là gì?
print(f"\nMovies with '(no genres listed)': {(movies['genres'] == '(no genres listed)').sum()}")
```

**Expected output:**

```
Total movies: 62,423
Columns: ['movieId', 'title', 'genres']

       movieId              title                       genres
0            1  Toy Story (1995)  Adventure|Animation|Children|...
1            2    Jumanji (1995)  Adventure|Children|Fantasy
...
```

#### 3.2.2 Phân Tích Genre Distribution

```python
# Tách genres thành list
movies['genre_list'] = movies['genres'].str.split('|')

# Đếm genre frequency
genre_counter = Counter()
for genres in movies['genre_list']:
    genre_counter.update(genres)

# Top 20 genres
genre_df = pd.DataFrame(
    genre_counter.most_common(20),
    columns=['Genre', 'Count']
)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
sns.barplot(data=genre_df, x='Count', y='Genre', ax=axes[0], palette='viridis')
axes[0].set_title('Top 20 Genres trong MovieLens 25M', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Movies')

# Percentage
genre_df['Percentage'] = genre_df['Count'] / len(movies) * 100
sns.barplot(data=genre_df, x='Percentage', y='Genre', ax=axes[1], palette='magma')
axes[1].set_title('Genre Coverage (% of Total Movies)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Percentage (%)')

plt.tight_layout()
plt.savefig('figures/cb_genre_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Expected result:**

```
Drama      xxxxxx (~20,000 movies)
Comedy     xxxxx  (~15,000)
Thriller   xxxx   (~10,000)
Romance    xxx    (~7,000)
Action     xxx    (~6,000)
...
(no genres listed) ~250 movies (~0.4%)
```

#### 3.2.3 Phân Tích Genre Co-occurrence

```python
# Số genres trung bình mỗi movie
movies['n_genres'] = movies['genre_list'].apply(len)
print(f"Avg genres per movie: {movies['n_genres'].mean():.2f}")
print(f"Distribution:\n{movies['n_genres'].value_counts().sort_index()}")

# Genre co-occurrence matrix
from itertools import combinations
co_occurrence = Counter()
for genres in movies['genre_list']:
    for pair in combinations(sorted(genres), 2):
        co_occurrence[pair] += 1

# Top 10 cặp genre thường xuất hiện cùng nhau
top_pairs = co_occurrence.most_common(10)
for (g1, g2), count in top_pairs:
    print(f"{g1:15s} + {g2:15s}: {count:,} movies")
```

**Expected:**

```
Avg genres per movie: 2.05

Drama + Comedy    :  4,200 movies
Drama + Romance   :  3,800 movies
Comedy + Romance  :  3,200 movies
Drama + Thriller  :  3,500 movies
...
```

### 3.3 TF-IDF Sanity Check (Mini Demo)

**Notebook cell demo để verify hiểu lý thuyết:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Test với 5 movies
sample_movies = movies.head(5).copy()

# Vectorize (split bằng '|')
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
sample_tfidf = tfidf.fit_transform(sample_movies['genres'])

print(f"TF-IDF matrix shape: {sample_tfidf.shape}")
print(f"Vocabulary: {tfidf.get_feature_names_out()}")
print(f"TF-IDF for Toy Story:\n{sample_tfidf[0].toarray()}")

# Similarity matrix
sim = cosine_similarity(sample_tfidf)
print(f"\nSimilarity matrix:\n{sim.round(3)}")
```

**Expected output:**

```
TF-IDF matrix shape: (5, 10)
TF-IDF for Toy Story: [[0.707 0.    0.    0.707 0.    0.    0.    0.    0.    0.707]]
Similarity matrix:
[[1.    0.579 0.    0.    0.349]
 [0.579 1.    0.    0.    0.    ]
 [0.    0.    1.    0.    0.    ]
 [0.    0.    0.    1.    0.    ]
 [0.349 0.    0.    0.    1.    ]]
```

**Quan sát:**
- Toy Story & Jumanji: 0.579 (cùng Adventure+Children)
- Toy Story & Grumpier Old Men: 0.0 (không overlap)
- Movies tương tự nhau (cùng row) = 1.0

### 3.4 Chuẩn Bị Tuần 3

Sau Tuần 1-2, Người 2 phải có:
- ✅ Hiểu rõ TF-IDF & cosine similarity
- ✅ Biết genre distribution, co-occurrence patterns
- ✅ Sample TF-IDF matrix hoạt động đúng
- ✅ Đã đọc `src/data_loader.py` của Người 1 để biết API

---

## 4. Tuần 3: Core Implementation - Content-Based

**Timeline:** 5 giờ
**Output:** `src/content_based.py` hoàn chỉnh + unit tests pass

### 4.1 File `src/content_based.py` - Full Code

```python
"""
Content-Based Filtering Recommender System

Sử dụng TF-IDF trên genres + cosine similarity để tìm items tương tự.
Recommend items dựa trên lịch sử tương tác của user.

References:
- TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
import logging
from typing import List, Tuple, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Content-Based Filtering recommender sử dụng TF-IDF trên item features.

    Workflow:
    1. Fit: Vectorize genres thành TF-IDF matrix
    2. Build user profile: aggregate TF-IDF vectors của items user đã tương tác
    3. Recommend: cosine similarity giữa user profile và all items
    4. Return top-N items chưa user tương tác

    Example:
        >>> cb = ContentBasedRecommender()
        >>> cb.fit(movies_df, genres_col='genres')
        >>> recommendations = cb.recommend(user_id=123, train_df=train_df, n=10)
    """

    def __init__(
        self,
        max_features: int = 100,
        min_df: int = 2,
        token_pattern: str = r'[^|]+'
    ):
        """
        Khởi tạo Content-Based Recommender.

        Args:
            max_features: Max số features (genres) trong TF-IDF vocabulary
            min_df: Min document frequency - loại bỏ genres quá hiếm
            token_pattern: Regex để split genres string (mặc định split bằng '|')
        """
        self.max_features = max_features
        self.min_df = min_df
        self.token_pattern = token_pattern

        # Components sẽ được set khi fit()
        self.tfidf: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[csr_matrix] = None
        self.movie_ids: Optional[np.ndarray] = None
        self.movie_id_to_idx: Optional[dict] = None
        self.idx_to_movie_id: Optional[dict] = None
        self.movies_df: Optional[pd.DataFrame] = None

        self._is_fitted = False
        logger.info(f"ContentBasedRecommender initialized (max_features={max_features}, min_df={min_df})")

    def fit(
        self,
        movies_df: pd.DataFrame,
        genres_col: str = 'genres'
    ) -> 'ContentBasedRecommender':
        """
        Fit model trên movies dataframe.

        Args:
            movies_df: DataFrame chứa movieId và genres
                       Columns: [movieId, title, genres]
            genres_col: Tên cột chứa genres (string phân cách bằng '|')

        Returns:
            self (for chaining)

        Raises:
            ValueError: Nếu thiếu columns
        """
        # Validation
        if 'movieId' not in movies_df.columns:
            raise ValueError("movies_df phải có cột 'movieId'")
        if genres_col not in movies_df.columns:
            raise ValueError(f"movies_df phải có cột '{genres_col}'")

        logger.info(f"Fitting Content-Based on {len(movies_df):,} movies...")

        # Lưu movies_df để dùng sau (cho titles trong recommendations)
        self.movies_df = movies_df.copy()

        # Build ID mappings
        self.movie_ids = movies_df['movieId'].values
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}

        # Preprocess genres: thay "(no genres listed)" bằng ""
        genres_processed = movies_df[genres_col].fillna('').astype(str)
        genres_processed = genres_processed.replace('(no genres listed)', '')

        # TF-IDF vectorization
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            token_pattern=self.token_pattern,
            lowercase=True,
            stop_words=None  # Genres không có stopwords
        )

        self.tfidf_matrix = self.tfidf.fit_transform(genres_processed)
        # Shape: (n_movies, n_features)

        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.tfidf.vocabulary_)}")
        logger.info(f"Sparsity: {(1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100:.2f}%")

        self._is_fitted = True
        return self

    def _build_user_profile(
        self,
        user_id: int,
        train_df: pd.DataFrame,
        threshold: float = 4.0
    ) -> csr_matrix:
        """
        Xây dựng user profile từ lịch sử tương tác.

        Strategy: Aggregate (sum) TF-IDF vectors của tất cả items user đã rate >= threshold,
        với weight = rating value (cao hơn = quan trọng hơn).

        Args:
            user_id: User ID cần build profile
            train_df: Training dataframe [userId, movieId, rating, ...]
            threshold: Chỉ lấy ratings >= threshold (positive feedback)

        Returns:
            Sparse vector shape (1, n_features) - aggregated user profile

        Raises:
            ValueError: Nếu user không có trong train_df
        """
        if not self._is_fitted:
            raise RuntimeError("Model chưa được fit(). Gọi fit() trước.")

        # Lấy positive interactions của user
        user_history = train_df[
            (train_df['userId'] == user_id) &
            (train_df['rating'] >= threshold)
        ]

        if len(user_history) == 0:
            raise ValueError(f"User {user_id} không có positive interactions trong train set")

        # Map movieIds → indices
        indices = []
        weights = []
        skipped = 0

        for _, row in user_history.iterrows():
            movie_id = row['movieId']
            if movie_id in self.movie_id_to_idx:
                indices.append(self.movie_id_to_idx[movie_id])
                weights.append(row['rating'])  # Weight = rating
            else:
                skipped += 1

        if not indices:
            raise ValueError(f"User {user_id} history không chứa items trong TF-IDF vocabulary")

        if skipped > 0:
            logger.warning(f"User {user_id}: skipped {skipped}/{len(user_history)} items (not in vocabulary)")

        # Aggregate TF-IDF vectors với weights
        user_vectors = self.tfidf_matrix[indices]  # (n_items, n_features)
        weight_array = np.array(weights).reshape(-1, 1)
        # Weighted sum
        user_profile = user_vectors.multiply(weight_array).sum(axis=0)
        # Shape: (1, n_features)

        # Normalize (L2) để similarity scores comparable across users
        from sklearn.preprocessing import normalize
        user_profile = normalize(user_profile, norm='l2')

        return csr_matrix(user_profile)

    def recommend(
        self,
        user_id: int,
        train_df: pd.DataFrame,
        n: int = 10,
        threshold: float = 4.0,
        exclude_seen: bool = True,
        return_scores: bool = False
    ) -> Union[List[Tuple[int, float]], List[int]]:
        """
        Generate top-N recommendations cho user.

        Args:
            user_id: User ID cần recommend
            train_df: Training dataframe
            n: Số lượng recommendations trả về
            threshold: Chỉ tính items rated >= threshold làm user history
            exclude_seen: Loại bỏ items user đã tương tác
            return_scores: Nếu True, return (movieId, score), ngược lại chỉ movieId

        Returns:
            List of (movieId, score) tuples hoặc List of movieIds

        Raises:
            ValueError: Nếu user không có history
        """
        if not self._is_fitted:
            raise RuntimeError("Model chưa được fit(). Gọi fit() trước.")

        # Build user profile
        try:
            user_profile = self._build_user_profile(user_id, train_df, threshold)
        except ValueError as e:
            logger.warning(f"Cannot recommend for user {user_id}: {e}")
            return []

        # Compute similarity: user_profile (1, n_features) vs all movies (n_movies, n_features)
        similarities = cosine_similarity(user_profile, self.tfidf_matrix).flatten()
        # Shape: (n_movies,)

        # Exclude seen items
        if exclude_seen:
            user_seen = set(train_df[train_df['userId'] == user_id]['movieId'].values)
            seen_indices = [self.movie_id_to_idx[mid] for mid in user_seen if mid in self.movie_id_to_idx]
            similarities[seen_indices] = -np.inf  # Set to -inf to exclude

        # Top-N
        top_indices = np.argsort(similarities)[::-1][:n]

        # Build results
        results = []
        for idx in top_indices:
            movie_id = self.movie_ids[idx]
            score = similarities[idx]
            if score == -np.inf:
                continue
            if return_scores:
                results.append((movie_id, float(score)))
            else:
                results.append(int(movie_id))

        return results

    def recommend_batch(
        self,
        user_ids: List[int],
        train_df: pd.DataFrame,
        n: int = 10,
        threshold: float = 4.0,
        exclude_seen: bool = True
    ) -> dict:
        """
        Batch recommendation cho nhiều users.

        Args:
            user_ids: List user IDs
            train_df: Training dataframe
            n: Top-N per user
            threshold: Rating threshold
            exclude_seen: Exclude seen items

        Returns:
            Dict {user_id: [(movieId, score), ...]}
        """
        logger.info(f"Generating recommendations for {len(user_ids):,} users...")

        results = {}
        skipped = 0

        for uid in user_ids:
            try:
                recs = self.recommend(
                    uid, train_df, n=n, threshold=threshold,
                    exclude_seen=exclude_seen, return_scores=True
                )
                results[uid] = recs
            except Exception as e:
                logger.debug(f"Skip user {uid}: {e}")
                skipped += 1

        logger.info(f"Generated recommendations for {len(results):,}/{len(user_ids):,} users (skipped: {skipped})")
        return results

    def get_similar_items(
        self,
        movie_id: int,
        n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Tìm N items tương tự với movie cho trước (dựa trên content features).

        Hữu ích cho:
        - "More like this" feature
        - Cold-start items (chỉ cần features, không cần interactions)
        - Item similarity analysis

        Args:
            movie_id: Movie ID cần tìm similar items
            n: Số lượng similar items trả về

        Returns:
            List of (movieId, similarity_score) tuples, sorted desc

        Raises:
            ValueError: Nếu movie_id không có trong vocabulary
        """
        if not self._is_fitted:
            raise RuntimeError("Model chưa được fit(). Gọi fit() trước.")

        if movie_id not in self.movie_id_to_idx:
            raise ValueError(f"Movie {movie_id} không có trong TF-IDF vocabulary")

        idx = self.movie_id_to_idx[movie_id]
        movie_vector = self.tfidf_matrix[idx]

        # Cosine similarity với all movies
        similarities = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()

        # Top-N+1 (để loại chính nó ra)
        top_indices = np.argsort(similarities)[::-1][:n+1]

        results = []
        for similar_idx in top_indices:
            if similar_idx == idx:
                continue  # Skip chính nó
            results.append((int(self.movie_ids[similar_idx]), float(similarities[similar_idx])))
            if len(results) == n:
                break

        return results

    def get_movie_features(self, movie_id: int) -> dict:
        """
        Lấy TF-IDF features của một movie (cho debugging/explainability).

        Args:
            movie_id: Movie ID

        Returns:
            Dict {feature_name: tfidf_score}
        """
        if not self._is_fitted:
            raise RuntimeError("Model chưa được fit(). Gọi fit() trước.")

        if movie_id not in self.movie_id_to_idx:
            raise ValueError(f"Movie {movie_id} không có trong vocabulary")

        idx = self.movie_id_to_idx[movie_id]
        vector = self.tfidf_matrix[idx].toarray().flatten()

        feature_names = self.tfidf.get_feature_names_out()
        features = {name: float(score) for name, score in zip(feature_names, vector) if score > 0}

        # Sort by score
        features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True))

        return features

    def save(self, model_dir: Union[str, Path]) -> None:
        """
        Save model components ra disk.

        Saved files:
        - tfidf.pkl: TfidfVectorizer object
        - tfidf_matrix.npz: Sparse TF-IDF matrix
        - movie_ids.pkl: Array of movie IDs
        - metadata.pkl: Model metadata (max_features, min_df, etc.)

        Args:
            model_dir: Directory path để save model
        """
        if not self._is_fitted:
            raise RuntimeError("Không thể save model chưa được fit()")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save TF-IDF vectorizer
        with open(model_dir / 'tfidf.pkl', 'wb') as f:
            pickle.dump(self.tfidf, f)

        # Save sparse matrix
        save_npz(model_dir / 'tfidf_matrix.npz', self.tfidf_matrix)

        # Save movie IDs
        with open(model_dir / 'movie_ids.pkl', 'wb') as f:
            pickle.dump(self.movie_ids, f)

        # Save metadata
        metadata = {
            'max_features': self.max_features,
            'min_df': self.min_df,
            'token_pattern': self.token_pattern,
            'n_movies': len(self.movie_ids),
            'vocab_size': len(self.tfidf.vocabulary_)
        }
        with open(model_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Model saved to {model_dir}")

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> 'ContentBasedRecommender':
        """
        Load model từ disk.

        Args:
            model_dir: Directory path chứa saved model

        Returns:
            Loaded ContentBasedRecommender instance
        """
        model_dir = Path(model_dir)

        # Load metadata
        with open(model_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            max_features=metadata['max_features'],
            min_df=metadata['min_df'],
            token_pattern=metadata['token_pattern']
        )

        # Load components
        with open(model_dir / 'tfidf.pkl', 'rb') as f:
            instance.tfidf = pickle.load(f)

        instance.tfidf_matrix = load_npz(model_dir / 'tfidf_matrix.npz')

        with open(model_dir / 'movie_ids.pkl', 'rb') as f:
            instance.movie_ids = pickle.load(f)

        instance.movie_id_to_idx = {mid: idx for idx, mid in enumerate(instance.movie_ids)}
        instance.idx_to_movie_id = {idx: mid for mid, idx in instance.movie_id_to_idx.items()}
        instance._is_fitted = True

        logger.info(f"Model loaded from {model_dir} ({metadata['n_movies']:,} movies, vocab={metadata['vocab_size']})")
        return instance


# === CONVENIENCE FUNCTIONS ===

def evaluate_coverage(recommendations: dict, catalog_size: int) -> float:
    """
    Tính catalog coverage: % items được recommend ít nhất 1 lần.

    Args:
        recommendations: Dict {user_id: [(movieId, score), ...]}
        catalog_size: Tổng số items trong catalog

    Returns:
        Coverage ratio [0, 1]
    """
    recommended_items = set()
    for recs in recommendations.values():
        for movie_id, _ in recs:
            recommended_items.add(movie_id)
    return len(recommended_items) / catalog_size


def evaluate_diversity(recommendations: dict, similarity_matrix: np.ndarray,
                        movie_id_to_idx: dict) -> float:
    """
    Tính intra-list diversity: trung bình pairwise distance trong top-N.

    Diversity = 1 - similarity (distance)
    Higher = more diverse recommendations

    Args:
        recommendations: Dict {user_id: [(movieId, score), ...]}
        similarity_matrix: Item-item similarity matrix
        movie_id_to_idx: Movie ID → index mapping

    Returns:
        Average diversity score [0, 1]
    """
    diversities = []
    for recs in recommendations.values():
        movie_ids = [mid for mid, _ in recs]
        indices = [movie_id_to_idx[mid] for mid in movie_ids if mid in movie_id_to_idx]
        if len(indices) < 2:
            continue

        # Pairwise similarity
        n = len(indices)
        sim_sum = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                sim_sum += similarity_matrix[indices[i], indices[j]]
                count += 1

        avg_sim = sim_sum / count if count > 0 else 0
        diversity = 1 - avg_sim
        diversities.append(diversity)

    return np.mean(diversities) if diversities else 0.0
```

### 4.2 Quick Test Sau Khi Implement

Tạo `tests/test_content_based.py`:

```python
import pytest
import pandas as pd
import numpy as np
from src.content_based import ContentBasedRecommender


@pytest.fixture
def sample_movies():
    """Sample movies cho testing"""
    return pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride'],
        'genres': [
            'Adventure|Animation|Children|Comedy|Fantasy',
            'Adventure|Children|Fantasy',
            'Comedy|Romance',
            'Comedy|Drama|Romance',
            'Comedy'
        ]
    })


@pytest.fixture
def sample_train():
    """Sample train interactions"""
    return pd.DataFrame({
        'userId': [1, 1, 1, 2, 2],
        'movieId': [1, 2, 3, 4, 5],
        'rating': [5.0, 4.5, 4.0, 5.0, 4.0]
    })


def test_fit_basic(sample_movies):
    """Test basic fit"""
    cb = ContentBasedRecommender()
    cb.fit(sample_movies)

    assert cb._is_fitted
    assert cb.tfidf_matrix.shape[0] == 5
    assert cb.movie_id_to_idx[1] == 0


def test_recommend_basic(sample_movies, sample_train):
    """Test basic recommend"""
    cb = ContentBasedRecommender()
    cb.fit(sample_movies)

    recs = cb.recommend(user_id=1, train_df=sample_train, n=3)

    assert len(recs) > 0
    # Items 1,2,3 already in user 1's history → should not be recommended
    recommended_ids = [r for r in recs]
    assert 1 not in recommended_ids
    assert 2 not in recommended_ids
    assert 3 not in recommended_ids


def test_get_similar_items(sample_movies):
    """Test similar items"""
    cb = ContentBasedRecommender()
    cb.fit(sample_movies)

    # Toy Story (id=1) should be similar to Jumanji (id=2) - both Adventure+Children
    similar = cb.get_similar_items(movie_id=1, n=3)
    similar_ids = [mid for mid, _ in similar]

    assert 2 in similar_ids  # Jumanji
    assert 1 not in similar_ids  # Không chứa chính nó


def test_save_load(sample_movies, tmp_path):
    """Test save/load"""
    cb = ContentBasedRecommender()
    cb.fit(sample_movies)

    save_path = tmp_path / "test_cb_model"
    cb.save(save_path)

    loaded = ContentBasedRecommender.load(save_path)
    assert loaded._is_fitted
    assert loaded.tfidf_matrix.shape == cb.tfidf_matrix.shape
```

**Run tests:**

```bash
cd /home/huyquang/Workspace/Personal/PyCharmProjects/RecommendationSystem
pytest tests/test_content_based.py -v
```

**Expected:**

```
test_fit_basic PASSED
test_recommend_basic PASSED
test_get_similar_items PASSED
test_save_load PASSED
========== 4 passed in 2.3s ==========
```

---

## 5. Tuần 3: Evaluation Module

**Timeline:** 3 giờ
**Output:** `src/evaluation.py` với 6 metrics + unit tests

### 5.1 File `src/evaluation.py` - Full Code

```python
"""
Evaluation metrics cho recommendation models.

Implement các ranking metrics:
- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- Hit Rate@K
- Coverage

References:
- Cremonesi et al. (2010). Performance of recommender algorithms on top-N recommendation tasks.
- Herlocker et al. (2004). Evaluating collaborative filtering recommender systems.
"""

import numpy as np
import pandas as pd
from typing import List, Set, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RecommenderEvaluator:
    """
    Evaluator cho top-N recommendation tasks.

    Usage:
        >>> evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
        >>> metrics = evaluator.evaluate_model(
        ...     recommendations={user_id: [(movie_id, score), ...]},
        ...     test_df=test_df,
        ...     catalog_size=n_movies
        ... )
        >>> print(metrics)
    """

    def __init__(self, k_values: List[int] = None, relevance_threshold: float = 4.0):
        """
        Args:
            k_values: List các K values để evaluate (default: [5, 10, 20])
            relevance_threshold: Ratings >= threshold được coi là relevant
        """
        self.k_values = k_values if k_values is not None else [5, 10, 20]
        self.relevance_threshold = relevance_threshold
        logger.info(f"Evaluator initialized (k={self.k_values}, threshold={relevance_threshold})")

    def precision_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Precision@K = (# relevant items in top-K) / K

        Args:
            recommended: List recommended items (sorted by score desc)
            relevant: Set of relevant items (ground truth)
            k: Cut-off

        Returns:
            Precision score [0, 1]
        """
        if k <= 0:
            return 0.0

        recommended_k = recommended[:k]
        relevant_in_top_k = len(set(recommended_k) & relevant)
        return relevant_in_top_k / k

    def recall_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Recall@K = (# relevant items in top-K) / (total # relevant items)

        Args:
            recommended: List recommended items
            relevant: Set of relevant items
            k: Cut-off

        Returns:
            Recall score [0, 1]
        """
        if not relevant:
            return 0.0

        recommended_k = recommended[:k]
        relevant_in_top_k = len(set(recommended_k) & relevant)
        return relevant_in_top_k / len(relevant)

    def ndcg_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        NDCG@K (Normalized Discounted Cumulative Gain).

        DCG = Σ (rel_i / log2(i+1))  for i=1..K
        NDCG = DCG / IDCG

        Trong đó rel_i = 1 nếu item ở vị trí i relevant, 0 nếu không.
        Position discount: log2(i+1)

        Args:
            recommended: List recommended items
            relevant: Set of relevant items
            k: Cut-off

        Returns:
            NDCG score [0, 1]
        """
        if k <= 0:
            return 0.0

        recommended_k = recommended[:k]

        # DCG
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant:
                # Position discount: log2(i+2) vì i bắt đầu từ 0
                dcg += 1.0 / np.log2(i + 2)

        # IDCG (best possible DCG - all relevant items ở top)
        idcg = 0.0
        for i in range(min(len(relevant), k)):
            idcg += 1.0 / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def average_precision_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Average Precision@K = (1/R) × Σ Precision@i × rel(i)
        trong đó R = total relevant items, rel(i) = 1 nếu item ở vị trí i relevant.

        Args:
            recommended: List recommended items
            relevant: Set of relevant items
            k: Cut-off

        Returns:
            AP@K score [0, 1]
        """
        if not relevant:
            return 0.0

        recommended_k = recommended[:k]
        score = 0.0
        num_hits = 0

        for i, item in enumerate(recommended_k):
            if item in relevant:
                num_hits += 1
                # Precision tại vị trí i+1
                precision_at_i = num_hits / (i + 1)
                score += precision_at_i

        # Normalize by min(R, K)
        return score / min(len(relevant), k)

    def hit_rate_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Hit Rate@K = 1 nếu có ít nhất 1 relevant item trong top-K, 0 nếu không.

        Args:
            recommended: List recommended items
            relevant: Set of relevant items
            k: Cut-off

        Returns:
            1.0 nếu hit, 0.0 nếu miss
        """
        if k <= 0:
            return 0.0
        recommended_k = recommended[:k]
        return 1.0 if any(item in relevant for item in recommended_k) else 0.0

    def _build_test_relevant_set(
        self,
        test_df: pd.DataFrame
    ) -> Dict[int, Set[int]]:
        """
        Build ground truth: {user_id: {relevant_movie_ids}}

        Relevant = items rated >= relevance_threshold trong test set.

        Args:
            test_df: Test dataframe [userId, movieId, rating, ...]

        Returns:
            Dict mapping user_id → set of relevant movie_ids
        """
        relevant_df = test_df[test_df['rating'] >= self.relevance_threshold]
        relevant_dict = relevant_df.groupby('userId')['movieId'].apply(set).to_dict()
        logger.info(f"Built test relevant set: {len(relevant_dict):,} users")
        return relevant_dict

    def evaluate_user(
        self,
        user_id: int,
        recommended: List[int],
        relevant: Set[int]
    ) -> Dict[str, float]:
        """
        Evaluate một user với all metrics và all K values.

        Args:
            user_id: User ID (for logging)
            recommended: List recommended items
            relevant: Set relevant items

        Returns:
            Dict of metrics {metric_name: value}
        """
        user_metrics = {}

        for k in self.k_values:
            user_metrics[f'Precision@{k}'] = self.precision_at_k(recommended, relevant, k)
            user_metrics[f'Recall@{k}'] = self.recall_at_k(recommended, relevant, k)
            user_metrics[f'NDCG@{k}'] = self.ndcg_at_k(recommended, relevant, k)
            user_metrics[f'MAP@{k}'] = self.average_precision_at_k(recommended, relevant, k)
            user_metrics[f'HitRate@{k}'] = self.hit_rate_at_k(recommended, relevant, k)

        return user_metrics

    def evaluate_model(
        self,
        recommendations: Dict[int, List[Tuple[int, float]]],
        test_df: pd.DataFrame,
        catalog_size: int = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model trên toàn bộ test users.

        Args:
            recommendations: Dict {user_id: [(movieId, score), ...]}
            test_df: Test dataframe
            catalog_size: Tổng số items (for coverage metric)
            verbose: Print progress

        Returns:
            Dict of averaged metrics
        """
        # Build ground truth
        relevant_dict = self._build_test_relevant_set(test_df)

        # Filter recommendations: chỉ giữ users có trong test set
        eval_users = set(recommendations.keys()) & set(relevant_dict.keys())

        if verbose:
            logger.info(f"Evaluating on {len(eval_users):,} users...")

        # Aggregate metrics across users
        all_metrics = []
        for user_id in eval_users:
            recs = recommendations[user_id]
            recommended_ids = [mid for mid, _ in recs]
            relevant = relevant_dict[user_id]

            user_metrics = self.evaluate_user(user_id, recommended_ids, relevant)
            all_metrics.append(user_metrics)

        # Average across users
        if not all_metrics:
            logger.warning("No users to evaluate!")
            return {}

        metrics_df = pd.DataFrame(all_metrics)
        avg_metrics = metrics_df.mean().to_dict()

        # Coverage (beyond-accuracy)
        if catalog_size is not None:
            recommended_items = set()
            for recs in recommendations.values():
                for mid, _ in recs:
                    recommended_items.add(mid)
            coverage = len(recommended_items) / catalog_size
            avg_metrics['Coverage'] = coverage

        if verbose:
            logger.info("Evaluation complete!")
            for metric, value in sorted(avg_metrics.items()):
                logger.info(f"  {metric}: {value:.4f}")

        return avg_metrics

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        So sánh nhiều models trên cùng metrics.

        Args:
            model_results: Dict {model_name: metrics_dict}

        Returns:
            DataFrame comparison
        """
        df = pd.DataFrame(model_results).T
        df.index.name = 'Model'
        return df.round(4)


# === STANDALONE USAGE ===

def quick_evaluate(
    recommendations: Dict[int, List[int]],
    test_df: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
    relevance_threshold: float = 4.0
) -> Dict[str, float]:
    """
    Convenience function cho quick evaluation.

    Args:
        recommendations: Dict {user_id: [movieId, ...]}
        test_df: Test dataframe
        k_values: List of K values
        relevance_threshold: Rating threshold

    Returns:
        Dict of metrics
    """
    evaluator = RecommenderEvaluator(k_values=k_values, relevance_threshold=relevance_threshold)

    # Convert to (movieId, score) format
    recs_with_scores = {
        uid: [(mid, 1.0) for mid in recs]
        for uid, recs in recommendations.items()
    }

    return evaluator.evaluate_model(recs_with_scores, test_df)
```

### 5.2 Test Evaluation Module

Tạo `tests/test_evaluation.py`:

```python
import pytest
import pandas as pd
import numpy as np
from src.evaluation import RecommenderEvaluator


@pytest.fixture
def evaluator():
    return RecommenderEvaluator(k_values=[5, 10], relevance_threshold=4.0)


def test_precision_at_k(evaluator):
    recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    relevant = {2, 4, 6, 8, 20}  # 5 relevant items

    # Top 5: 1,2,3,4,5 → relevant: {2,4} → 2/5 = 0.4
    assert evaluator.precision_at_k(recommended, relevant, k=5) == 0.4

    # Top 10: 1-10 → relevant: {2,4,6,8} → 4/10 = 0.4
    assert evaluator.precision_at_k(recommended, relevant, k=10) == 0.4


def test_recall_at_k(evaluator):
    recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    relevant = {2, 4, 6, 8, 20}  # 5 relevant items

    # Top 5: found 2 → 2/5 = 0.4
    assert evaluator.recall_at_k(recommended, relevant, k=5) == 0.4

    # Top 10: found 4 → 4/5 = 0.8
    assert evaluator.recall_at_k(recommended, relevant, k=10) == 0.8


def test_ndcg_at_k(evaluator):
    recommended = [1, 2, 3, 4, 5]  # relevant at positions 2, 4 (1-indexed)
    relevant = {2, 4}

    # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63
    # NDCG = 1.5 / 1.63 ≈ 0.92
    ndcg = evaluator.ndcg_at_k(recommended, relevant, k=5)
    assert 0.9 < ndcg < 0.95


def test_hit_rate_at_k(evaluator):
    recommended_hit = [1, 2, 3, 4, 5]
    relevant_hit = {3}

    recommended_miss = [1, 2, 3, 4, 5]
    relevant_miss = {99}

    assert evaluator.hit_rate_at_k(recommended_hit, relevant_hit, k=5) == 1.0
    assert evaluator.hit_rate_at_k(recommended_miss, relevant_miss, k=5) == 0.0


def test_evaluate_model_end_to_end(evaluator):
    """Test full evaluation pipeline"""
    # Mock recommendations
    recommendations = {
        1: [(10, 0.9), (20, 0.8), (30, 0.7), (40, 0.6), (50, 0.5)],
        2: [(15, 0.9), (25, 0.8), (35, 0.7), (45, 0.6), (55, 0.5)]
    }

    # Mock test set
    test_df = pd.DataFrame({
        'userId': [1, 1, 2, 2],
        'movieId': [10, 20, 15, 99],  # User 2: 99 not in recs
        'rating': [5.0, 4.5, 5.0, 4.0]
    })

    metrics = evaluator.evaluate_model(recommendations, test_df, catalog_size=100)

    assert 'Precision@5' in metrics
    assert 'Recall@5' in metrics
    assert 'NDCG@5' in metrics
    assert 'Coverage' in metrics
    assert 0 <= metrics['Coverage'] <= 1
```

**Run tests:**

```bash
pytest tests/test_evaluation.py -v
```

**Expected:**

```
test_precision_at_k PASSED
test_recall_at_k PASSED
test_ndcg_at_k PASSED
test_hit_rate_at_k PASSED
test_evaluate_model_end_to_end PASSED
========== 5 passed in 1.8s ==========
```

---

## 6. Tuần 4: Experiments Notebook

**Timeline:** 5 giờ
**Output:** `notebooks/04-content-based-experiments.ipynb` với 6 visualizations

### 6.1 Notebook Structure

**File:** `notebooks/04-content-based-experiments.ipynb`

**Cells tổng quan:**

```python
# Cell 1: Setup & Imports
"""
Content-Based Filtering Experiments
- Author: Người 2
- Date: Tuần 4
- Dataset: MovieLens 25M
"""
import sys

sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import time

from src.data_loader import ML25MLoader
from src.content_based import ContentBasedRecommender
from src.evaluation import RecommenderEvaluator
from src.collaborative_filtering import ALSRecommender  # From Person 1
from src.popularity_baselines import PopularityBaseline  # From Person 1

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Cell 2: Load Data
loader = ML25MLoader('data/raw/ml-25m')

# Load movies (cho CB)
movies = loader.load_movies()
print(f"Movies: {len(movies):,}")
print(movies.head())

# Load processed ratings (từ Person 1's output)
train_df = pd.read_csv('data/processed/train_ml25m.csv')
test_df = pd.read_csv('data/processed/test_ml25m.csv')

print(f"Train: {len(train_df):,} interactions")
print(f"Test: {len(test_df):,} interactions")

# Cell 3: Train Content-Based Model
print("=" * 60)
print("Training Content-Based Model")
print("=" * 60)

start = time.time()
cb_model = ContentBasedRecommender(max_features=100, min_df=2)
cb_model.fit(movies, genres_col='genres')
print(f"\nTraining time: {time.time() - start:.2f}s")

# Quick test: similar to Toy Story
toy_story_id = 1  # movieId=1
similar = cb_model.get_similar_items(toy_story_id, n=5)
print(f"\nMovies similar to Toy Story:")
for mid, score in similar:
    title = movies[movies['movieId'] == mid]['title'].values[0]
    print(f"  [{score:.3f}] {title}")

# Cell 4: Generate Recommendations cho Test Users
print("=" * 60)
print("Generating Recommendations")
print("=" * 60)

# Sample 1000 test users (full evaluation có thể chậm)
test_users = test_df['userId'].unique()[:1000]
print(f"Evaluating on {len(test_users):,} users...")

start = time.time()
cb_recs = cb_model.recommend_batch(
    user_ids=list(test_users),
    train_df=train_df,
    n=20,  # Top-20
    threshold=4.0
)
print(f"Recommendation time: {time.time() - start:.2f}s")

# Cell 5: Evaluate Content-Based
print("=" * 60)
print("Evaluating Content-Based")
print("=" * 60)

evaluator = RecommenderEvaluator(k_values=[5, 10, 20], relevance_threshold=4.0)
cb_metrics = evaluator.evaluate_model(
    recommendations=cb_recs,
    test_df=test_df,
    catalog_size=len(movies),
    verbose=True
)

print("\nContent-Based Metrics:")
for k, v in sorted(cb_metrics.items()):
    print(f"  {k}: {v:.4f}")

# Save metrics
with open('results/cb_metrics.json', 'w') as f:
    json.dump(cb_metrics, f, indent=2)

# Cell 6: Load Popularity Baseline (từ Person 1)
print("=" * 60)
print("Evaluating Popularity Baseline")
print("=" * 60)

pop_baseline = PopularityBaseline()
pop_baseline.fit(train_df)

pop_recs = {}
top_popular = pop_baseline.get_top_n(n=20)
for uid in test_users:
    # Filter items user đã thấy
    seen = set(train_df[train_df['userId'] == uid]['movieId'])
    recs = [mid for mid in top_popular if mid not in seen][:20]
    pop_recs[uid] = [(mid, 1.0) for mid in recs]

pop_metrics = evaluator.evaluate_model(pop_recs, test_df, catalog_size=len(movies))

# Cell 7: Load ALS Results (từ Person 1)
print("=" * 60)
print("Evaluating ALS (CF)")
print("=" * 60)

als_model = ALSRecommender.load('models/als_best.pkl')

als_recs = als_model.recommend_batch(
    user_ids=list(test_users),
    user_item_matrix=...,  # Load từ Person 1
    n=20,
    filter_already_liked=True
)

als_metrics = evaluator.evaluate_model(als_recs, test_df, catalog_size=len(movies))

# Cell 8: Comparison Table
comparison_df = evaluator.compare_models({
    'Popularity': pop_metrics,
    'ALS (CF)': als_metrics,
    'Content-Based': cb_metrics
})

print("\n" + "=" * 60)
print("METHOD COMPARISON")
print("=" * 60)
print(comparison_df.to_string())

# Save
comparison_df.to_csv('results/all_methods_comparison.csv')

# Cell 9: Visualization 1 - Metrics Comparison
fig, ax = plt.subplots(figsize=(14, 6))

# Plot Precision, Recall, NDCG at K=10
metrics_to_plot = ['Precision@10', 'Recall@10', 'NDCG@10']
model_names = ['Popularity', 'ALS (CF)', 'Content-Based']

x = np.arange(len(metrics_to_plot))
width = 0.25

for i, model in enumerate(model_names):
    values = [comparison_df.loc[model, m] for m in metrics_to_plot]
    ax.bar(x + i * width, values, width, label=model)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Comparison: Popularity vs ALS vs Content-Based', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_to_plot)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/all_methods_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Cell 10: Visualization 2 - Metrics vs K
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax_idx, metric_base in enumerate(['Precision', 'Recall', 'NDCG']):
    ax = axes[ax_idx]

    for model in model_names:
        values = [comparison_df.loc[model, f'{metric_base}@{k}'] for k in [5, 10, 20]]
        ax.plot([5, 10, 20], values, marker='o', label=model, linewidth=2)

    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel(f'{metric_base}@K', fontsize=12)
    ax.set_title(f'{metric_base}@K vs K', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/cb_metrics_vs_k.png', dpi=150, bbox_inches='tight')
plt.show()

# Cell 11: Genre Distribution Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Count genres in top-20 recommendations across all users
from collections import Counter

genre_counter = Counter()
movies_dict = dict(zip(movies['movieId'], movies['genres']))

for uid, recs in cb_recs.items():
    for mid, _ in recs:
        genres = movies_dict.get(mid, '').split('|')
        genre_counter.update(genres)

genre_dist = pd.DataFrame(genre_counter.most_common(15), columns=['Genre', 'Count'])

sns.barplot(data=genre_dist, x='Count', y='Genre', ax=axes[0], palette='viridis')
axes[0].set_title('Top Genres trong Content-Based Recommendations', fontsize=13, fontweight='bold')

# Compare với overall genre distribution
overall_counter = Counter()
for genres_str in movies['genres']:
    if genres_str != '(no genres listed)':
        overall_counter.update(genres_str.split('|'))

overall_dist = pd.DataFrame(overall_counter.most_common(15), columns=['Genre', 'Count'])

sns.barplot(data=overall_dist, x='Count', y='Genre', ax=axes[1], palette='magma')
axes[1].set_title('Overall Genre Distribution trong Catalog', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/cb_genre_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Cell 12: Visualization - Similar Movies Example
# Toy Story → top 5 similar
toy_story_id = 1
similar_movies = cb_model.get_similar_items(toy_story_id, n=5)

fig, ax = plt.subplots(figsize=(10, 6))
titles = ['Toy Story (1995)']  # Source movie
scores = [1.0]

for mid, score in similar_movies:
    title = movies[movies['movieId'] == mid]['title'].values[0]
    titles.append(title)
    scores.append(score)

# Horizontal bar chart
y_pos = np.arange(len(titles))
colors = ['red'] + ['steelblue'] * len(similar_movies)

ax.barh(y_pos, scores, color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(titles)
ax.invert_yaxis()
ax.set_xlabel('Similarity Score', fontsize=12)
ax.set_title('Top 5 Movies Similar to "Toy Story"', fontsize=14, fontweight='bold')

for i, score in enumerate(scores):
    ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/cb_similar_movies.png', dpi=150, bbox_inches='tight')
plt.show()

# Cell 13: Cold-start Analysis
print("=" * 60)
print("Cold-Start Analysis")
print("=" * 60)

# Chia users theo activity level (số ratings trong train)
user_activity = train_df.groupby('userId').size()
user_activity_bins = pd.cut(user_activity, bins=[0, 10, 50, 200, float('inf')],
                            labels=['Very Low (1-10)', 'Low (11-50)', 'Medium (51-200)', 'High (>200)'])

cold_start_metrics = {}
for activity_level in ['Very Low (1-10)', 'Low (11-50)', 'Medium (51-200)', 'High (>200)']:
    # Filter users
    activity_users = user_activity[user_activity_bins == activity_level].index
    activity_users_in_test = [uid for uid in test_users if uid in activity_users]

    if len(activity_users_in_test) < 10:
        continue

    # Evaluate on this segment
    activity_recs = {uid: cb_recs[uid] for uid in activity_users_in_test if uid in cb_recs}
    seg_metrics = evaluator.evaluate_model(activity_recs, test_df, catalog_size=len(movies), verbose=False)
    cold_start_metrics[activity_level] = seg_metrics

cold_start_df = pd.DataFrame(cold_start_metrics).T

print("\nContent-Based Performance by User Activity Level:")
print(cold_start_df[['Precision@10', 'Recall@10', 'NDCG@10']].to_string())

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))

metrics_to_show = ['Precision@10', 'Recall@10', 'NDCG@10']
for metric in metrics_to_show:
    ax.plot(cold_start_df.index, cold_start_df[metric], marker='o', label=metric, linewidth=2)

ax.set_xlabel('User Activity Level', fontsize=12)
ax.set_ylabel('Metric Score', fontsize=12)
ax.set_title('Content-Based Performance by User Activity (Cold-Start Analysis)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/cb_cold_start_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Cell 14: Case Study - Specific User
print("=" * 60)
print("Case Study: User Analysis")
print("=" * 60)

# Pick user with diverse history
sample_user = test_users[10]  # Random user

# User history
user_history = train_df[(train_df['userId'] == sample_user) & (train_df['rating'] >= 4.0)]
print(f"\nUser {sample_user} history (top 10):")
history_titles = movies[movies['movieId'].isin(user_history['movieId'])][['title', 'genres']].head(10)
print(history_titles.to_string())

# Recommendations
user_recs = cb_recs[sample_user]
print(f"\nTop 10 Content-Based Recommendations:")
rec_titles = movies[movies['movieId'].isin([mid for mid, _ in user_recs[:10]])][['title', 'genres']].head(10)
print(rec_titles.to_string())

# Cell 15: Save All Results
print("=" * 60)
print("Saving Results")
print("=" * 60)

# Save metrics
results = {
    'cb_metrics': cb_metrics,
    'pop_metrics': pop_metrics,
    'als_metrics': als_metrics,
    'cold_start_metrics': {k: v for k, v in cold_start_metrics.items()}
}

with open('results/all_experiments.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n✓ All experiments complete!")
print(f"✓ Saved to results/all_experiments.json")
print(f"✓ Figures saved to figures/cb_*.png")
```

### 6.2 Expected Outputs

**Console output:**

```
Training Content-Based Model
Training time: 3.45s
TF-IDF matrix shape: (62423, 24)
Vocabulary size: 24

Movies similar to Toy Story:
  [0.813] Toy Story 2 (1999)
  [0.752] A Bug's Life (1998)
  [0.689] Monsters, Inc. (2001)
  [0.645] Shrek (2001)
  [0.598] Finding Nemo (2003)

Generating Recommendations
Evaluating on 1,000 users...
Recommendation time: 12.34s

Evaluating Content-Based
Evaluation complete!
  Coverage: 0.2156
  HitRate@5: 0.3421
  HitRate@10: 0.5234
  HitRate@20: 0.6789
  MAP@5: 0.1234
  MAP@10: 0.1456
  MAP@20: 0.1678
  NDCG@5: 0.1823
  NDCG@10: 0.2145
  NDCG@20: 0.2567
  Precision@5: 0.1567
  Precision@10: 0.1234
  Precision@20: 0.0987
  Recall@5: 0.0823
  Recall@10: 0.1234
  Recall@20: 0.1789
```

**Visualizations generated:**

```
figures/
├── all_methods_comparison.png         # Bar chart: 3 methods x 3 metrics
├── cb_metrics_vs_k.png                # Line chart: metrics vs K
├── cb_genre_distribution.png          # Top genres in recs vs catalog
├── cb_similar_movies.png              # Toy Story similar movies
└── cb_cold_start_analysis.png         # Performance by user activity
```

---

## 7. Tuần 5: Report Writing

**Timeline:** 8 giờ
**Output:** Chapters 2.3, 2.4, 4.4, 5.1, 5.4, lead Ch.6

### 7.1 Chapter 2.3: Content-based Filtering (2 trang)

**Section 2.3.1: Feature Extraction**

```markdown
Content-based filtering dựa trên giả thuyết: nếu user thích một item, họ sẽ thích
các items có đặc trưng tương tự. Với MovieLens, mỗi movie được biểu diễn bằng
metadata (title, genres). Trong nghiên cứu này, chúng tôi sử dụng **genres**
làm primary feature vì:

1. **Available**: mỗi movie có ít nhất 1 genre (trừ 0.4% không có)
2. **Meaningful**: genres phản ánh semantic content của movie
3. **Multi-label**: mỗi movie có thể thuộc nhiều genres (trung bình 2.05)

**Preprocessing:**
- Split genres string bằng delimiter '|'
- Loại bỏ movies với "(no genres listed)"
- Chuẩn hóa: lowercase, strip whitespace
```

**Section 2.3.2: TF-IDF và Similarity Measures**

```markdown
**TF-IDF Vectorization:**

Chúng tôi áp dụng Term Frequency - Inverse Document Frequency (TF-IDF)
để chuyển đổi genres thành feature vectors. Công thức:

TF(t, d) = (số lần term t xuất hiện trong document d) / (tổng số terms)
IDF(t) = log(N / df(t))
TF-IDF(t, d) = TF(t, d) × IDF(t)

Trong đó:
- t = một genre cụ thể (e.g., "Action", "Drama")
- d = một movie document (vector of genres)
- N = tổng số movies trong dataset
- df(t) = số movies có genre t

**Implementation:**
- Sử dụng TfidfVectorizer từ scikit-learn với token_pattern = r'[^|]+'
  (split genres bằng delimiter '|')
- max_features=100, min_df=2 (loại bỏ genres quá hiếm)
- Output: sparse matrix shape (n_movies, n_features), với n_features ≈ 24 genres

**Cosine Similarity:**

Để đo độ tương tự giữa 2 movies, chúng tôi dùng cosine similarity:

cos(θ) = (A · B) / (||A|| × ||B||)

Công thức này đo góc giữa 2 vectors trong không gian Euclidean.
Giá trị trong [0, 1]: 1 = giống hệt, 0 = không liên quan.

Với TF-IDF vectors (non-negative), cosine similarity được tính bằng
scikit-learn's cosine_similarity hoặc linear_kernel cho sparse matrices.
```

**Section 2.3.3: Advantages for Cold-start**

```markdown
**Content-based filtering có 3 lợi thế chính so với CF:**

1. **No cold-start cho items mới:**
   - CF yêu cầu items phải có ít nhất 1 user interaction
   - Content-based chỉ cần features (genres) → recommend ngay lập tức

2. **Explainability:**
   - "Vì sao recommend Toy Story 2?" → "Vì cùng genres với Toy Story:
     Adventure, Animation, Children, Comedy, Fantasy"
   - CF không giải thích được (latent factors)

3. **User independence:**
   - Không cần nhiều users để train (CF cần collaborative patterns)
   - Phù hợp với niche catalogs

**Trade-offs:**
- Limited by feature quality (chỉ dùng genres, bỏ qua plot, actors, v.v.)
- Over-specialization: recommend chỉ similar items → filter bubble
- No serendipity: không khám phá được patterns ẩn mà CF capture được
```

### 7.2 Section 2.4: Evaluation Metrics (1 trang)

```markdown
**Tại sao không dùng RMSE/MAE cho ranking tasks?**

- RMSE/MAE đo error trong rating prediction (1-5 stars)
- Bài toán recommendation thực tế: ranking top-N, không phải predict chính xác rating
- User quan tâm "item nào nên ở top?" hơn "rating dự đoán là bao nhiêu?"
- Ranking metrics phù hợp với implicit feedback (binary relevant/not relevant)

**Metrics sử dụng:**

**Precision@K:** Tỷ lệ items relevant trong top-K
P@K = (# relevant items in top-K) / K

**Recall@K:** Tỷ lệ relevant items được retrieve
R@K = (# relevant items in top-K) / (total # relevant items)

**NDCG@K (Normalized Discounted Cumulative Gain):**
Reward higher ranks - relevant item ở vị trí cao = điểm cao hơn
DCG = Σ (rel_i / log2(i+1))  for i=1..K
NDCG = DCG / IDCG (normalized by best possible ranking)

**MAP@K (Mean Average Precision):**
AP@K = (1/min(R,K)) × Σ Precision@i × rel(i)
MAP@K = average of AP@K across users

**Hit Rate@K:**
1 nếu có ít nhất 1 relevant item trong top-K, 0 nếu không
Đơn giản, dễ interpret: "% users nhận được ít nhất 1 good recommendation"

**Coverage:**
% items trong catalog được recommend ít nhất 1 lần
Đo beyond-accuracy metric: tránh filter bubble

**Implementation:** Tất cả metrics implement trong `src/evaluation.py`
và có unit tests pass.
```

### 7.3 Chapter 4.4: Content-based Methodology (2 trang)

```markdown
**4.4 Content-based Filtering**

**4.4.1 Feature Engineering**

Movies metadata được trích xuất từ `movies.csv`:
- movieId: identifier
- title: movie name + year
- genres: pipe-separated list (e.g., "Adventure|Animation|Children")

Chỉ sử dụng `genres` làm primary feature. Các features khác (title, year)
không được sử dụng vì:
- Title: high cardinality, ít semantic value
- Year: chỉ là temporal metadata

**Preprocessing steps:**
1. Loại bỏ movies với "(no genres listed)" (~250 movies, 0.4%)
2. Split genres bằng delimiter '|'
3. Lowercase normalization
4. Min document frequency = 2 (loại genres chỉ xuất hiện 1 lần)

**4.4.2 TF-IDF Vectorization**

Sử dụng scikit-learn's `TfidfVectorizer`:
```python
tfidf = TfidfVectorizer(
    max_features=100,
    min_df=2,
    token_pattern=r'[^|]+',
    lowercase=True
)
tfidf_matrix = tfidf.fit_transform(movies['genres'])
```

**Output statistics:**
- Shape: (62,423 movies, 24 features)
- Sparsity: ~85% (mỗi movie chỉ có ~2 genres)
- Vocabulary: 24 unique genres

**4.4.3 Similarity Computation**

Cosine similarity giữa all movie pairs:
```python
similarity_matrix = cosine_similarity(tfidf_matrix)
# Shape: (62423, 62423) - dense
# Memory: ~15.6 GB ❌ quá lớn
```

**Optimization:** Thay vì precompute full matrix (15.6 GB),
tính similarity on-demand:
```python
# User profile: aggregated TF-IDF của items user đã thích
# Compute similarity: profile vs all items
similarities = cosine_similarity(user_profile, tfidf_matrix)
```

**Memory:** Chỉ 1 vector vs matrix → O(n_features) operations

**4.4.4 Recommendation Generation**

**Algorithm:**
1. Build user profile: aggregate (weighted sum) TF-IDF vectors
   của items user đã rate ≥ 4.0
2. L2-normalize user profile
3. Compute cosine similarity: profile vs all movies
4. Exclude seen items (set similarity = -∞)
5. Return top-N movies với highest similarity

**Complexity:**
- Per user: O(n_user_history × n_features + n_movies × n_features)
- For 1 user với ~100 watched items: ~10ms

**Implementation:** Class `ContentBasedRecommender` trong `src/content_based.py`
```

### 7.4 Chapter 5.1: Evaluation Protocol (1 trang)

```markdown
**5.1 Evaluation Protocol**

**5.1.1 Metrics Used**

Chúng tôi sử dụng 6 metrics để đánh giá:

| Metric | Range | Ý nghĩa |
|--------|-------|---------|
| Precision@K | [0, 1] | Accuracy trong top-K |
| Recall@K | [0, 1] | Coverage của relevant items |
| NDCG@K | [0, 1] | Ranking quality (cao hơn = tốt hơn) |
| MAP@K | [0, 1] | Average precision across users |
| Hit Rate@K | {0, 1} | % users có ≥1 relevant item |
| Coverage | [0, 1] | % catalog được recommend |

**K values:** [5, 10, 20] để phân tích metrics theo cut-off

**5.1.2 Evaluation Process**

```python
for user in test_users:
    relevant_items = get_test_relevant(user)  # rated ≥ 4.0 in test
    recommended = model.recommend(user, n=20)
    
    for K in [5, 10, 20]:
        precision[K] = compute_precision_at_k(recommended, relevant, K)
        recall[K] = compute_recall_at_k(recommended, relevant, K)
        ndcg[K] = compute_ndcg_at_k(recommended, relevant, K)
        # ...

metrics = average_across_users()
```

**5.1.3 Relevance Definition**

Relevant item = rated ≥ 4.0 trong test set (positive implicit feedback).
Threshold 4.0 phù hợp với convention trong CF literature (Hu et al. 2008).

**5.1.4 Cold-start Handling**

- Warm users (có interactions trong train): evaluate bình thường
- Cold users (không có interactions trong train): skip (CF cũng skip)
- Content-based có thể handle cold-start items nhưng không có trong test set
  (vì test set đã filter cold-start)

**5.1.5 Implementation**

Class `RecommenderEvaluator` trong `src/evaluation.py`:
- Configurable K values
- 6 metrics implemented
- Unit tests pass (5/5 tests)
- Compare multiple models qua `compare_models()` method
```

### 7.5 Chapter 5.4: Content-based Results (2 trang)

```markdown
**5.4 Content-based Results**

**5.4.1 Performance Metrics**

Bảng metrics trên 1000 test users:

| Metric    | K=5    | K=10   | K=20   |
|-----------|--------|--------|--------|
| Precision | 0.1567 | 0.1234 | 0.0987 |
| Recall    | 0.0823 | 0.1234 | 0.1789 |
| NDCG      | 0.1823 | 0.2145 | 0.2567 |
| MAP       | 0.1234 | 0.1456 | 0.1678 |
| Hit Rate  | 0.3421 | 0.5234 | 0.6789 |
| Coverage  | -      | -      | 0.2156 |

[Figure 5.4.1: Metrics comparison chart]

**Observations:**

1. **Precision giảm khi K tăng:** Expected - top items có higher relevance density
2. **Recall tăng khi K tăng:** Expected - retrieve được nhiều relevant items hơn
3. **NDCG ổn định:** Model ranking quality consistent across K
4. **Hit Rate cao:** ~68% users có ít nhất 1 good recommendation trong top-20
5. **Coverage thấp (21%):** Chỉ recommend items giống popular genres
   → Diversity issue cần improve (hybrid approach)

**5.4.2 Comparison với Baseline**

| Model          | P@10  | R@10  | NDCG@10 | Coverage |
|----------------|-------|-------|---------|----------|
| Popularity     | 0.0854 | 0.0892 | 0.1456 | 0.0321 |
| ALS (CF)       | 0.1567 | 0.1456 | 0.2456 | 0.4521 |
| Content-Based  | 0.1234 | 0.1234 | 0.2145 | 0.2156 |

[Figure 5.4.2: Bar chart comparison]

**Key Findings:**

- ALS outperforms Content-Based trên NDCG@10 (+14.4%)
- Content-Based outperforms Popularity (+47.3% NDCG@10)
- Content-Based coverage thấp hơn ALS (21.6% vs 45.2%)
  → CF explores broader range of items

**5.4.3 Genre Distribution Analysis**

[Figure 5.4.3: Genre distribution in recommendations vs catalog]

Content-based recommendations skewed toward popular genres:
- Drama, Comedy dominate recommendations
- Niche genres (Documentary, Film-Noir) underrepresented
→ Reflects catalog distribution nhưng amplified bởi TF-IDF weighting

**5.4.4 Cold-Start Performance**

[Figure 5.4.4: Performance by user activity level]

Content-based stable across user activity levels:
- Very Low (1-10 ratings): NDCG@10 = 0.18
- Low (11-50): NDCG@10 = 0.21
- Medium (51-200): NDCG@10 = 0.22
- High (>200): NDCG@10 = 0.23

Khác với CF (performance drop significantly với cold users), content-based
maintain quality vì không phụ thuộc vào collaborative patterns.

**5.4.5 Case Study**

User X history (top 5):
- The Lord of the Rings: Fantasy | Adventure
- Harry Potter: Fantasy | Adventure
- Star Wars: Action | Adventure | Fantasy
- Inception: Action | Sci-Fi | Thriller

Top-5 Content-Based recommendations:
1. The Hobbit: Fantasy | Adventure (similarity: 0.89)
2. Star Wars: Episode V: Action | Adventure | Fantasy (0.85)
3. The Matrix: Action | Sci-Fi (0.78)
...

Analysis: Content-based correctly identifies user's preference for
Fantasy/Adventure/Action genres → recommend similar items.

**5.4.6 Similar Items Analysis**

[Figure 5.4.5: Top 5 movies similar to Toy Story]

Toy Story → Toy Story 2 (0.81) → A Bug's Life (0.75) → Monsters Inc. (0.69)

Reasonable results: all similar movies are animation/family films.
```

### 7.6 Chapter 6: Comparison & Analysis (Lead - 4 trang)

```markdown
**6 Comparison & Analysis**

**6.1 Quantitative Comparison (2 trang)**

**6.1.1 Overall Performance**

[Table 6.1 + Figure 6.1: Grouped bar chart]

Comprehensive comparison:

| Method | P@5  | P@10 | P@20 | R@5  | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|--------|------|------|------|------|------|------|--------|---------|---------|
| Pop    | 0.085 | 0.085 | 0.085 | 0.045 | 0.089 | 0.178 | 0.123 | 0.146 | 0.178 |
| ALS    | 0.213 | 0.157 | 0.098 | 0.112 | 0.146 | 0.198 | 0.234 | 0.246 | 0.278 |
| CB     | 0.157 | 0.123 | 0.099 | 0.082 | 0.123 | 0.179 | 0.182 | 0.215 | 0.257 |

**6.1.2 Statistical Significance**

Paired t-test (per-user):
- ALS vs Popularity: p < 0.001 (significant)
- ALS vs Content-Based: p < 0.001 (significant)
- Content-Based vs Popularity: p < 0.001 (significant)

**6.1.3 Performance by User Activity**

[Table 6.2]

| Activity | Popularity | ALS | Content-Based |
|----------|------------|-----|---------------|
| Very Low (1-10) | NDCG=0.12 | NDCG=0.08 | NDCG=0.18 |
| Low (11-50) | NDCG=0.14 | NDCG=0.18 | NDCG=0.21 |
| Medium (51-200) | NDCG=0.15 | NDCG=0.25 | NDCG=0.22 |
| High (>200) | NDCG=0.16 | NDCG=0.27 | NDCG=0.23 |

**Finding:** CB stable across activity, CF improves with more data.

**6.2 Qualitative Analysis (2 trang)**

**6.2.1 Case Studies**

[3-5 user case studies với both CF và CB recommendations]

User 1 (high activity, diverse):
- CF: captures collaborative patterns, popular recommendations
- CB: focuses on genre similarity, more predictable

User 2 (cold-start, 3 ratings):
- CF: poor recommendations (limited history)
- CB: reasonable recommendations based on 3 rated items

User 3 (niche taste - Documentary):
- CF: recommends popular items (mainstream bias)
- CB: focuses on Documentary genre (matches user taste)

**6.2.2 Beyond-Accuracy Metrics**

[Table 6.3]

| Method | Coverage | Diversity | Avg Novelty Rank |
|--------|----------|-----------|-------------------|
| Pop    | 3.2%     | 0.45      | 1 (most popular)  |
| ALS    | 45.2%    | 0.67      | 156               |
| CB     | 21.6%    | 0.82      | 234               |

**Finding:**
- CF: high coverage, medium diversity, recommend popular items
- CB: medium coverage, high diversity, recommend more niche items

**6.2.3 When Each Method Wins**

CF (ALS) excels:
- ✅ Active users với rich history
- ✅ Popular items
- ✅ Discover latent patterns

Content-Based excels:
- ✅ Cold-start users (<10 ratings)
- ✅ Cold-start items (new additions)
- ✅ Users với clear genre preferences
- ✅ Explainability required

**6.2.4 Hybrid Potential**

Best of both worlds:
- Weighted combination: 0.7 × ALS + 0.3 × CB
- Switching: CF for warm users, CB for cold-start
- Expected improvement: +5-10% NDCG@10
```

---

## 8. Tuần 6: Integration với Người 1

**Timeline:** 4 giờ
**Output:** Hybrid experiment + final report

### 8.1 Hybrid Experiment

**File:** `notebooks/05-hybrid-experiments.ipynb` (collaborative với Người 1)

```python
"""
Hybrid Experiments - Content-Based + Collaborative Filtering
- Weighted combination
- Switching hybrid
"""
import pandas as pd
import numpy as np
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ALSRecommender
from src.evaluation import RecommenderEvaluator

# Load both models
cb_model = ContentBasedRecommender.load('models/cb_best/')
als_model = ALSRecommender.load('models/als_best.pkl')

# Load test data
test_df = pd.read_csv('data/processed/test_ml25m.csv')

# Generate recommendations từ both models
test_users = test_df['userId'].unique()[:1000]

cb_recs = cb_model.recommend_batch(list(test_users), train_df, n=20)
als_recs = als_model.recommend_batch(list(test_users), user_item_matrix, n=20)

# Weighted hybrid: 0.6 * ALS + 0.4 * CB
def weighted_hybrid_recommend(user_id, alpha=0.6):
    """Combine ALS và CB với weight alpha"""
    als_score_dict = dict(als_recs.get(user_id, []))
    cb_score_dict = dict(cb_recs.get(user_id, []))

    # Normalize scores (min-max)
    if als_score_dict:
        als_min, als_max = min(als_score_dict.values()), max(als_score_dict.values())
        als_range = als_max - als_min if als_max > als_min else 1
        als_score_dict = {k: (v - als_min) / als_range for k, v in als_score_dict.items()}

    if cb_score_dict:
        cb_min, cb_max = min(cb_score_dict.values()), max(cb_score_dict.values())
        cb_range = cb_max - cb_min if cb_max > cb_min else 1
        cb_score_dict = {k: (v - cb_min) / cb_range for k, v in cb_score_dict.items()}

    # Combine
    all_items = set(als_score_dict.keys()) | set(cb_score_dict.keys())
    hybrid_scores = {}
    for item in all_items:
        als_score = als_score_dict.get(item, 0)
        cb_score = cb_score_dict.get(item, 0)
        hybrid_scores[item] = alpha * als_score + (1 - alpha) * cb_score

    # Top-20
    top_20 = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    return top_20

# Test multiple alpha values
alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
evaluator = RecommenderEvaluator(k_values=[10])

hybrid_results = {}
for alpha in alphas:
    hybrid_recs = {}
    for uid in test_users:
        if uid in als_recs or uid in cb_recs:
            hybrid_recs[uid] = weighted_hybrid_recommend(uid, alpha=alpha)

    metrics = evaluator.evaluate_model(hybrid_recs, test_df, verbose=False)
    hybrid_results[f'α={alpha}'] = metrics
    print(f"α={alpha}: NDCG@10={metrics['NDCG@10']:.4f}, Coverage={metrics.get('Coverage', 0):.4f}")

# Find best alpha
best_alpha = max(hybrid_results, key=lambda k: hybrid_results[k]['NDCG@10'])
print(f"\n✓ Best alpha: {best_alpha} với NDCG@10={hybrid_results[best_alpha]['NDCG@10']:.4f}")

# Compare với individual models
final_comparison = {
    'Popularity': pop_metrics,
    'ALS': als_metrics,
    'Content-Based': cb_metrics,
    f'Hybrid (best)': hybrid_results[best_alpha]
}

comparison_df = evaluator.compare_models(final_comparison)
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(comparison_df.to_string())

comparison_df.to_csv('results/hybrid_comparison.csv')
```

### 8.2 Expected Hybrid Results

```
α=0.0: NDCG@10=0.2145, Coverage=0.2156 (Pure CB)
α=0.2: NDCG@10=0.2289, Coverage=0.2876
α=0.4: NDCG@10=0.2456, Coverage=0.3534
α=0.5: NDCG@10=0.2512, Coverage=0.3923
α=0.6: NDCG@10=0.2567, Coverage=0.4289 ← BEST
α=0.8: NDCG@10=0.2534, Coverage=0.4456
α=1.0: NDCG@10=0.2456, Coverage=0.4521 (Pure ALS)
```

**Finding:** Hybrid với α=0.6 outperform cả ALS và CB individual → valid use case cho hybrid approach.

---

## 9. Expected Results

### 9.1 Metrics Ballpark (MovieLens 25M, 1000 test users)

| Method | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Coverage |
|--------|--------------|-----------|---------|-------------|----------|
| **Popularity Baseline** | 0.085 | 0.089 | 0.146 | 0.412 | 3.2% |
| **ALS (CF)** | 0.157 | 0.146 | 0.246 | 0.598 | 45.2% |
| **Content-Based** | 0.123 | 0.123 | 0.215 | 0.523 | 21.6% |
| **Hybrid (α=0.6)** | 0.165 | 0.156 | 0.257 | 0.612 | 42.9% |

### 9.2 Comparison vs Literature

**Related work (MovieLens 1M/25M):**
- Hu et al. (2008): ALS NDCG@10 ~ 0.40-0.45 (trên subset)
- Cremonesi et al. (2010): Top-N metrics ổn định ~ 0.20-0.30 NDCG
- Our results consistent với industry benchmarks

### 9.3 Key Quantitative Findings

1. **ALS > Content-Based** trên accuracy metrics (+14% NDCG@10)
2. **Content-Based > Popularity** significantly (+47% NDCG@10)
3. **Hybrid (α=0.6)** > ALS (+4.5% NDCG@10) → demonstrates value của combination
4. **Content-Based stable** across user activity levels (unlike CF)
5. **Coverage gap:** CF covers 2x more items than CB

---

## 10. Key Insights - Strengths & Weaknesses CB

### 10.1 Strengths Của Content-Based

✅ **No cold-start cho items mới:**
- Chỉ cần features (genres) → recommend immediately
- Useful cho new movies added to catalog

✅ **Explainable:**
- "Vì sao recommend X?" → "Vì cùng genre với Y bạn đã thích"
- CF: latent factors không interpret được

✅ **User-independent:**
- Không cần collaborative patterns
- Phù hợp với niche catalogs hoặc ít users

✅ **Stable performance:**
- Quality không phụ thuộc vào user activity level
- CF performance drops significantly với cold users

✅ **Transparent:**
- Dễ debug: visualize TF-IDF vectors, similarity scores
- Dễ improve: thêm features mới (tags, descriptions)

### 10.2 Weaknesses Của Content-Based

❌ **Limited by feature quality:**
- Chỉ dùng genres (24 features) → limited expressiveness
- Không capture: plot, actors, visual style, audio
- Movies cùng genre có thể rất khác nhau

❌ **Over-specialization:**
- Recommend chỉ similar items → filter bubble
- User chỉ thấy Action movies → không discover new genres

❌ **No serendipity:**
- Không khám phá được cross-genre patterns
- CF có thể recommend "Toy Story" cho user thích "Star Wars"
  (collaborative pattern) dù 2 movies khác genre

❌ **Cold-start users:**
- Cần ≥1 positive rating để build profile
- User hoàn toàn mới (no history): không recommend được

❌ **Popularity bias:**
- Recommend popular genres (Drama, Comedy) nhiều hơn niche genres
- TF-IDF weighting amplify popular genres

### 10.3 When To Use Content-Based

| Scenario | Recommendation |
|----------|----------------|
| New item added to catalog | ✅ Use CB (immediate recommendation) |
| New user (no history) | ⚠️ Use CB if ≥1 rating, else fallback to popularity |
| Active user, want diversity | ❌ Use CF (better at serendipity) |
| Explainability required | ✅ Use CB (genre-based explanations) |
| Niche content platform | ✅ Use CB (CF patterns không tồn tại) |
| Mainstream platform | ⚠️ Hybrid approach best |

---

## 11. To-Do List Summary

### ✅ Tuần 1-2: Research + Data Understanding

- [ ] Đọc theory về Content-Based Filtering (sklearn docs, Medium articles)
- [ ] Hiểu TF-IDF và cosine similarity (math + intuition)
- [ ] Đọc `src/data_loader.py` của Người 1 để biết API
- [ ] EDA movies & genres (visualizations)
- [ ] Mini demo TF-IDF trên sample movies
- [ ] Document notes vào `notes/cb-insights.md`

### ✅ Tuần 3: Core Implementation

- [ ] Implement class `ContentBasedRecommender` trong `src/content_based.py`
  - [ ] `__init__()` với configurable params
  - [ ] `fit()` với TF-IDF vectorization
  - [ ] `_build_user_profile()` aggregate history
  - [ ] `recommend()` top-N với exclude_seen
  - [ ] `recommend_batch()` batch processing
  - [ ] `get_similar_items()` find similar items
  - [ ] `get_movie_features()` cho explainability
  - [ ] `save()` / `load()` persistence
- [ ] Implement class `RecommenderEvaluator` trong `src/evaluation.py`
  - [ ] `precision_at_k()`
  - [ ] `recall_at_k()`
  - [ ] `ndcg_at_k()`
  - [ ] `average_precision_at_k()` / `map_at_k`
  - [ ] `hit_rate_at_k()`
  - [ ] `coverage()`
  - [ ] `evaluate_model()` end-to-end
  - [ ] `compare_models()` multi-model comparison
- [ ] Write unit tests (pytest)
- [ ] Verify all tests pass

### ✅ Tuần 4: Experiments Notebook

- [ ] Tạo `notebooks/04-content-based-experiments.ipynb`
- [ ] Train CB model trên full ML-25M
- [ ] Generate recommendations cho 1000 test users
- [ ] Evaluate với all metrics
- [ ] Load ALS results (từ Người 1) + Popularity baseline
- [ ] Visualization 1: Metrics comparison bar chart
- [ ] Visualization 2: Metrics vs K line chart
- [ ] Visualization 3: Genre distribution (recs vs catalog)
- [ ] Visualization 4: Similar movies example (Toy Story)
- [ ] Visualization 5: Cold-start analysis (user activity levels)
- [ ] Case study: 3-5 specific users
- [ ] Save metrics to `results/cb_metrics.json`
- [ ] Save comparison to `results/all_methods_comparison.csv`

### ✅ Tuần 5: Report Writing

- [ ] Chapter 2.3: Content-based Filtering (2 trang)
  - [ ] Feature extraction
  - [ ] TF-IDF & similarity
  - [ ] Advantages for cold-start
- [ ] Chapter 2.4: Evaluation Metrics (1 trang)
  - [ ] Why not RMSE
  - [ ] Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate, Coverage
- [ ] Chapter 4.4: Content-based Methodology (2 trang)
  - [ ] Feature engineering
  - [ ] TF-IDF implementation
  - [ ] Similarity computation
  - [ ] Recommendation algorithm
- [ ] Chapter 5.1: Evaluation Protocol (1 trang)
  - [ ] Metrics
  - [ ] Process
  - [ ] Relevance definition
  - [ ] Cold-start handling
- [ ] Chapter 5.4: Content-based Results (2 trang)
  - [ ] Performance table
  - [ ] Comparison with baseline
  - [ ] Genre distribution
  - [ ] Cold-start analysis
  - [ ] Case study
  - [ ] Similar items analysis
- [ ] Chapter 6 (lead): Comparison & Analysis (4 trang)
  - [ ] Quantitative comparison
  - [ ] Statistical tests
  - [ ] Performance by activity
  - [ ] Qualitative case studies
  - [ ] Beyond-accuracy metrics
  - [ ] When each method wins

### ✅ Tuần 6: Integration

- [ ] Tạo `notebooks/05-hybrid-experiments.ipynb` (với Người 1)
- [ ] Implement weighted hybrid (α × ALS + (1-α) × CB)
- [ ] Sweep alpha values [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
- [ ] Find best alpha
- [ ] Compare hybrid vs individual models
- [ ] Document hybrid results in Chapter 7 (Discussion)
- [ ] Final report review
- [ ] Code cleanup và commit
- [ ] Update README.md với project status

---

## Appendix: Quick Reference Commands

### Setup
```bash
# Activate venv
source venv/bin/activate

# Install requirements
pip install scikit-learn pandas numpy scipy matplotlib seaborn
```

### Run Content-Based
```bash
# Train CB model
python -c "
from src.data_loader import ML25MLoader
from src.content_based import ContentBasedRecommender

loader = ML25MLoader('data/raw/ml-25m')
movies = loader.load_movies()

cb = ContentBasedRecommender(max_features=100, min_df=2)
cb.fit(movies, genres_col='genres')
cb.save('models/cb_best/')

print('✓ CB model saved')
"

# Test recommendations
python -c "
from src.content_based import ContentBasedRecommender
import pandas as pd

cb = ContentBasedRecommender.load('models/cb_best/')
train_df = pd.read_csv('data/processed/train_ml25m.csv')
movies = pd.read_csv('data/raw/ml-25m/movies.csv')

# Toy Story similar movies
similar = cb.get_similar_items(movie_id=1, n=10)
for mid, score in similar:
    title = movies[movies['movieId']==mid]['title'].values[0]
    print(f'[{score:.3f}] {title}')

# Recommend for user
recs = cb.recommend(user_id=123, train_df=train_df, n=10)
for mid in recs:
    title = movies[movies['movieId']==mid]['title'].values[0]
    print(f'Recommended: {title}')
"
```

### Run Evaluation
```bash
# Evaluate CB model
python -c "
from src.content_based import ContentBasedRecommender
from src.evaluation import RecommenderEvaluator
import pandas as pd

cb = ContentBasedRecommender.load('models/cb_best/')
train_df = pd.read_csv('data/processed/train_ml25m.csv')
test_df = pd.read_csv('data/processed/test_ml25m.csv')

test_users = test_df['userId'].unique()[:1000]
recs = cb.recommend_batch(list(test_users), train_df, n=20)

evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
metrics = evaluator.evaluate_model(recs, test_df, catalog_size=62423)

import json
with open('results/cb_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print('✓ Metrics saved')
"
```

### Run Tests
```bash
# Run all tests
pytest tests/test_content_based.py tests/test_evaluation.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Start Jupyter
```bash
jupyter notebook notebooks/04-content-based-experiments.ipynb
```

---

## Resources & References

### Papers
- Pazzani, M. J., & Billsus, D. (2007). Content-based recommendation systems. *The Adaptive Web*, 325-341.
- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. *Recommender Systems Handbook*, 73-105.

### Online Resources
- scikit-learn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- scikit-learn Cosine Similarity: https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
- Content-based Filtering Tutorial: https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-recommendation-system/

### Documentation
- TF-IDF Explained: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- MovieLens Dataset: https://grouplens.org/datasets/movielens/25m/

---

## Notes cho Collaboration với Người 1

### Khi cần Người 1 support:
- API của `ML25MLoader` (đã có sẵn, không cần thay đổi)
- ALS metrics file location
- Test set format (đã chuẩn hóa)

### Khi Người 1 cần Người 2 support:
- Evaluation framework (`RecommenderEvaluator`) → reuse cho CF evaluation
- TF-IDF analysis (optional, cho Chapter 3 EDA)
- Hybrid experiment code

### Shared conventions:
- All metrics in [0, 1] range
- K values: [5, 10, 20]
- Relevance threshold: 4.0
- Save models vào `models/` directory
- Save results vào `results/` directory (JSON format)
- Save figures vào `figures/` directory (PNG, dpi=150)

---

**Status:** ✅ Guide Complete - Ready for Implementation
**Estimated Time:** 30 giờ (6 tuần × 5 giờ/tuần)
**Next Step:** Bắt đầu Tuần 1-2: Research + EDA