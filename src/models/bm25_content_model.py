"""CB-2: Mô hình nội dung dùng trọng số BM25 trên tag document.

BM25 khác TF-IDF ở hai chỗ, đúng hai chỗ mà tag document cần:
    - **Bão hoà tần suất**: c/(c + k1·…) chặn trên, nên một tag được 500 người gán
      không áp đảo tag được 5 người gán gấp 100 lần. `sublinear_tf` của TF-IDF cũng
      làm việc này nhưng bằng log, không có tham số điều chỉnh.
    - **Chuẩn hoá độ dài tài liệu** qua b: phim blockbuster có hàng nghìn tag không
      tự động giống mọi thứ. Điều này đáng chú ý ở đây vì số tag mỗi phim tương quan
      0,77 với độ phổ biến, tức độ dài tài liệu chính là popularity đội lốt nội dung.

Công thức (Robertson & Zaragoza 2009), N = số item, df_t = số item chứa tag t:

    idf_t = log((N − df_t + 0.5) / (df_t + 0.5) + 1)
    w_it  = idf_t · c_it·(k1 + 1) / (c_it + k1·(1 − b + b·len_i/avg_len))

Viết thẳng trên CSR (~10 dòng vector hoá) thay vì dùng `rank_bm25`: thư viện đó chạy
vòng lặp Python trên từng document, chậm hơn nhiều bậc và không trả về ma trận thưa.
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.content_item_knn_scorer import ContentItemKnnScorer
from src.models.content_profile_scorer import ContentProfileScorer
from src.models.feature_cache_key import cache_key
from src.models.vector_normalization import safe_l2_normalize

EPS = 1e-12


def bm25_weight(counts: sparse.csr_matrix, k1: float = 1.2, b: float = 0.75) -> sparse.csr_matrix:
    """Áp trọng số BM25 lên ma trận đếm [n_docs, n_terms]."""
    if k1 < 0 or not 0.0 <= b <= 1.0:
        # b > 1 đẩy mẫu số tiến sát 0 với document ngắn → trọng số bùng nổ
        raise ValueError(f"cần k1 >= 0 và b trong [0, 1], nhận k1={k1}, b={b}")
    counts = counts.tocsr().astype("float32")
    n_docs = counts.shape[0]

    df = np.asarray((counts > 0).sum(axis=0)).ravel().astype("float32")
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype("float32")

    doc_len = np.asarray(counts.sum(axis=1)).ravel().astype("float32")
    # avg_len tính trên **mọi** document, kể cả 188 phim không có tag. Điều này kéo avg_len
    # xuống (107,9 thay vì 109,4 nếu bỏ document rỗng) nên phạt độ dài nặng hơn ~1,4%.
    # Giữ nguyên theo định nghĩa BM25 gốc: "trung bình độ dài của bộ sưu tập".
    avg_len = max(float(doc_len.mean()), EPS)
    # Mẫu số phụ thuộc document nên tính sẵn theo hàng rồi bung ra từng phần tử nnz
    norm_per_doc = k1 * (1.0 - b + b * doc_len / avg_len)
    rows = np.repeat(np.arange(n_docs, dtype="int32"), np.diff(counts.indptr))

    weighted = counts.copy()
    c = counts.data
    weighted.data = (idf[counts.indices] * c * (k1 + 1.0) / (c + norm_per_doc[rows])).astype("float32")
    return weighted


class Bm25ContentModel(BaseRecommender):
    """X = hstack(genres, BM25(tag counts)) L2-norm → profile hoặc item-knn scorer."""

    name = "cb2-bm25"

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        min_df: int = 3,
        max_df: float = 0.5,
        max_features: int = 20_000,
        scorer: str = "profile",
        weight_mode: str = "binary_pos",
        k_neighbors: int = 50,
        half_life_days: float | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.scorer_type = scorer
        self.weight_mode = weight_mode
        self.k_neighbors = k_neighbors
        self.half_life_days = half_life_days
        self._scorer: BaseRecommender | None = None

    def fit(self, train: TrainData, item_features: dict) -> "Bm25ContentModel":
        X = self._build_X(item_features)
        if self.scorer_type == "profile":
            self._scorer = ContentProfileScorer(
                weight_mode=self.weight_mode, half_life_days=self.half_life_days)
        elif self.scorer_type == "item_knn":
            self._scorer = ContentItemKnnScorer(
                sim_name=self._sim_cache_name(), k_neighbors=self.k_neighbors,
                weight_mode=self.weight_mode, half_life_days=self.half_life_days)
        else:
            raise ValueError(f"scorer phải là 'profile' hoặc 'item_knn', nhận '{self.scorer_type}'")
        self._scorer.fit(train, {**item_features, "X": X})
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return self._scorer.score_users(user_idx)

    def _build_X(self, item_features: dict) -> sparse.csr_matrix:
        for key in ("genres", "tag_docs"):
            if key not in item_features:
                raise KeyError(
                    f"Thiếu đặc trưng '{key}'. Chạy: python -m src.data.build_item_content_features")
        genres = item_features["genres"].tocsr()
        tag_docs = item_features["tag_docs"]
        if not np.array_equal(tag_docs["item_idx"].to_numpy(), np.arange(len(tag_docs))):
            raise ValueError("tag_docs không thẳng hàng với item_idx; chạy lại build_item_content_features")

        counts = CountVectorizer(
            min_df=self.min_df, max_df=self.max_df, max_features=self.max_features,
            dtype=np.int32,
        ).fit_transform(tag_docs["tag_text"].fillna("").tolist())

        return safe_l2_normalize(
            sparse.hstack([genres, bm25_weight(counts, self.k1, self.b)], format="csr"))

    def _sim_cache_name(self) -> str:
        return cache_key("bm25", {"k1": self.k1, "b": self.b, "min_df": self.min_df,
                                  "max_df": self.max_df, "max_features": self.max_features})
