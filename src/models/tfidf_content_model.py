"""CB-1: Mô hình nội dung dựa trên TF-IDF (genres + tag do user gán).

Xây ma trận X [I, D] **sparse** từ 2 nguồn:
    1. genres one-hot (19 chiều, luôn có)
    2. TF-IDF trên tag document (0 — 40k chiều, tuỳ cấu hình)

X được L2-norm theo hàng → cosine sim = tích vô hướng → dùng được với cả
profile scorer và item-KNN scorer.

Giữ sparse xuyên suốt: mật độ thực đo là 0.26% (unigram) / 0.14% (bigram). Bản dense
tốn 0.68GB / 2.06GB và peak 2.75GB / 8.28GB lúc build, trong khi CSR chỉ ~6MB.

Guard NaN: 183 item không có tag → TF-IDF vector rỗng. Sau hstack với genres, một số
vẫn có thể norm = 0 (nếu cũng thiếu genres). ``safe_l2_normalize`` giữ hàng đó bằng 0.
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.content_item_knn_scorer import ContentItemKnnScorer
from src.models.content_profile_scorer import ContentProfileScorer
from src.models.feature_cache_key import cache_key
from src.models.vector_normalization import safe_l2_normalize


class TfidfContentModel(BaseRecommender):
    """Wrapper: build X từ config rồi uỷ quyền cho scorer (profile hoặc item-knn)."""

    name = "cb1-tfidf"

    def __init__(
        self,
        use_tags: bool = False,
        min_df: int = 3,
        max_df: float = 0.5,
        max_features: int = 20_000,
        sublinear_tf: bool = True,
        ngram_range: tuple[int, int] = (1, 1),
        genre_weight: float | None = None,
        scorer: str = "profile",
        weight_mode: str = "binary_pos",
        k_neighbors: int = 50,
        normalize_sim: bool = False,
        half_life_days: float | None = None,
    ):
        self.use_tags = use_tags
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.ngram_range = tuple(ngram_range)
        self.genre_weight = genre_weight
        self.scorer_type = scorer
        self.weight_mode = weight_mode
        self.k_neighbors = k_neighbors
        self.normalize_sim = normalize_sim
        self.half_life_days = half_life_days
        self._scorer: BaseRecommender | None = None

    def fit(self, train: TrainData, item_features: dict) -> "TfidfContentModel":
        X = self._build_X(item_features)
        features_with_X = {**item_features, "X": X}

        if self.scorer_type == "profile":
            self._scorer = ContentProfileScorer(
                weight_mode=self.weight_mode, half_life_days=self.half_life_days)
        elif self.scorer_type == "item_knn":
            self._scorer = ContentItemKnnScorer(
                k_neighbors=self.k_neighbors,
                weight_mode=self.weight_mode,
                sim_name=self._sim_cache_name(),
                normalize_sim=self.normalize_sim,
                half_life_days=self.half_life_days,
            )
        else:
            raise ValueError(f"scorer phải là 'profile' hoặc 'item_knn', nhận '{self.scorer_type}'")
        self._scorer.fit(train, features_with_X)
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return self._scorer.score_users(user_idx)

    # ── private ──────────────────────────────────────────────────────────

    def _build_X(self, item_features: dict) -> sparse.csr_matrix:
        """Build và L2-normalize item feature matrix (CSR)."""
        genres = _require(item_features, "genres").tocsr()

        if not self.use_tags:
            return safe_l2_normalize(genres)

        tag_docs = _require(item_features, "tag_docs")
        # Mọi ma trận đặc trưng thẳng hàng theo vị trí với item_idx. Nếu giả định này
        # vỡ thì X lệch hàng và sai toàn bộ mà không có lỗi nào được ném ra.
        if not np.array_equal(tag_docs["item_idx"].to_numpy(), np.arange(len(tag_docs))):
            raise ValueError("tag_docs không thẳng hàng với item_idx; chạy lại build_item_content_features")
        if len(tag_docs) != genres.shape[0]:
            raise ValueError(f"tag_docs {len(tag_docs)} dòng nhưng genres có {genres.shape[0]} item")

        vectorizer = TfidfVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            ngram_range=self.ngram_range,
            norm=None,  # chuẩn hoá thủ công sau hstack
            dtype=np.float32,
        )
        tfidf = vectorizer.fit_transform(tag_docs["tag_text"].fillna("").tolist())

        if self.genre_weight is not None:
            # Theo spec bước 3: hstack(α·genres_norm, (1−α)·tags_norm). Đây là một cách
            # tham số hoá **khác** với nhánh None (nối thẳng khối thô), nên α → 0 KHÔNG
            # hội tụ về baseline None mà về "chỉ TF-IDF đã chuẩn hoá, bỏ genres".
            # Hai nhánh vì thế không nằm trên cùng một đường cong khi đọc bảng ablation.
            alpha = self.genre_weight
            blocks = [alpha * safe_l2_normalize(genres), (1 - alpha) * safe_l2_normalize(tfidf)]
        else:
            blocks = [genres, tfidf]

        return safe_l2_normalize(sparse.hstack(blocks, format="csr"))

    def _sim_cache_name(self) -> str:
        """Khoá cache sim phải bao gồm **mọi** tham số ảnh hưởng tới X."""
        return cache_key("tfidf", {
            "use_tags": self.use_tags,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "ngram_range": self.ngram_range,
            "genre_weight": self.genre_weight,
        })


def _require(item_features: dict, key: str):
    if key not in item_features:
        raise KeyError(
            f"Thiếu đặc trưng '{key}'. Chạy: python -m src.data.build_item_content_features"
        )
    return item_features[key]
