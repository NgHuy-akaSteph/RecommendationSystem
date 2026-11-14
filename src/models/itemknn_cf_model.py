"""Baseline CF: ItemKNN cosine trên ma trận tương tác (Sarwar et al. 2001).

Điểm mấu chốt: đây **cùng một thuật toán** với CB item-KNN của phase 2, chỉ khác cách
biểu diễn item. Content-based dùng X = đặc trưng nội dung [I × D]; collaborative dùng
X = train_csrᵀ [I × U], tức mỗi item được mô tả bằng "những ai đã thích nó". Vì
`ContentItemKnnScorer` nhận X sparse bất kỳ nên toàn bộ phần tính sim block-wise, cắt
top-k và cache được tái dùng nguyên vẹn — lớp này chỉ dựng X.

Sự đối xứng đó cũng là một luận điểm của báo cáo: khoảng cách CB vs CF ở đây thuần tuý
đến từ biểu diễn item, không lẫn khác biệt về scorer hay giao thức đánh giá.
"""

import numpy as np
from scipy import sparse

from src import paths
from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.content_item_knn_scorer import ContentItemKnnScorer
from src.models.vector_normalization import safe_l2_normalize


class ItemKnnCfModel(BaseRecommender):
    """Item-item cosine trên đồng xuất hiện user, dùng lại scorer của phase 2."""

    name = "cf-itemknn"

    def __init__(self, k_neighbors: int = 100, weight_mode: str = "binary_pos",
                 normalize_sim: bool = False):
        # Chỉ nhận binary_pos: X phía item là nhị phân chỉ gồm tương tác relevant, nên
        # weight_mode="rating" sẽ cho phim 1 sao trọng số dương ở phía user trong khi
        # không gian item hoàn toàn bỏ qua nó — bất đối xứng âm thầm, không phải thứ
        # muốn có trong một baseline dùng để so sánh.
        if weight_mode != "binary_pos":
            raise ValueError(
                f"ItemKnnCfModel chỉ hỗ trợ weight_mode='binary_pos', nhận '{weight_mode}'")
        self.k_neighbors = k_neighbors
        self.weight_mode = weight_mode
        self.normalize_sim = normalize_sim
        self._scorer: ContentItemKnnScorer | None = None

    def fit(self, train: TrainData, item_features: dict) -> "ItemKnnCfModel":
        X = build_cf_item_matrix(train)
        self._scorer = ContentItemKnnScorer(
            sim_name="cf_itemknn",          # vân tay nội dung X phân biệt các train khác nhau
            k_neighbors=self.k_neighbors,
            weight_mode=self.weight_mode,
            normalize_sim=self.normalize_sim,
        )
        self._scorer.fit(train, {**item_features, "X": X})
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return self._scorer.score_users(user_idx)


def build_cf_item_matrix(train: TrainData) -> sparse.csr_matrix:
    """X [n_items, n_users] nhị phân, L2-norm theo hàng → cosine item-item.

    Chỉ giữ tương tác relevant (rating ≥ 4), thống nhất với MostPop và với nhãn test:
    một phim bị chấm 1 sao không nói lên rằng hai người cùng xem nó thì giống nhau.
    """
    binary = train.train_csr.copy()
    binary.data = (binary.data >= paths.RELEVANCE_THRESHOLD).astype("float32")
    binary.eliminate_zeros()
    return safe_l2_normalize(binary.T.tocsr())
