"""Hai baseline không cá nhân hoá: Random và MostPop.

Random là mốc dưới (sanity check: NDCG@10 phải gần 0). MostPop là mốc tối thiểu mà
mọi mô hình cá nhân hoá phải vượt qua - trên dữ liệu phim, đây là baseline mạnh
một cách đáng ngạc nhiên (Cremonesi et al., 2010).
"""

import numpy as np

from src import paths
from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Xếp hạng ngẫu nhiên, dùng để kiểm tra pipeline đánh giá không bị lỗi.

    Điểm của mỗi user sinh từ một dòng ngẫu nhiên riêng (SeedSequence theo user_idx)
    nên kết quả không phụ thuộc vào cách chia lô - điều kiện để tái lập được ở phase 6.
    """

    name = "random"

    def __init__(self, seed: int = paths.SEED):
        self.seed = seed
        self.n_items = 0

    def fit(self, train: TrainData, item_features: dict) -> "RandomRecommender":
        self.n_items = train.n_items
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        scores = np.empty((len(user_idx), self.n_items), dtype="float32")
        for row, user in enumerate(user_idx):
            seed_seq = np.random.SeedSequence(entropy=self.seed, spawn_key=(int(user),))
            scores[row] = np.random.default_rng(seed_seq).random(self.n_items, dtype="float32")
        return scores


class MostPopRecommender(BaseRecommender):
    """Gợi ý theo số lượt rating relevant (>= 4) trong train, giống nhau cho mọi user."""

    name = "mostpop"

    def __init__(self):
        self.popularity: np.ndarray | None = None

    def fit(self, train: TrainData, item_features: dict) -> "MostPopRecommender":
        self.popularity = train.item_popularity.astype("float32")
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        assert self.popularity is not None, "Phải gọi fit() trước"
        # broadcast_to trả về view; evaluator sao chép trước khi ghi đè nên an toàn
        return np.broadcast_to(self.popularity, (len(user_idx), self.popularity.size))
