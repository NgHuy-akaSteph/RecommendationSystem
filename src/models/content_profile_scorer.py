"""Scorer dùng hồ sơ Rocchio: P = normalize(W @ X); S = P @ Xᵀ.

Kết hợp với bất kỳ item matrix X nào (TF-IDF, genome, BM25). Profile P là vector
trung bình có trọng số của các item user đã tương tác trong không gian đặc trưng,
chuẩn hoá L2 để cosine sim = tích vô hướng.

Hai quyết định về bộ nhớ:
    1. Profile tính **per-batch** trong score_users thay vì precompute toàn bộ 162k
       user tại fit() — tránh OOM khi D lớn.
    2. X được giữ nguyên **sparse** nếu đầu vào sparse. TF-IDF bigram có mật độ
       0.14%: bản dense tốn 2.06GB (peak 8.28GB khi build) trong khi CSR chỉ ~6MB.
       Phép nhân cũng nhanh hơn vì chi phí tỉ lệ với nnz chứ không phải I×D.
"""

import numpy as np
from scipy import sparse

from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.user_history_weights import build_user_weights
from src.models.vector_normalization import safe_l2_normalize


class ContentProfileScorer(BaseRecommender):
    """Rocchio profile scorer dùng chung cho mọi item representation X."""

    name = "content-profile"

    def __init__(self, weight_mode: str = "binary_pos",
                 half_life_days: float | None = None):
        self.weight_mode = weight_mode
        self.half_life_days = half_life_days
        self._X = None                                # [I, D] sparse hoặc dense, đã L2-norm
        self._W: sparse.csr_matrix | None = None      # [U, I] sparse weights

    def fit(self, train: TrainData, item_features: dict) -> "ContentProfileScorer":
        """Cần item_features["X"] — ma trận L2-normalised [I, D] (sparse hoặc dense)."""
        self._X = item_features["X"]
        self._W = build_user_weights(train, self.weight_mode, self.half_life_days)
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        """Tính profile on-the-fly per batch rồi nhân với X.

        P_batch = normalize(W[batch] @ X)  → [B, D]
        S_batch = (X @ P_batchᵀ)ᵀ         → [B, I]

        Nhân theo chiều ``X @ Pᵀ`` thay vì ``P @ Xᵀ``: khi X là CSR, scipy chỉ duyệt
        nnz của X; chuyển vị một CSR để làm toán tử trái sẽ tốn thêm một lần chuyển
        đổi sang CSC.
        """
        profiles = safe_l2_normalize(self._W[user_idx] @ self._X)   # [B, D]
        scores = self._X @ profiles.T                               # [I, B]
        if sparse.issparse(scores):
            scores = scores.toarray()
        return np.ascontiguousarray(np.asarray(scores).T, dtype="float32")
