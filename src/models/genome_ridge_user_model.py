"""CB-3 tối ưu: Per-user Ridge Regression trên genome-SVD.

Thay vì dùng profile trung bình cố định, mô hình này học một vector trọng số w_u
riêng cho từng user bằng ridge regression trên không gian genome-SVD:

    Z = genome_svd [I, d], d=64
    Với user u: Z_u = Z[items_u], y_u = r_u − ū
    w_u = solve(Z_uᵀ Z_u + λI, Z_uᵀ y_u)
    S_u = Z @ w_u + ū

User có < min_train item sẽ fallback về profile scorer (ridge không ổn định với ít dữ liệu).

Việc gom Z_uᵀZ_u phải chạy vòng lặp vì số item mỗi user khác nhau (ragged), nhưng
**phép solve và phép chấm điểm đều batch**: một ``np.linalg.solve`` trên stack [B,d,d]
và một matmul [I,d]@[d,B]. Gọi BLAS-3 một lần thay vì B lần BLAS-2.
"""

import numpy as np

from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.content_profile_scorer import ContentProfileScorer
from src.models.genome_content_model import build_genome_item_matrix
from src.models.user_history_weights import per_user_mean

RIDGE_BATCH = 500  # số user mỗi lần solve (RAM: B × d × d × 4 bytes)


class GenomeRidgeUserModel(BaseRecommender):
    """Per-user ridge regression trên genome-SVD."""

    name = "cb3-ridge"

    def __init__(
        self,
        svd_dim: int = 64,
        standardize: str | None = None,
        threshold: float | None = None,
        ridge_lambda: float = 10.0,
        min_train: int = 5,
        weight_mode: str = "binary_pos",
    ):
        self.svd_dim = svd_dim
        self.standardize = standardize
        self.threshold = threshold
        self.ridge_lambda = ridge_lambda
        self.min_train = min_train
        self.weight_mode = weight_mode
        self._Z: np.ndarray | None = None        # [I, d] genome-SVD L2-normed
        self._train_csr = None
        self._user_means: np.ndarray | None = None
        self._fallback: ContentProfileScorer | None = None
        self._n_items = 0

    def fit(self, train: TrainData, item_features: dict) -> "GenomeRidgeUserModel":
        self._Z = build_genome_item_matrix(
            item_features, self.standardize, self.threshold, self.svd_dim
        )
        self._train_csr = train.train_csr.tocsr()
        self._n_items = train.n_items
        self._user_means = per_user_mean(self._train_csr)

        # Fallback scorer cho user ít dữ liệu
        self._fallback = ContentProfileScorer(weight_mode=self.weight_mode)
        self._fallback.fit(train, {**item_features, "X": self._Z})
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        """Ridge scoring theo batch, fallback cho user có ít item."""
        scores = np.empty((len(user_idx), self._n_items), dtype="float32")

        nnz_per_user = np.diff(self._train_csr.indptr)[user_idx]
        ridge_mask = nnz_per_user >= self.min_train

        if (~ridge_mask).any():
            scores[~ridge_mask] = self._fallback.score_users(user_idx[~ridge_mask])

        ridge_positions = np.flatnonzero(ridge_mask)
        for start in range(0, len(ridge_positions), RIDGE_BATCH):
            batch_pos = ridge_positions[start: start + RIDGE_BATCH]
            scores[batch_pos] = self._ridge_batch(user_idx[batch_pos])

        return scores

    def _ridge_batch(self, user_idx: np.ndarray) -> np.ndarray:
        """Solve batch [B,d,d] một lần, rồi chấm điểm bằng một matmul [I,d]@[d,B]."""
        Z = self._Z
        d = Z.shape[1]
        B = len(user_idx)
        indptr, indices, data = self._train_csr.indptr, self._train_csr.indices, self._train_csr.data

        # Gom hệ phương trình: ragged theo user nên phải lặp, nhưng mỗi vòng chỉ là d×d
        ZtZ = np.tile((self.ridge_lambda * np.eye(d, dtype="float32")), (B, 1, 1))
        Zty = np.empty((B, d), dtype="float32")
        for i, u in enumerate(user_idx):
            lo, hi = indptr[u], indptr[u + 1]
            Z_u = Z[indices[lo:hi]]                             # [n_u, d]
            y_u = data[lo:hi].astype("float32") - self._user_means[u]
            ZtZ[i] += Z_u.T @ Z_u
            Zty[i] = Z_u.T @ y_u

        try:
            W = np.linalg.solve(ZtZ, Zty[:, :, None])[:, :, 0]  # [B, d]
        except np.linalg.LinAlgError:
            # λ = 0 có thể sinh ma trận suy biến; hạ về lstsq cho riêng user hỏng
            W = np.stack([np.linalg.lstsq(ZtZ[i], Zty[i], rcond=None)[0] for i in range(B)])

        # S = Z @ Wᵀ + ū  → [I, B] rồi chuyển vị. Hằng số ū không đổi thứ hạng trong
        # một user nhưng vẫn cộng để điểm giữ đúng thang rating (phase 4 cần).
        scores = (Z @ W.T).T + self._user_means[user_idx][:, None]
        return np.ascontiguousarray(scores, dtype="float32")
