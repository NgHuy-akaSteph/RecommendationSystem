"""Baseline CF: Alternating Least Squares cho phản hồi ẩn (Hu, Koren & Volinsky 2008).

Dùng `implicit.als.AlternatingLeastSquares` (CPU, đã xác minh wheel cp313 trên Windows).

Ngữ nghĩa confidence — chỗ dễ sai nhất:
`implicit` coi **giá trị lưu trong ma trận chính là c_ui**, rồi dùng `(c−1)` để tích luỹ
YᵀC_uY và `c` cho vế phải (xem `implicit.cpu.als.user_linear_equation`). Do đó:

    - phải lưu ``1 + alpha``, không phải ``alpha``;
    - **chỉ được lưu tương tác relevant**. Nếu lưu cả rating thấp với c = 1 thì số hạng
      `(c−1)` bằng 0 nhưng `b += 1·factor` vẫn cộng vào, tức mô hình học "user thích phim
      này với độ tin cậy 1" cho đúng những phim user đã chấm điểm thấp.

Ngưỡng relevant lấy từ `paths.RELEVANCE_THRESHOLD`, thống nhất với MostPop, ItemKNN-CF và
nhãn test.
"""

import numpy as np
from scipy import sparse

from src import paths
from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender


class AlsCfModel(BaseRecommender):
    """Matrix factorization phản hồi ẩn."""

    name = "cf-als"

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
        num_threads: int = 8,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.num_threads = num_threads
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None

    def fit(self, train: TrainData, item_features: dict) -> "AlsCfModel":
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False,
            num_threads=self.num_threads,
            random_state=paths.SEED,
        )
        with _single_threaded_blas():
            model.fit(build_confidence_matrix(train, self.alpha), show_progress=False)

        self._user_factors = np.asarray(model.user_factors, dtype="float32")
        self._item_factors = np.asarray(model.item_factors, dtype="float32")
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return (self._user_factors[user_idx] @ self._item_factors.T).astype("float32")


def build_confidence_matrix(train: TrainData, alpha: float) -> sparse.csr_matrix:
    """C [n_users, n_items] = 1 + alpha trên các tương tác relevant, 0 ở phần còn lại."""
    csr = train.train_csr.copy()
    csr.data = np.where(csr.data >= paths.RELEVANCE_THRESHOLD,
                        np.float32(1.0 + alpha), np.float32(0.0))
    csr.eliminate_zeros()
    return csr


def _single_threaded_blas():
    """Ép BLAS về 1 luồng trong lúc fit.

    ALS đã tự song song hoá theo user; để OpenBLAS mở thêm threadpool lồng nhau gây
    tranh chấp và chậm đi nhiều lần (chính `implicit` cũng cảnh báo điều này).
    """
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:  # không có threadpoolctl thì chạy bình thường, chỉ chậm hơn
        from contextlib import nullcontext
        return nullcontext()
