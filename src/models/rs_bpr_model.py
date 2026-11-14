"""RS-BPR: BPR-MF của RS, port sang giao thức batch chung.

Bản gốc (`src/recommenders/bpr_recommender.py`) dùng `implicit.bpr.BayesianPersonalizedRanking`
nhưng trên **ma trận do chính nó dựng** từ df rating dương (`DataPreprocessor.build_user_item_matrix`),
nên catalog lệch 13.816 phim của pipeline chung và không dùng được chung bộ nhận xếp hạng.

Bản port giữ nguyên thuật toán (BPR-MF, SGD trên triplet) nhưng:
    - dùng ma trận **nhị phân relevant** lấy từ `train.train_csr` (rating ≥ 4) — đúng
      nghĩa "phản hồi ẩn" của BPR, và đúng ngưỡng với nhãn test;
    - dùng chung catalog `train.n_items` (13.816);
    - chấm điểm theo lô: `S[u, j] = U[u] · V[j]`.

BPR tối ưu trực tiếp hàm ranking nên về lý thuyết phù hợp Top-N hơn ALS — đây là lý do
giữ nó trong bảng CF, không phải để trùng lặp ALS.

**Chi phí (đo trên catalog 13.816 phim, 10,07M tương tác relevant, 12 core):**

| cấu hình | thời gian fit | tái lập được |
|---|---|---|
| 30 epoch, 12 luồng | 20s | **không** (race khi lấy mẫu âm) |
| 30 epoch, 1 luồng | 121s | có |
| 100 epoch, 12 luồng | 65s | không |

Mặc định của lớp là `num_threads=1`: mất 6× thời gian để đổi lại tính tái lập, đổi lại
mỗi dòng số liệu trong báo cáo chạy lại được sau này.

Mọi tham số đều lấy từ `paths.SEED` / `paths.RELEVANCE_THRESHOLD` để không lệch với
phần còn lại của pipeline.
"""

import numpy as np
from scipy import sparse

from src import paths
from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.seen_item_masking import mask_seen_items


class RsBprModel(BaseRecommender):
    """BPR-MF trên ma trận nhị phân relevant."""

    name = "rs-bpr"

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        iterations: int = 30,
        verify_negative_samples: bool = True,
        random_state: int = paths.SEED,
        num_threads: int = 1,
    ):
        """
        Args:
            factors: số chiều latent.
            learning_rate: tốc độ SGD.
            regularization: cường độ L2.
            iterations: số epoch SGD. Bản gốc dùng 100; đo thực tế trên catalog
                13.816 phim / 10,07M tương tác relevant: 100 epoch ≈ 65s (12 luồng).
            verify_negative_samples: kiểm tra mẫu âm thực sự chưa tương tác.
            random_state: seed khởi tạo factor.
            num_threads: **mặc định 1 — bắt buộc để tái lập được**. `implicit` lấy mẫu
                âm bằng RNG dùng chung cho cả các luồng, nên với >1 luồng hai lần chạy
                cùng seed cho kết quả khác nhau (đo được: `num_threads=2` đã cho hai lần
                chạy khác nhau). Mất đi tính tái lập để lấy 6× tốc độ là cái giá không
                đáng trong một pipeline nghiên cứu — số liệu trong báo cáo phải chạy lại
                được sau này. Nếu cần tốc độ, đặt `num_threads=0` và **ghi rõ** vào
                báo cáo rằng dòng đó không tái lập được.
        """
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.verify_negative_samples = verify_negative_samples
        self.random_state = random_state
        self.num_threads = num_threads
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._train_csr: sparse.csr_matrix | None = None

    def fit(self, train: TrainData, item_features: dict) -> "RsBprModel":
        from implicit.bpr import BayesianPersonalizedRanking

        model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            verify_negative_samples=self.verify_negative_samples,
            random_state=self.random_state,
            num_threads=self.num_threads,
        )
        binary = build_binary_relevant_matrix(train)
        print(f"  [BPR] fit {binary.shape[0]:,} user × {binary.shape[1]:,} item, "
              f"{binary.nnz:,} tương tác relevant, factors={self.factors}, "
              f"iters={self.iterations}")
        model.fit(binary, show_progress=True)

        self._user_factors = np.asarray(model.user_factors, dtype="float32")
        self._item_factors = np.asarray(model.item_factors, dtype="float32")
        self._train_csr = train.train_csr
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        if self._user_factors is None:
            raise RuntimeError("gọi fit() trước score_users()")
        scores = (self._user_factors[user_idx] @ self._item_factors.T).astype("float32")
        return mask_seen_items(scores, self._train_csr, user_idx)


def build_binary_relevant_matrix(train: TrainData) -> sparse.csr_matrix:
    """B [n_users, n_items] nhị phân: 1 nếu rating ≥ ngưỡng relevant, 0 nếu khác.

    BPR là mô hình phản hồi ẩn: "đã tương tác" là tín hiệu, còn mức độ thích thì không
    dùng. Nếu giữ cả rating thấp với giá trị 1.0, mô hình học "user thích phim này"
    cho đúng những phim user chấm 1 sao — sai ngữ nghĩa.

    Trả về CSR float32; `eliminate_zeros()` để mọi ô 0 thật sự biến mất khỏi ma trận
    (nếu không, `implicit` coi đó là tương tác đã xem).
    """
    binary = train.train_csr.copy()
    binary.data = (binary.data >= paths.RELEVANCE_THRESHOLD).astype("float32")
    binary.eliminate_zeros()
    return binary
