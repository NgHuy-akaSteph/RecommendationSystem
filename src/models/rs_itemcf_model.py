"""RS-Item-CF: Item-based CF của RS, port sang giao thức batch chung.

Bản gốc (`src/recommenders/item_cf.py`) dùng **cosine đầy đủ** trên ma trận
item×user rồi chấm điểm `Σ_{i đã xem} sim(i, j) · rating_i`. Ở catalog 13.816 phim,
ma trận cosine đầy đủ dense float32 là 763 MB — không thể dựng trong bộ nhớ của
một tiến trình đánh giá. Bản port dùng lại bộ cắt top-k block-wise đã có sẵn
(`ContentItemKnnScorer`), nên:

    K[j, i] = cosine(i, j) trong top-k hàng xóm của j      (sparse, ~2,8M nnz)
    S[u, j] = Σ_{i ∈ N(j)} W[u, i] · K[j, i]  =  W @ Kᵀ

**Ba điểm khác bản gốc, đều phải nêu khi đọc bảng kết quả:**

1. **Hướng nhân.** Bản gốc tính `X @ (Xᵀ · r_u)`, tức `W @ K` — lấy hàng của item
   **user đã xem**. Bản port tính `W @ Kᵀ` — lấy hàng của item **ứng viên**. K bất
   đối xứng (top-k cắt theo hàng) nên hai phép này cho kết quả khác nhau; trên dev,
   `W @ Kᵀ` cao hơn `W @ K` khoảng 16% (CB-1) / 8% (CB-3). Sửa ở đây là **sửa hướng
   tính đúng** (chỉ `W @ Kᵀ` mới hội tụ về hồ sơ Rocchio khi k → I), không phải
   đổi thuật toán.
2. **Trọng số phía item.** Bản gốc đưa rating thật vào ma trận item×user. Bản port
   để phía item là **nhị phân relevant** (`build_cf_item_matrix`) và đặt trọng số
   rating ở **phía user** (`W`): một phim bị chấm 1 sao không nói lên rằng hai
   người cùng xem nó thì giống nhau, nhưng nó *có* tín hiệu về sở thích của chính
   user đó. Đây là điểm khác biệt thật sự so với `cf-itemknn` của CB (cùng thuật
   toán, cùng hướng nhân, chỉ khác trọng số).
3. **Catalog.** Bản gốc tự dựng `movie_map` từ df rating dương nên lệch catalog;
   bản port dùng chung `train.n_items` (13.816) của CB.

Ngoài ra `score_users` tự mask item user đã xem (−∞). Evaluator cũng mask, nhưng
mô hình không được phụ thuộc vào điều đó — đặc biệt khi mô hình này được dùng như
thành phần của một lớp fusion khác.
"""

import numpy as np
from scipy import sparse

from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.content_item_knn_scorer import ContentItemKnnScorer
from src.models.itemknn_cf_model import build_cf_item_matrix
from src.models.seen_item_masking import mask_seen_items


class RsItemCfModel(BaseRecommender):
    """Item-CF trọng số rating, chấm theo hướng `W @ Kᵀ` trên top-k hàng xóm."""

    name = "rs-itemcf"

    def __init__(self, k_neighbors: int = 200, weight_mode: str = "rating",
                 normalize_sim: bool = False):
        """
        Args:
            k_neighbors: số hàng xóm giữ lại mỗi item (K).
            weight_mode: trọng số phía **user**. ``"rating"`` giữ nguyên giá trị rating
                (khác `cf-itemknn` của CB vốn dùng ``"binary_pos"``); ``"binary_pos"``
                dùng cho ablation.
            normalize_sim: chia điểm cho Σ|sim| của các hàng xóm user đã tương tác.
        """
        self.k_neighbors = k_neighbors
        self.weight_mode = weight_mode
        self.normalize_sim = normalize_sim
        self._scorer: ContentItemKnnScorer | None = None
        self._train_csr: sparse.csr_matrix | None = None

    def fit(self, train: TrainData, item_features: dict) -> "RsItemCfModel":
        # X phía item: nhị phân relevant, L2-norm theo hàng → cosine item-item.
        X = build_cf_item_matrix(train)
        self._scorer = ContentItemKnnScorer(
            sim_name="rs_itemcf",       # vân tay X tách khỏi cache của cf-itemknn
            k_neighbors=self.k_neighbors,
            weight_mode=self.weight_mode,
            normalize_sim=self.normalize_sim,
        )
        self._scorer.fit(train, {"X": X})
        self._train_csr = train.train_csr
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        if self._scorer is None:
            raise RuntimeError("gọi fit() trước score_users()")
        scores = self._scorer.score_users(user_idx)
        return mask_seen_items(scores, self._train_csr, user_idx)
