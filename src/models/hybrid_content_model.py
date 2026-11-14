"""CB-4: Hybrid late fusion nhiều mô hình + popularity prior.

    S = Σ_m w_m · minmax_row(S_m)  +  λ_pop · pop_norm

Một lớp phục vụ hai mục đích, không cần hai lớp:
    - **1 component + λ_pop > 0** = thí nghiệm popularity prior trên đúng một mô hình.
      Đây là trục đối chứng bắt buộc: phase 2 cho thấy CB đẩy item hiếm lên đầu (median
      popularity top-10 là 3.109/8.230 so với 49.556 của MostPop) và số tag mỗi phim
      tương quan 0,77 với popularity, nên phải tách được "nội dung thật" khỏi "popularity
      đội lốt nội dung" trước khi quy công cho hybrid.
    - **nhiều component** = hybrid late fusion đúng nghĩa.

Mô hình con được dựng qua `model_registry` nên mọi cấu hình đã chạy ở phase 2 tái dùng
nguyên vẹn, không sao chép tham số.
"""

import numpy as np

from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.score_fusion_utils import minmax_row, popularity_prior


class HybridLateFusionModel(BaseRecommender):
    """Cộng điểm đã chuẩn hoá của nhiều mô hình con, tuỳ chọn thêm popularity prior."""

    name = "cb4-hybrid"

    def __init__(
        self,
        components: list[dict],
        weights: list[float] | None = None,
        pop_prior: float = 0.0,
    ):
        """
        Args:
            components: mỗi phần tử ``{"model": tên_trong_registry, "params": {...}}``.
            weights: trọng số từng component; mặc định chia đều. Được chuẩn hoá về tổng 1
                để λ_pop giữ nguyên ý nghĩa khi số component thay đổi.
            pop_prior: λ_pop, cộng vào điểm đã minmax. 0 = content thuần.
        """
        if not components:
            raise ValueError("hybrid cần ít nhất 1 component")
        if weights is not None and len(weights) != len(components):
            raise ValueError(f"có {len(components)} component nhưng {len(weights)} trọng số")

        self.components = components
        raw_weights = np.asarray(weights if weights is not None
                                 else [1.0] * len(components), dtype="float32")
        total = float(raw_weights.sum())
        if total <= 0:
            raise ValueError("tổng trọng số phải dương")
        self.weights = raw_weights / total
        self.pop_prior = pop_prior
        self._models: list[BaseRecommender] = []
        self._pop: np.ndarray | None = None
        self._train_csr = None

    def fit(self, train: TrainData, item_features: dict) -> "HybridLateFusionModel":
        # import cục bộ: registry import chính lớp này nên import ở đầu file sẽ vòng
        from src.eval.model_registry import build_model

        if any(c["model"] == self.name for c in self.components):
            raise ValueError("không lồng hybrid trong hybrid: config vòng sẽ đệ quy vô hạn")

        self._train_csr = train.train_csr
        self._models = [
            build_model(c["model"], c.get("params")).fit(train, item_features)
            for c in self.components
        ]
        self._pop = popularity_prior(train) if self.pop_prior else None
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("gọi fit() trước score_users()")

        # Chuẩn hoá trên đúng tập ứng viên (bỏ item đã xem) — xem minmax_row
        seen = self._train_csr[user_idx].toarray() > 0
        candidate_mask = ~seen

        scores = None
        for weight, model in zip(self.weights, self._models):
            part = weight * minmax_row(model.score_users(user_idx), candidate_mask)
            scores = part if scores is None else scores + part

        if self._pop is not None:
            scores = scores + self.pop_prior * self._pop  # broadcast [I] lên [B, I]
        return scores.astype("float32")
