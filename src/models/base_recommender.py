"""Giao diện chung cho mọi mô hình gợi ý.

Tách `fit` khỏi `score_users` cho phép evaluator chấm điểm theo lô (batch), giữ
bộ nhớ ở mức [batch_size, n_items] thay vì [n_users, n_items] (162k x 13.8k float32
= 8.9GB, không khả thi).

`fit` chỉ nhận `TrainData`, không nhận `Dataset`: mô hình không có đường nào chạm
tới nhãn test, kể cả do gõ nhầm.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.data.build_interaction_matrices import TrainData


class BaseRecommender(ABC):
    """Lớp cơ sở. Mô hình con chỉ cần cài đặt `fit` và `score_users`."""

    name: str = "base"

    @abstractmethod
    def fit(self, train: TrainData, item_features: dict) -> "BaseRecommender":
        """Huấn luyện từ dữ liệu train."""

    @abstractmethod
    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        """Điểm dự đoán cho một lô user.

        Args:
            user_idx: mảng chỉ số user [batch_size].

        Returns:
            Ma trận float32 [batch_size, n_items]. Evaluator sẽ tự loại item đã có
            trong train, mô hình không cần xử lý việc đó. Evaluator luôn sao chép
            kết quả trước khi ghi đè nên mô hình được phép trả về mảng dùng lại.
        """
