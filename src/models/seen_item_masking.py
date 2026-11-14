"""Mask item user đã tương tác trong train, dùng chung cho mọi mô hình.

Hai lý do để có một chỗ duy nhất:

1. **Mô hình phải tự an toàn.** Evaluator luôn ghi −∞ lên item đã xem, nhưng mô hình
   không được phụ thuộc vào điều đó: một mô hình dùng làm thành phần của lớp fusion
   khác sẽ trả cả item đã xem ra ngoài nếu không tự loại.

2. **Sentinel phải hữu hạn.** `minmax_row` (dùng cho hybrid) ném `ValueError` khi thấy
   điểm không hữu hạn, và `implicit` cũng không chấp nhận −∞ trong ma trận trọng số.
   Vì vậy dùng một giá trị rất âm thay vì −∞: vẫn bị loại khỏi mọi top-k, nhưng vẫn
   là số hữu hạn nên phép cộng/minmax vẫn chạy được.

Giá trị chọn là −1e9: đủ âm để không bao giờ lọt vào top-k (mọi điểm ứng viên thực
sự có |score| ≪ 1e9), nhưng khi chia cho một hiệu nhỏ (hi−lo ~ 1e−6) vẫn nằm trong
float32 thay vì tràn.
"""

import numpy as np
from scipy import sparse

SEEN_ITEM_SCORE = -1e9


def mask_seen_items(
    scores: np.ndarray, train_csr: sparse.csr_matrix, user_idx: np.ndarray
) -> np.ndarray:
    """Ghi `SEEN_ITEM_SCORE` lên item user đã tương tác trong train (in-place).

    Args:
        scores: [batch_size, n_items], sẽ bị ghi đè tại chỗ.
        train_csr: ma trận train [n_users, n_items].
        user_idx: chỉ số user của batch, khớp với hàng của `scores`.

    Returns:
        Chính mảng `scores` đã ghi đè (để gọn khi viết `return mask_seen_items(...)`).
    """
    seen = train_csr[user_idx]
    rows, cols = seen.nonzero()
    scores[rows, cols] = SEEN_ITEM_SCORE
    return scores
