"""Xây trọng số lịch sử người dùng W [n_users, n_items] từ ma trận train.

Ba chế độ:
    - binary_pos: 1 nếu rating ≥ 4, 0 ngược lại. Rocchio đơn giản, không phân biệt mức thích.
    - rating: giữ nguyên giá trị rating. Phim 5 sao được đóng góp nhiều hơn 4 sao.
    - centered: r − ū_train. Phim không thích có trọng số âm, kéo profile ra xa thể loại đó.
"""

import numpy as np
from scipy import sparse

from src import paths
from src.data.build_interaction_matrices import TrainData

WEIGHT_MODES = ("binary_pos", "rating", "centered")


def build_user_weights(train: TrainData, mode: str = "binary_pos",
                       half_life_days: float | None = None) -> sparse.csr_matrix:
    """Tạo ma trận trọng số W [n_users, n_items] theo chế độ chỉ định.

    Args:
        train: dữ liệu huấn luyện (train_csr chứa rating).
        mode: một trong ``binary_pos``, ``rating``, ``centered``.
        half_life_days: nếu đặt, nhân thêm hệ số suy giảm theo thời gian
            ``0.5 ** (Δt / half_life)`` — phim xem lâu rồi đóng góp ít hơn vào hồ sơ.

    Returns:
        CSR matrix float32 [n_users, n_items]. Các ô không tương tác giữ giá trị 0
        (kể cả ở chế độ centered — dùng 0 thay vì −ū vì user chưa xem phim đó nên
        không có ý kiến, khác với xem rồi mà không thích).
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"mode phải là một trong {WEIGHT_MODES}, nhận '{mode}'")

    csr = train.train_csr.copy()

    if mode == "binary_pos":
        csr.data = (csr.data >= paths.RELEVANCE_THRESHOLD).astype("float32")

    elif mode == "rating":
        csr.data = csr.data.astype("float32")

    elif mode == "centered":
        # ū = trung bình rating train của mỗi user (chỉ trên item đã xem)
        user_means = per_user_mean(csr)
        csr.data = csr.data.astype("float32")
        # np.repeat theo indptr, không dùng nonzero(): nonzero() bỏ qua các số 0 được
        # lưu tường minh nên độ dài lệch với csr.data và phép trừ sẽ nổ broadcast.
        rows = np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))
        csr.data -= user_means[rows]

    if half_life_days is not None:
        from src.models.score_fusion_utils import time_decay_weights  # tránh import vòng
        # time_decay_weights tự kiểm tra thẳng từng cặp (user, item) trước khi trả về
        csr.data *= time_decay_weights(train, half_life_days)

    csr.eliminate_zeros()
    return csr


def per_user_mean(csr: sparse.csr_matrix) -> np.ndarray:
    """Trung bình rating mỗi user, trả về mảng [n_users]. User rỗng → 0.

    Public vì genome_ridge_user_model cũng cần đúng phép này (DRY).
    """
    counts = np.diff(csr.indptr)
    sums = np.asarray(csr.sum(axis=1)).ravel()
    means = np.zeros(csr.shape[0], dtype="float32")
    mask = counts > 0
    means[mask] = sums[mask] / counts[mask]
    return means
