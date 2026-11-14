"""Chuẩn hoá L2 theo hàng, dùng chung cho mọi item matrix X và mọi user profile.

Tách riêng vì cả tfidf_content_model, genome_content_model, content_profile_scorer
đều cần đúng một phép này (DRY). Hàm xử lý được cả sparse lẫn dense và luôn an toàn
với hàng có norm = 0 (183 item không có tag → vector rỗng → cosine sinh NaN nếu chia
thẳng cho norm).
"""

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize as _sk_normalize

EPS = 1e-12


def safe_l2_normalize(X):
    """L2-normalize theo hàng. Hàng có norm = 0 được giữ nguyên bằng 0 (không NaN).

    Args:
        X: ma trận [n_rows, n_cols], sparse (bất kỳ format nào) hoặc dense ndarray.

    Returns:
        Cùng kiểu với đầu vào (sparse → CSR), dtype float32.
    """
    if sparse.issparse(X):
        # sklearn bỏ qua hàng toàn 0 thay vì chia cho 0 → an toàn sẵn
        return _sk_normalize(X.tocsr(), norm="l2", axis=1, copy=True).astype("float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(norms, EPS)).astype("float32")

