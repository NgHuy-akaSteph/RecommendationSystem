"""Sinh khoá cache ổn định từ tập tham số sinh ra ma trận đặc trưng.

Trước đây tên cache chỉ ghép vài tham số dễ đọc (ví dụ ``sim_tfidf_tags20000_k50``),
bỏ sót min_df/max_df/sublinear_tf. Hai cấu hình khác nhau ở các tham số đó sinh ra
cùng một tên file và **nạp nhầm cache một cách im lặng** — sai số không báo lỗi.
Hash toàn bộ tham số loại bỏ hẳn lớp lỗi này.
"""

import hashlib
import json

import numpy as np
from scipy import sparse


def cache_key(prefix: str, params: dict) -> str:
    """Trả về ``{prefix}_{hash10}``, ổn định giữa các lần chạy và các máy.

    Args:
        prefix: tiền tố dễ đọc để người còn nhận ra file (ví dụ ``tfidf``, ``genome``).
        params: mọi tham số ảnh hưởng tới nội dung ma trận. Thiếu một tham số nào
            ở đây là mở lại đúng lỗi nạp nhầm cache, nên phải liệt kê đủ.
    """
    blob = json.dumps(params, sort_keys=True, default=str)
    return f"{prefix}_{hashlib.md5(blob.encode('utf-8')).hexdigest()[:10]}"


def matrix_fingerprint(X) -> str:
    """Vân tay của **chính ma trận** X, không phụ thuộc vào tham số sinh ra nó.

    Hash theo tham số vẫn thủng hai đường: (1) người gọi quên liệt kê một tham số,
    (2) *code* dựng X đổi mà tham số không đổi (sửa công thức blend, nâng cấp sklearn
    làm đổi ``token_pattern`` mặc định). Cả hai đều cho cache cũ với shape khớp → nạp
    im lặng. Vân tay lấy từ nội dung X nên miễn nhiễm với cả hai.

    Lấy mẫu thưa dữ liệu (tối đa ~4096 phần tử) để chi phí không phụ thuộc kích thước X.
    """
    h = hashlib.md5()
    h.update(np.asarray(X.shape, dtype="int64").tobytes())
    if sparse.issparse(X):
        X = X.tocsr()
        data = X.data
        # Hash cả indices: hai ma trận cùng tập giá trị khác 0 nhưng khác vị trí là hai ma
        # trận khác nhau, và đường CF chỉ dựa vào vân tay này để tách các split
        idx = X.indices
        if idx.size:
            h.update(np.ascontiguousarray(idx[:: max(1, idx.size // 4096)]).tobytes())
    else:
        data = np.ascontiguousarray(X).ravel()
    h.update(np.asarray(data.shape, dtype="int64").tobytes())
    if data.size:
        h.update(np.ascontiguousarray(data[:: max(1, data.size // 4096)]).tobytes())
    return h.hexdigest()[:10]
