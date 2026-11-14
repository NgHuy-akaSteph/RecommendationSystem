"""Các chỉ số xếp hạng top-K cho bài toán gợi ý.

Định nghĩa dùng relevance nhị phân (item relevant <=> rating >= 4 trong test),
theo Järvelin & Kekäläinen (2002) và Herlocker et al. (2004):

    P@k    = hits@k / k
    R@k    = hits@k / |Rel_u|
    HR@k   = 1 nếu hits@k > 0
    NDCG@k = DCG@k / IDCG@k,  DCG@k = sum_{i=1..k} rel_i / log2(i+1)
             IDCG@k = sum_{i=1..min(|Rel_u|, k)} 1 / log2(i+1)

Mọi hàm trả về mảng per-user (không phải trung bình) để phase 4 chạy được
Wilcoxon signed-rank và bootstrap CI mà không phải đánh giá lại mô hình.
"""

import numpy as np
from scipy import sparse

DEFAULT_K_LIST = (5, 10, 20, 50)


def gather_relevance(rel_csr: sparse.csr_matrix, topk_idx: np.ndarray) -> np.ndarray:
    """Tra cứu nhãn relevant cho các item được gợi ý.

    Args:
        rel_csr: ma trận nhị phân [n_users_eval, n_items], chỉ số hàng khớp topk_idx.
        topk_idx: [n_users_eval, k_max] chỉ số item đã sắp theo thứ hạng giảm dần.

    Returns:
        Mảng float32 [n_users_eval, k_max] gồm 0/1.
    """
    # Hàng rào catalog: item ngoài [0, n_items) không thể tra nhãn. Nếu không chặn,
    # `searchsorted` sẽ trả về vị trí hợp lệ nhưng so sánh luôn False → item đó âm
    # thầm được tính là "không trúng", biến một lỗi lập trình thành điểm số thấp hơn.
    n_items = rel_csr.shape[1]
    if topk_idx.size:
        lo, hi = int(topk_idx.min()), int(topk_idx.max())
        if lo < 0 or hi >= n_items:
            raise ValueError(
                f"Danh sách gợi ý chứa item ngoài catalog [0, {n_items}): "
                f"min={lo}, max={hi}. Item ngoài catalog không thể tra nhãn relevant — "
                "sửa lại cách ánh xạ item của mô hình."
            )

    # searchsorted yêu cầu chỉ số trong mỗi hàng đã sắp xếp; không sửa ma trận của caller
    rel_csr = rel_csr.tocsr()
    if not rel_csr.has_sorted_indices:
        rel_csr = rel_csr.copy()
        rel_csr.sort_indices()
    indptr, indices = rel_csr.indptr, rel_csr.indices

    flags = np.zeros(topk_idx.shape, dtype="float32")
    for row in range(topk_idx.shape[0]):
        row_items = indices[indptr[row] : indptr[row + 1]]
        if row_items.size == 0:
            continue
        pos = np.searchsorted(row_items, topk_idx[row])
        pos_clipped = np.minimum(pos, row_items.size - 1)
        flags[row] = (row_items[pos_clipped] == topk_idx[row]).astype("float32")
    return flags


def compute_metrics(
    rel_flags: np.ndarray, n_relevant: np.ndarray, k_list=DEFAULT_K_LIST
) -> dict[str, np.ndarray]:
    """Tính P/R/NDCG/HR@k per-user từ nhãn relevant của danh sách đã xếp hạng.

    Args:
        rel_flags: [n_users, k_max] nhãn 0/1 theo thứ hạng.
        n_relevant: [n_users] tổng số item relevant của user trong test.
        k_list: các giá trị k cần tính (phải <= k_max).

    Returns:
        dict tên_metric -> mảng [n_users] float32.
    """
    k_max = rel_flags.shape[1]
    if max(k_list) > k_max:
        raise ValueError(f"k={max(k_list)} vượt quá số item được xếp hạng ({k_max})")
    n_relevant = np.asarray(n_relevant, dtype="float32")
    if np.any(n_relevant <= 0):
        raise ValueError("Mọi user đánh giá phải có ít nhất 1 item relevant")

    discounts = 1.0 / np.log2(np.arange(2, k_max + 2, dtype="float64"))
    cum_hits = np.cumsum(rel_flags, axis=1)
    cum_dcg = np.cumsum(rel_flags * discounts, axis=1)
    cum_ideal = np.cumsum(discounts)  # IDCG tích luỹ cho trường hợp lý tưởng

    metrics: dict[str, np.ndarray] = {}
    for k in k_list:
        hits = cum_hits[:, k - 1]
        # IDCG@k = tổng discount của min(|Rel_u|, k) vị trí đầu
        n_ideal = np.minimum(n_relevant, k).astype("int32")
        idcg = cum_ideal[n_ideal - 1]

        metrics[f"P@{k}"] = (hits / k).astype("float32")
        metrics[f"R@{k}"] = (hits / n_relevant).astype("float32")
        metrics[f"HR@{k}"] = (hits > 0).astype("float32")
        metrics[f"NDCG@{k}"] = (cum_dcg[:, k - 1] / idcg).astype("float32")
    return metrics


def evaluate_ranking(
    topk_idx: np.ndarray, rel_csr: sparse.csr_matrix, k_list=DEFAULT_K_LIST
) -> dict[str, np.ndarray]:
    """Tiện ích: từ danh sách xếp hạng và ma trận relevant -> metric per-user."""
    rel_flags = gather_relevance(rel_csr, topk_idx)
    n_relevant = np.asarray(rel_csr.getnnz(axis=1), dtype="float32")
    return compute_metrics(rel_flags, n_relevant, k_list)


def summarize(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    """Trung bình theo user cho từng metric."""
    return {name: float(values.mean()) for name, values in metrics.items()}


def format_table(summary: dict[str, float], k_list=DEFAULT_K_LIST) -> str:
    """Bảng text gọn để in ra console."""
    header = f"{'metric':<8}" + "".join(f"{'@' + str(k):>10}" for k in k_list)
    lines = [header, "-" * len(header)]
    for name in ("P", "R", "NDCG", "HR"):
        row = f"{name:<8}" + "".join(f"{summary[f'{name}@{k}']:>10.4f}" for k in k_list)
        lines.append(row)
    return "\n".join(lines)
