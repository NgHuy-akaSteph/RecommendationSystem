"""Tiện ích hợp nhất điểm số: chuẩn hoá theo hàng, popularity prior, time decay.

Dùng chung cho hybrid late fusion và cho thí nghiệm popularity prior trên từng mô hình.
"""

import numpy as np

from src import paths
from src.data.build_interaction_matrices import TrainData

EPS = 1e-12


def minmax_row(scores: np.ndarray, candidate_mask: np.ndarray | None = None) -> np.ndarray:
    """Đưa mỗi hàng về [0, 1], tính min/max **chỉ trên các item ứng viên**.

    Bắt buộc trước khi cộng điểm của nhiều mô hình: thang điểm của chúng không so được
    (cosine của profile nằm trong [−1,1], còn tổng sim của item-KNN có thể lên hàng chục).

    `candidate_mask` không phải tuỳ chọn trang trí. Với mô hình nội dung, item mà user đã
    xem chính là item điểm cao nhất — và evaluator loại bỏ đúng những item đó ngay sau khi
    chấm. Nếu lấy min/max trên toàn bộ 13.816 item thì `hi` do một item đã xem quyết định,
    nên thang điểm của phần ứng viên lại phụ thuộc vào khoảng cách "đã xem / chưa xem"
    riêng của từng mô hình. Khi đó trọng số w = 0,75 **không** còn nghĩa là "75% mô hình A"
    mà là "75% của một thang do chính A tự đặt", và bảng quét trọng số mất ý nghĩa.

    Tự nó **không** đổi thứ hạng trong một user (biến đổi đơn điệu theo hàng), nên dùng
    riêng lẻ là vô hại; chỉ phép cộng sau đó mới đổi thứ hạng. Hàng hằng số (mọi điểm bằng
    nhau, ví dụ user không có item nào rating ≥ 4) trả về toàn 0 thay vì NaN.

    Args:
        scores: [B, n_items], phải hữu hạn — evaluator chỉ ghi −inf **sau** khi chấm điểm.
        candidate_mask: [B, n_items] bool, True = item được phép gợi ý. Hàng không có ứng
            viên nào (không xảy ra trên dữ liệu thật) rơi về chuẩn hoá trên toàn hàng.
    """
    if not np.isfinite(scores).all():
        raise ValueError("minmax_row nhận điểm không hữu hạn; evaluator chỉ ghi -inf sau khi chấm")

    if candidate_mask is None:
        lo = scores.min(axis=1, keepdims=True)
        hi = scores.max(axis=1, keepdims=True)
    else:
        usable = candidate_mask | ~candidate_mask.any(axis=1, keepdims=True)
        masked = np.where(usable, scores, np.nan)
        lo = np.nanmin(masked, axis=1, keepdims=True)
        hi = np.nanmax(masked, axis=1, keepdims=True)
    # Item đã xem có thể ra ngoài [0,1]; không sao, evaluator ghi -inf lên chúng ngay sau đó
    return ((scores - lo) / np.maximum(hi - lo, EPS)).astype("float32")


def popularity_prior(train: TrainData) -> np.ndarray:
    """Vector prior [n_items] trong [0, 1] theo log độ phổ biến trong train.

    Dùng log1p vì phân bố popularity của MovieLens lệch phải rất mạnh (phim đầu bảng
    nhiều hơn median vài nghìn lần); cộng thẳng count sẽ biến prior thành toàn bộ tín hiệu.
    Chuẩn hoá về [0,1] để λ_pop có cùng đơn vị với điểm đã minmax, nhờ đó so sánh được
    giữa các mô hình.

    Popularity đếm trên train của mô hình, không đụng test.
    """
    log_pop = np.log1p(train.item_popularity.astype("float32"))
    return (log_pop / max(float(log_pop.max()), EPS)).astype("float32")


def time_decay_weights(train: TrainData, half_life_days: float) -> np.ndarray:
    """Hệ số suy giảm theo thời gian cho từng tương tác train, khớp thứ tự `train_csr.data`.

    w = 0.5 ** (Δt / half_life), Δt = số ngày từ tương tác tới lần train cuối của user.
    Dùng half-life thay vì λ mũ vì dễ diễn giải trong báo cáo ("sau N ngày còn một nửa").
    """
    df = train.train_df
    csr = train.train_csr
    user_col = df["user_idx"].to_numpy()
    item_col = df["item_idx"].to_numpy()
    # Khoá gộp rồi argsort một lần: nhanh hơn lexsort 29× trên split thật (0,84s vs 24,5s)
    order = np.argsort(user_col.astype("int64") * train.n_items + item_col, kind="stable")

    # Kiểm tra thẳng vào từng cặp (user, item). So độ dài là vô nghĩa: df và csr.data luôn
    # cùng số phần tử (trùng lặp đã bị _to_csr chặn), nên guard theo độ dài không bao giờ
    # kích hoạt — đúng chế độ hỏng cần bắt lại là **sai thứ tự**, không phải sai độ dài.
    csr_rows = np.repeat(np.arange(csr.shape[0], dtype="int64"), np.diff(csr.indptr))
    if not (np.array_equal(user_col[order], csr_rows)
            and np.array_equal(item_col[order], csr.indices)):
        raise ValueError(
            "train_df sau khi sắp không khớp thứ tự (user, item) của train_csr.data; "
            "hệ số time-decay sẽ rơi nhầm ô"
        )

    ts = df["ts"].to_numpy()[order].astype("float64")
    users = user_col[order]

    last_ts = np.zeros(train.n_users, dtype="float64")
    np.maximum.at(last_ts, users, ts)
    delta_days = (last_ts[users] - ts) / 86400.0
    return np.power(0.5, delta_days / max(half_life_days, EPS)).astype("float32")


def relevant_mask(train: TrainData) -> np.ndarray:
    """Mặt nạ tương tác relevant, khớp thứ tự `train_csr.data`."""
    return train.train_csr.data >= paths.RELEVANCE_THRESHOLD
