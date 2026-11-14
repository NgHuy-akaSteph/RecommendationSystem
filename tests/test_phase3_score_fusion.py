"""Test tiện ích hợp nhất điểm phase 3: minmax_row, popularity prior, time decay."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.data.build_interaction_matrices import TrainData
from src.models.score_fusion_utils import minmax_row, popularity_prior, time_decay_weights
from tests.conftest import _mock_train


def _train_with_ts(ts_offsets_days) -> TrainData:
    """TrainData 1 user, item 0..n-1, timestamp lùi dần theo ts_offsets_days."""
    n = len(ts_offsets_days)
    last = 1_600_000_000
    df = pd.DataFrame({
        "user_idx": np.zeros(n, dtype="int32"),
        "item_idx": np.arange(n, dtype="int32"),
        "rating": np.full(n, 5.0, dtype="float32"),
        "ts": [last - int(d * 86400) for d in ts_offsets_days],
    })
    csr = sparse.csr_matrix((df["rating"], (df["user_idx"], df["item_idx"])), shape=(1, n))
    csr.sort_indices()
    return TrainData(train_csr=csr, train_df=df, n_users=1, n_items=n)


# ── minmax_row ───────────────────────────────────────────────────────────

def test_minmax_row_preserves_within_user_ranking():
    """minmax theo hàng là biến đổi đơn điệu → không được đổi thứ hạng của user nào."""
    S = np.array([[3.0, -1.0, 7.0], [2.0, 2.0, 2.0]], dtype="float32")
    out = minmax_row(S)
    assert np.allclose(out[0], [0.5, 0.0, 1.0])
    assert np.allclose(out[1], 0.0)            # hàng hằng số → 0, không NaN
    assert np.array_equal(np.argsort(S[0]), np.argsort(out[0]))


def test_minmax_row_scales_by_candidates_only():
    """min/max phải lấy trên item ứng viên; item đã xem không được quyết định thang đo."""
    S = np.array([[10.0, 1.0, 3.0]], dtype="float32")
    mask = np.array([[False, True, True]])     # item 0 đã xem, bị loại
    out = minmax_row(S, mask)
    assert out[0, 1] == pytest.approx(0.0)     # ứng viên thấp nhất
    assert out[0, 2] == pytest.approx(1.0)     # ứng viên cao nhất
    assert out[0, 0] > 1.0                     # item đã xem tràn ra ngoài [0,1] — chấp nhận
    # Không có mask thì item đã xem nuốt hết thang đo và hai ứng viên bị nén lại
    naive = minmax_row(S)
    assert naive[0, 2] - naive[0, 1] < out[0, 2] - out[0, 1]


def test_minmax_row_all_seen_row_falls_back():
    S = np.array([[1.0, 2.0]], dtype="float32")
    out = minmax_row(S, np.array([[False, False]]))
    assert np.isfinite(out).all() and np.allclose(out[0], [0.0, 1.0])


def test_minmax_row_rejects_non_finite():
    with pytest.raises(ValueError, match="hữu hạn"):
        minmax_row(np.array([[1.0, -np.inf]], dtype="float32"))


# ── prior và time decay ──────────────────────────────────────────────────

def test_popularity_prior_range_and_order():
    train = _mock_train(n_users=20, n_items=10)
    prior = popularity_prior(train)
    assert prior.shape == (10,)
    assert prior.min() >= 0.0 and prior.max() == pytest.approx(1.0)
    assert np.array_equal(np.argsort(train.item_popularity), np.argsort(prior))


def test_time_decay_halves_after_one_half_life():
    train = _train_with_ts([0, 365, 730])
    decay = time_decay_weights(train, half_life_days=365.0)
    assert len(decay) == len(train.train_csr.data)
    assert decay[0] == pytest.approx(1.0)      # tương tác cuối: không suy giảm
    assert decay[1] == pytest.approx(0.5, rel=1e-4)
    assert decay[2] == pytest.approx(0.25, rel=1e-4)


def test_time_decay_tolerates_permuted_train_df():
    """Hoán vị thứ tự dòng của train_df là vô hại: phép sắp xếp bên trong khôi phục lại."""
    train = _train_with_ts([0, 365, 730])
    expected = time_decay_weights(train, half_life_days=365.0)
    train.train_df = train.train_df.iloc[[2, 0, 1]].reset_index(drop=True)
    assert np.allclose(time_decay_weights(train, half_life_days=365.0), expected)


def test_time_decay_detects_train_df_not_matching_csr():
    """Guard phải bắt được SAI CẶP (user, item), không phải chỉ sai độ dài.

    Guard cũ so `len(decay) != len(csr.data)` — hai giá trị này luôn bằng nhau (trùng lặp
    đã bị `_to_csr` chặn) nên nó là code chết: xoá hẳn phép sắp xếp vẫn không test nào đỏ.
    """
    train = _train_with_ts([0, 365, 730])
    # item 1 biến mất, item 0 xuất hiện hai lần → không còn khớp train_csr.indices
    train.train_df = train.train_df.assign(item_idx=[2, 0, 0])
    with pytest.raises(ValueError, match="không khớp thứ tự"):
        time_decay_weights(train, half_life_days=365.0)


def test_time_decay_guard_fires_if_sort_removed(monkeypatch):
    """Bỏ phép sắp xếp bên trong phải làm guard nổ, không được im lặng chấm sai ô."""
    import src.models.score_fusion_utils as fusion
    train = _train_with_ts([0, 365, 730])
    train.train_df = train.train_df.iloc[[2, 0, 1]].reset_index(drop=True)
    monkeypatch.setattr(fusion.np, "argsort", lambda a, **kw: np.arange(len(a)))
    with pytest.raises(ValueError, match="không khớp thứ tự"):
        fusion.time_decay_weights(train, half_life_days=365.0)


def test_time_decay_wired_into_user_weights():
    from src.models.user_history_weights import build_user_weights
    train = _train_with_ts([0, 365])
    W = build_user_weights(train, "binary_pos", half_life_days=365.0)
    assert np.allclose(np.sort(W.toarray()[0])[-2:], [0.5, 1.0])
