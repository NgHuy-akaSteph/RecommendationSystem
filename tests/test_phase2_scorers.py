"""Test tầng dùng chung của Phase 2: chuẩn hoá vector, trọng số user, hai scorer."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.data.build_interaction_matrices import TrainData
from src.models.content_item_knn_scorer import ContentItemKnnScorer, _build_topk_sim_blockwise
from src.models.content_profile_scorer import ContentProfileScorer
from src.models.user_history_weights import build_user_weights
from src.models.vector_normalization import safe_l2_normalize
from tests.conftest import _mock_train, _unit_rows


# ── chuẩn hoá ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("to_sparse", [False, True])
def test_safe_l2_normalize_handles_zero_rows(to_sparse):
    """Hàng toàn 0 (183 item không có tag) phải giữ 0, tuyệt đối không ra NaN."""
    X = np.array([[3.0, 4.0], [0.0, 0.0], [-1.0, 0.0]], dtype="float32")
    out = safe_l2_normalize(sparse.csr_matrix(X) if to_sparse else X)
    dense = out.toarray() if sparse.issparse(out) else out

    assert np.isfinite(dense).all()
    assert np.allclose(dense[0], [0.6, 0.8])
    assert np.allclose(dense[1], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(dense[[0, 2]], axis=1), 1.0)
    assert dense.dtype == np.float32


# ── trọng số user ────────────────────────────────────────────────────────

def test_build_user_weights():
    train = _mock_train(n_users=5, n_items=10)
    assert np.all(build_user_weights(train, "binary_pos").data == 1.0)
    assert np.allclose(build_user_weights(train, "rating").data, train.train_csr.data)
    with pytest.raises(ValueError):
        build_user_weights(train, "invalid_mode")


def test_centered_weights_subtract_correct_user_mean():
    train = _mock_train(n_users=5, n_items=10)
    W = build_user_weights(train, "centered").toarray()
    csr = train.train_csr
    for u in range(5):
        items = csr.getrow(u).indices
        expected = csr.getrow(u).data - csr.getrow(u).data.mean()
        assert np.allclose(W[u, items], expected, atol=1e-5)


def test_centered_weights_survive_explicit_stored_zero():
    """csr.nonzero() bỏ qua số 0 lưu tường minh → từng làm lệch chiều mảng rows."""
    csr = sparse.csr_matrix(np.array([[4.0, 1.0], [2.0, 3.0]], dtype="float32"))
    csr.data[1] = 0.0  # entry (0,1) vẫn được lưu nhưng giá trị 0 → nonzero() bỏ qua
    train = TrainData(train_csr=csr, train_df=pd.DataFrame(), n_users=2, n_items=2)
    W = build_user_weights(train, "centered")
    assert W.shape == (2, 2)
    assert np.isfinite(W.data).all()


# ── profile scorer ───────────────────────────────────────────────────────

def test_content_profile_scorer_sparse_matches_dense():
    """X sparse và X dense phải cho cùng điểm số — đường sparse là đường tối ưu RAM."""
    train = _mock_train(n_users=5, n_items=8)
    X_dense = _unit_rows(8, 4)
    users = np.array([0, 2, 4], dtype="int32")

    dense_scores = ContentProfileScorer().fit(train, {"X": X_dense}).score_users(users)
    sparse_scores = ContentProfileScorer().fit(train, {"X": sparse.csr_matrix(X_dense)}).score_users(users)

    assert dense_scores.shape == (3, 8)
    assert np.isfinite(dense_scores).all()
    assert np.allclose(dense_scores, sparse_scores, atol=1e-5)


# ── item-KNN scorer ──────────────────────────────────────────────────────

def test_topk_sim_is_exactly_topk():
    """Tập hàng xóm lưu lại phải đúng bằng top-k thật, không được lệch phần tử nào."""
    n_items, k = 60, 7
    X = _unit_rows(n_items, 5, seed=1)
    K = _build_topk_sim_blockwise(X, k=k)

    full = X @ X.T
    np.fill_diagonal(full, -np.inf)
    for i in range(n_items):
        assert set(K.getrow(i).indices) == set(np.argsort(-full[i])[:k]), f"hàng {i} sai top-k"
    assert np.all(np.diff(K.indptr) == k)
    assert np.allclose(K.diagonal(), 0.0)


def test_topk_sim_sparse_equals_dense():
    X = _unit_rows(40, 6, seed=2)
    X[3:6] = 0.0  # item không có đặc trưng
    assert abs(_build_topk_sim_blockwise(X, k=5)
               - _build_topk_sim_blockwise(sparse.csr_matrix(X), k=5)).max() < 1e-5


@pytest.mark.parametrize("k", [14, 15, 20])
def test_topk_sim_when_k_reaches_n_items(k):
    """k ≥ n_items: self-sim (−2) không được lọt vào ma trận hàng xóm."""
    n_items = 15
    K = _build_topk_sim_blockwise(_unit_rows(n_items, 4, seed=11), k=k)
    assert np.allclose(K.diagonal(), 0.0)
    assert np.all(np.diff(K.indptr) <= n_items - 1)


def test_item_knn_uses_transpose_orientation():
    """S phải bằng W @ Kᵀ. W @ K là ma trận khác vì K bất đối xứng (top-k theo hàng)."""
    train = _mock_train(n_users=6, n_items=25)
    X = _unit_rows(25, 5, seed=3)
    users = np.array([0, 1, 5], dtype="int32")

    scorer = ContentItemKnnScorer(sim_name="orient", k_neighbors=4).fit(train, {"X": X})
    K = _build_topk_sim_blockwise(X, k=4)
    W = build_user_weights(train, "binary_pos")[users]

    assert np.allclose(scorer.score_users(users), (W @ K.T).toarray(), atol=1e-5)
    assert not np.allclose((W @ K).toarray(), (W @ K.T).toarray())  # hai phép thực sự khác nhau


def test_item_knn_normalize_sim():
    train = _mock_train(n_users=6, n_items=25)
    X = _unit_rows(25, 5, seed=4)
    users = np.array([0, 3], dtype="int32")

    raw = ContentItemKnnScorer(sim_name="n0", k_neighbors=4).fit(train, {"X": X}).score_users(users)
    norm = ContentItemKnnScorer(sim_name="n1", k_neighbors=4,
                                normalize_sim=True).fit(train, {"X": X}).score_users(users)
    assert norm.shape == raw.shape
    assert np.isfinite(norm).all()
    # Chia cho Σ|sim| đưa điểm về trung bình có trọng số của các sim → trong [−1, 1]
    # (cận này chỉ đúng với trọng số không âm, tức weight_mode="binary_pos")
    assert np.abs(norm).max() <= 1.0 + 1e-5
    assert not np.allclose(norm, raw)


def test_sim_cache_keyed_by_matrix_content(tmp_path):
    """Hai X khác nhau không được dùng chung file cache dù cùng sim_name và cùng shape."""
    train = _mock_train(n_users=4, n_items=20)
    X1, X2 = _unit_rows(20, 4, seed=5), _unit_rows(20, 4, seed=6)
    users = np.array([0, 1], dtype="int32")

    s1 = ContentItemKnnScorer(sim_name="same", k_neighbors=3).fit(train, {"X": X1}).score_users(users)
    s2 = ContentItemKnnScorer(sim_name="same", k_neighbors=3).fit(train, {"X": X2}).score_users(users)
    assert not np.allclose(s1, s2), "cache bị dùng nhầm giữa hai X khác nhau"
    assert len(list(tmp_path.glob("sim_same_*_k3.npz"))) == 2

    # Cùng X → tái dùng đúng cache, kết quả không đổi
    again = ContentItemKnnScorer(sim_name="same", k_neighbors=3).fit(train, {"X": X1}).score_users(users)
    assert np.allclose(s1, again)
