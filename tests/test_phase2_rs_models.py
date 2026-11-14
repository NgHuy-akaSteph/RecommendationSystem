"""Test Phase 02: hợp đồng catalog chung của Item-CF và BPR port từ RS.

Trọng tâm: hai mô hình mới phải chấp đúng hợp đồng `BaseRecommender`
(`fit(train, item_features)` → `score_users(user_idx) -> [B, n_items]`) và phải dùng
**catalog chung** (13.816 phim) chứ không tự dựng `movie_map` riêng như bản gốc.
"""

import numpy as np
import pytest

from src import paths
from src.eval.model_registry import REGISTRY, build_model
from src.models.itemknn_cf_model import ItemKnnCfModel
from src.models.rs_bpr_model import RsBprModel, build_binary_relevant_matrix
from src.models.rs_itemcf_model import RsItemCfModel
from src.models.seen_item_masking import SEEN_ITEM_SCORE, mask_seen_items
from tests.conftest import _mock_train


# ── đăng ký ─────────────────────────────────────────────────────────────

def test_both_models_registered():
    assert "rs-itemcf" in REGISTRY
    assert "rs-bpr" in REGISTRY
    assert build_model("rs-itemcf").name == "rs-itemcf"
    assert build_model("rs-bpr").name == "rs-bpr"


def test_unknown_model_raises_with_hint():
    with pytest.raises(KeyError, match="rs-itemcf"):
        build_model("khong-ton-tai")


# ── hợp đồng catalog ───────────────────────────────────────────────────

@pytest.mark.parametrize("model_name,params", [
    ("rs-itemcf", {"k_neighbors": 5}),
    ("rs-bpr", {"factors": 8, "iterations": 2}),
])
def test_score_users_shape_is_batch_by_n_items(model_name, params):
    train = _mock_train(n_users=8, n_items=25)
    model = build_model(model_name, params).fit(train, {})
    users = np.array([0, 3, 7], dtype="int32")

    scores = model.score_users(users)
    assert scores.shape == (3, train.n_items)
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all(), "điểm phải hữu hạn (không NaN/Inf) để cộng/minmax được"


@pytest.mark.parametrize("model_name,params", [
    ("rs-itemcf", {"k_neighbors": 5}),
    ("rs-bpr", {"factors": 8, "iterations": 2}),
])
def test_score_users_never_ranks_already_seen_items(model_name, params):
    """Mô hình phải tự loại item đã xem — evaluator cũng mask, nhưng mô hình không
    được phụ thuộc vào điều đó (nó còn được dùng làm thành phần của lớp fusion khác)."""
    train = _mock_train(n_users=8, n_items=25)
    model = build_model(model_name, params).fit(train, {})
    users = np.array([0, 3, 7], dtype="int32")

    scores = model.score_users(users)
    seen = train.train_csr[users].toarray() > 0
    assert seen.any(), "fixture phải có user đã xem gì, nếu không test vô nghĩa"
    assert np.all(scores[seen] <= SEEN_ITEM_SCORE)


def test_score_users_with_empty_batch_is_no_error():
    train = _mock_train(n_users=5, n_items=12)
    for model in (RsItemCfModel(k_neighbors=3), RsBprModel(factors=4, iterations=2)):
        scores = model.fit(train, {}).score_users(np.array([], dtype="int32"))
        assert scores.shape == (0, train.n_items)


def test_model_rejects_index_out_of_catalog():
    """Score phải đúng số item của catalog chung, không phải số item trong df dương."""
    train = _mock_train(n_users=6, n_items=30)
    model = RsItemCfModel(k_neighbors=4).fit(train, {})
    scores = model.score_users(np.array([0, 1], dtype="int32"))
    assert scores.shape[1] == train.n_items == 30


def test_score_users_before_fit_raises():
    with pytest.raises(RuntimeError):
        RsItemCfModel().score_users(np.array([0], dtype="int32"))
    with pytest.raises(RuntimeError):
        RsBprModel().score_users(np.array([0], dtype="int32"))


# ── Item-CF ─────────────────────────────────────────────────────────────

def test_rs_itemcf_uses_shared_cf_item_matrix_not_own_movie_map():
    """Ma trận item phải là nhị phân relevant L2-norm — cùng nguồn với cf-itemknn."""
    from src.models.itemknn_cf_model import build_cf_item_matrix

    train = _mock_train(n_users=8, n_items=20)
    X = build_cf_item_matrix(train)
    assert X.shape == (20, 8)
    relevant = int((train.train_csr.data >= paths.RELEVANCE_THRESHOLD).sum())
    assert X.nnz == relevant


def test_rs_itemcf_differs_from_cb_itemknn_because_of_user_weights():
    """Hai mô hình cùng thuật toán, khác trọng số phía user → phải cho kết quả khác."""
    train = _mock_train(n_users=10, n_items=40)
    X = {"X": None}

    from src.models.itemknn_cf_model import build_cf_item_matrix
    features = {"X": build_cf_item_matrix(train)}

    rs = RsItemCfModel(k_neighbors=10, weight_mode="rating").fit(train, features)
    cb = ItemKnnCfModel(k_neighbors=10, weight_mode="binary_pos").fit(train, features)
    users = np.array([0, 1, 2], dtype="int32")

    assert not np.allclose(rs.score_users(users), cb.score_users(users))


def test_rs_itemcf_binary_weight_mode_matches_cb_itemknn():
    """Cùng thuật toán + cùng trọng số + cùng hướng W @ Kᵀ → phải cho cùng điểm.

    Chỉ so trên **item ứng viên**: `rs-itemcf` tự mask item đã xem (−1e9) còn
    `cf-itemknn` không mask, vì evaluator lo làm việc đó.
    """
    train = _mock_train(n_users=10, n_items=40)
    from src.models.itemknn_cf_model import build_cf_item_matrix
    features = {"X": build_cf_item_matrix(train)}

    rs = RsItemCfModel(k_neighbors=10, weight_mode="binary_pos").fit(train, features)
    cb = ItemKnnCfModel(k_neighbors=10, weight_mode="binary_pos").fit(train, features)
    users = np.array([0, 1, 2], dtype="int32")

    seen = train.train_csr[users].toarray() > 0
    assert np.allclose(rs.score_users(users)[~seen], cb.score_users(users)[~seen], atol=1e-5)


def test_rs_itemcf_uses_candidate_orientation_w_kt():
    """S = W @ Kᵀ, không phải W @ K (K bất đối xứng vì top-k cắt theo hàng)."""
    from src.models.content_item_knn_scorer import _build_topk_sim_blockwise
    from src.models.itemknn_cf_model import build_cf_item_matrix
    from src.models.user_history_weights import build_user_weights

    train = _mock_train(n_users=8, n_items=30)
    X = build_cf_item_matrix(train)
    model = RsItemCfModel(k_neighbors=6, weight_mode="binary_pos").fit(train, {"X": X})
    K = _build_topk_sim_blockwise(X, k=6)
    W = build_user_weights(train, "binary_pos")

    users = np.array([0, 2, 5], dtype="int32")
    expected = (W[users] @ K.T).toarray()
    got = model.score_users(users)

    # Item đã xem bị mask ở cả hai phía → so sánh trên item ứng viên
    seen = train.train_csr[users].toarray() > 0
    assert np.allclose(got[~seen], expected[~seen], atol=1e-4)


def test_rs_itemcf_rating_weight_mode_uses_raw_ratings():
    """weight_mode='rating' phải đưa rating thật vào phía user."""
    from src.models.user_history_weights import build_user_weights

    train = _mock_train(n_users=6, n_items=20)
    W = build_user_weights(train, "rating")
    assert np.allclose(W.data, train.train_csr.data)
    assert W.nnz == train.train_csr.nnz


# ── BPR ──────────────────────────────────────────────────────────────────

def test_bpr_binary_matrix_keeps_only_relevant():
    """BPR là phản hồi ẩn: giữ cả rating thấp sẽ dạy mô hình 'user thích phim 1 sao'."""
    train = _mock_train(n_users=6, n_items=20)
    B = build_binary_relevant_matrix(train)
    assert np.all(B.data == 1.0)
    assert B.nnz == int((train.train_csr.data >= paths.RELEVANCE_THRESHOLD).sum())
    assert B.shape == train.train_csr.shape


def test_bpr_is_reproducible_with_same_seed():
    """Cùng seed + 1 luồng → phải cho cùng điểm.

    `num_threads > 1` thì **không** reproducible (RNG lấy mẫu âm dùng chung giữa các
    luồng) — xem test kế tiếp.
    """
    train = _mock_train(n_users=10, n_items=30)
    users = np.array([0, 1, 2], dtype="int32")

    a = RsBprModel(factors=8, iterations=3, random_state=42, num_threads=1).fit(train, {}).score_users(users)
    b = RsBprModel(factors=8, iterations=3, random_state=42, num_threads=1).fit(train, {}).score_users(users)
    assert np.allclose(a, b, atol=1e-5)


def test_bpr_default_is_single_threaded_for_reproducibility():
    """Mặc định phải là 1 luồng — số liệu trong báo cáo phải chạy lại được."""
    assert RsBprModel().num_threads == 1


def test_bpr_scores_differ_from_als_scores():
    """Hai mô hình MF khác thuật toán tối ưu → điểm phải khác nhau."""
    from src.models.als_cf_model import AlsCfModel

    train = _mock_train(n_users=10, n_items=30)
    users = np.array([0, 1, 2], dtype="int32")

    bpr = RsBprModel(factors=8, iterations=3).fit(train, {}).score_users(users)
    als = AlsCfModel(factors=8, iterations=2, num_threads=1).fit(train, {}).score_users(users)
    assert not np.allclose(bpr, als, atol=1e-3)


# ── helper mask ─────────────────────────────────────────────────────────

def test_mask_seen_items_is_finite_and_marks_only_seen():
    train = _mock_train(n_users=5, n_items=12)
    scores = np.zeros((2, 12), dtype="float32")
    users = np.array([0, 4], dtype="int32")

    out = mask_seen_items(scores, train.train_csr, users)
    assert out is scores                       # in-place
    assert np.isfinite(scores).all()            # sentinel hữu hạn
    seen = train.train_csr[users].toarray() > 0
    assert np.all(scores[seen] == SEEN_ITEM_SCORE)
    assert np.all(scores[~seen] == 0.0)
