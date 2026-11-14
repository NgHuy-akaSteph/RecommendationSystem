"""Test hai mô hình nội dung Phase 2: CB-1 TF-IDF và CB-3 Tag Genome (+ ridge)."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.models.genome_content_model import GenomeContentModel, build_genome_item_matrix
from src.models.genome_ridge_user_model import GenomeRidgeUserModel
from src.models.tfidf_content_model import TfidfContentModel
from src.models.vector_normalization import safe_l2_normalize
from tests.conftest import _mock_train


def _mock_content(n_items=6):
    genres = sparse.lil_matrix((n_items, 19), dtype="float32")
    genres[0, 0] = genres[1, 1] = genres[2, 0] = 1.0        # item 3..5 không có genre
    texts = ["action hero", "comedy fun", "action comedy"] + [""] * (n_items - 3)
    tag_docs = pd.DataFrame({"item_idx": np.arange(n_items), "tag_text": texts})
    return {"genres": genres.tocsr(), "tag_docs": tag_docs}


# ── CB-1 TF-IDF ──────────────────────────────────────────────────────────

def test_tfidf_cold_items_produce_no_nan():
    """183 item không có tag (ở đây cả không có genre) không được sinh NaN."""
    train = _mock_train(n_users=4, n_items=6)
    model = TfidfContentModel(use_tags=True, min_df=1, max_df=1.0, scorer="profile")
    model.fit(train, _mock_content())

    X = model._build_X(_mock_content())
    assert sparse.issparse(X), "X phải giữ sparse để không tốn 8.28GB ở cấu hình bigram"
    assert np.isfinite(X.data).all()
    assert np.allclose(np.asarray(X[3:].todense()), 0.0)  # hàng rỗng giữ 0, không NaN

    scores = model.score_users(np.array([0, 1]))
    assert scores.shape == (2, 6)
    assert np.isfinite(scores).all()


def test_tfidf_rejects_misaligned_tag_docs():
    feats = _mock_content()
    feats["tag_docs"]["item_idx"] = np.arange(1, len(feats["tag_docs"]) + 1)
    with pytest.raises(ValueError, match="thẳng hàng"):
        TfidfContentModel(use_tags=True, min_df=1).fit(_mock_train(4, 6), feats)


def test_tfidf_requires_features():
    with pytest.raises(KeyError, match="build_item_content_features"):
        TfidfContentModel().fit(_mock_train(4, 6), {})


def test_tfidf_rejects_bad_scorer():
    with pytest.raises(ValueError, match="item_knn"):
        TfidfContentModel(scorer="bogus").fit(_mock_train(4, 6), _mock_content())


@pytest.mark.parametrize("override", [
    {"use_tags": False}, {"min_df": 5}, {"max_df": 0.9}, {"max_features": 5_000},
    {"sublinear_tf": False}, {"ngram_range": (1, 2)}, {"genre_weight": 0.4},
])
def test_tfidf_cache_key_changes_for_every_param(override):
    """Bỏ sót bất kỳ tham số nào khỏi khoá cache là mở lại lỗi nạp nhầm ma trận sim."""
    base = TfidfContentModel(use_tags=True)._sim_cache_name()
    assert base != TfidfContentModel(**{"use_tags": True, **override})._sim_cache_name()


# ── CB-3 Genome ──────────────────────────────────────────────────────────

def test_genome_threshold_applies_to_raw_relevance():
    """Threshold phải cắt relevance thô, không phải giá trị sau z-score."""
    genome = np.array([[0.05, 0.9, 0.5], [0.2, 0.05, 0.7], [0.9, 0.3, 0.05]], dtype="float32")
    feats = {"genome": genome}

    expected = genome.copy()
    expected[expected < 0.1] = 0.0
    assert np.allclose(build_genome_item_matrix(feats, threshold=0.1), safe_l2_normalize(expected))

    # Nếu cắt sau z-score thì kết quả khác hẳn — chốt lại để không quay về thứ tự cũ
    z = (genome - genome.mean(0)) / np.maximum(genome.std(0), 1e-12)
    z[z < 0.1] = 0.0
    assert not np.allclose(build_genome_item_matrix(feats, "zscore", 0.1), safe_l2_normalize(z))


def test_genome_does_not_mutate_input():
    """Threshold ghi in-place; nếu quên copy sẽ ghi đè thẳng file item_genome.npy."""
    genome = np.array([[0.05, 0.9], [0.2, 0.6]], dtype="float32")
    build_genome_item_matrix({"genome": genome}, threshold=0.1)
    assert np.allclose(genome, [[0.05, 0.9], [0.2, 0.6]])


def test_genome_standardize_modes():
    feats = {"genome": np.random.default_rng(6).random((10, 8)).astype("float32")}
    for mode in (None, "zscore", "minmax"):
        X = build_genome_item_matrix(feats, standardize=mode)
        assert X.shape == (10, 8) and np.isfinite(X).all()
    with pytest.raises(ValueError):
        build_genome_item_matrix(feats, standardize="bogus")


def test_genome_constant_column_does_not_nan():
    """Cột hằng (std = 0, hoặc bị threshold xoá sạch) phải ra 0 chứ không NaN."""
    genome = np.array([[0.5, 0.05], [0.5, 0.02], [0.5, 0.01]], dtype="float32")
    for kwargs in ({"standardize": "zscore"}, {"standardize": "minmax"},
                   {"standardize": "zscore", "threshold": 0.1}):
        assert np.isfinite(build_genome_item_matrix({"genome": genome}, **kwargs)).all()


@pytest.mark.parametrize("override", [
    {"standardize": "zscore"}, {"standardize": "minmax"}, {"threshold": 0.1}, {"svd_dim": 64},
])
def test_genome_cache_key_changes_for_every_param(override):
    assert GenomeContentModel()._sim_cache_name() != GenomeContentModel(**override)._sim_cache_name()


def test_svd_cache_not_shared_across_preprocessing():
    """Cache SVD gắn với X đầu vào: đổi threshold phải sinh ma trận khác, không nạp lại."""
    g = np.random.default_rng(0).random((30, 12)).astype("float32")
    a = build_genome_item_matrix({"genome": g}, "zscore", None, 4)
    b = build_genome_item_matrix({"genome": g}, "zscore", 0.3, 4)
    assert not np.allclose(a, b)


def test_genome_content_model_svd():
    train = _mock_train(n_users=4, n_items=10)
    feats = {"genome": np.random.default_rng(7).random((10, 20)).astype("float32")}
    model = GenomeContentModel(standardize="zscore", svd_dim=4, scorer="profile").fit(train, feats)
    scores = model.score_users(np.array([0, 2]))
    assert scores.shape == (2, 10) and np.isfinite(scores).all()


def test_genome_model_rejects_bad_scorer():
    with pytest.raises(ValueError, match="item_knn"):
        GenomeContentModel(scorer="bogus").fit(
            _mock_train(4, 10), {"genome": np.random.default_rng(8).random((10, 6)).astype("float32")})


# ── CB-3 Ridge ───────────────────────────────────────────────────────────

def test_ridge_batched_solve_matches_per_user_reference():
    """Solve theo batch [B,d,d] phải cho đúng kết quả như giải từng user một."""
    train = _mock_train(n_users=6, n_items=12, n_per_user=6)
    feats = {"genome": np.random.default_rng(9).random((12, 16)).astype("float32")}
    lam, d = 2.0, 4
    model = GenomeRidgeUserModel(svd_dim=d, standardize="zscore", ridge_lambda=lam,
                                 min_train=2).fit(train, feats)

    users = np.array([0, 1, 2, 3], dtype="int32")
    got = model.score_users(users)

    Z, csr = model._Z, train.train_csr
    for row, u in enumerate(users):
        items, ratings = csr.getrow(u).indices, csr.getrow(u).data
        Z_u, mean = Z[items], model._user_means[u]
        w = np.linalg.solve(Z_u.T @ Z_u + lam * np.eye(d, dtype="float32"),
                            Z_u.T @ (ratings - mean))
        assert np.allclose(got[row], Z @ w + mean, atol=1e-4)


def test_ridge_falls_back_for_sparse_users():
    train = _mock_train(n_users=6, n_items=12, n_per_user=3)
    feats = {"genome": np.random.default_rng(10).random((12, 16)).astype("float32")}
    model = GenomeRidgeUserModel(svd_dim=4, standardize="zscore", min_train=99).fit(train, feats)
    users = np.array([0, 1, 3], dtype="int32")
    assert np.allclose(model.score_users(users), model._fallback.score_users(users))
