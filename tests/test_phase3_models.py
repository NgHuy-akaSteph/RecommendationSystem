"""Test phase 3: BM25 và baseline CF. Hợp nhất điểm/hybrid ở test_phase3_hybrid_fusion."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import paths
from src.models.als_cf_model import build_confidence_matrix
from src.models.bm25_content_model import Bm25ContentModel, bm25_weight
from src.models.itemknn_cf_model import build_cf_item_matrix
from tests.conftest import _mock_train


# ── BM25 ─────────────────────────────────────────────────────────────────

def test_bm25_matches_formula():
    counts = sparse.csr_matrix(np.array([[3, 0, 1], [1, 2, 0], [0, 0, 5]], dtype="int32"))
    k1, b = 1.2, 0.75
    got = bm25_weight(counts, k1, b).toarray()

    n_docs = 3
    df = np.array([2, 1, 2], dtype="float64")
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
    doc_len = np.array([4, 3, 5], dtype="float64")
    avg = doc_len.mean()
    dense = counts.toarray()
    for i in range(3):
        for t in range(3):
            c = dense[i, t]
            expected = 0.0 if c == 0 else (
                idf[t] * c * (k1 + 1) / (c + k1 * (1 - b + b * doc_len[i] / avg)))
            assert got[i, t] == pytest.approx(expected, rel=1e-5)


def test_bm25_b_zero_removes_length_normalisation():
    """b=0 → mẫu số không phụ thuộc độ dài tài liệu, hai doc cùng count ra cùng trọng số."""
    counts = sparse.csr_matrix(np.array([[2, 0, 0, 0], [2, 1, 1, 1]], dtype="int32"))
    w = bm25_weight(counts, k1=1.2, b=0.0).toarray()
    assert w[0, 0] == pytest.approx(w[1, 0], rel=1e-6)
    w_b1 = bm25_weight(counts, k1=1.2, b=1.0).toarray()
    assert w_b1[0, 0] > w_b1[1, 0]  # doc dài bị phạt


def test_bm25_small_k1_approaches_pure_idf():
    """k1 → 0 là bão hoà tuyệt đối: trọng số chỉ còn idf, không phụ thuộc số lần lặp."""
    counts = sparse.csr_matrix(np.array([[1, 0], [9, 1]], dtype="int32"))
    w = bm25_weight(counts, k1=1e-6, b=0.0).toarray()
    assert w[0, 0] == pytest.approx(w[1, 0], rel=1e-3)


def test_bm25_model_end_to_end():
    train = _mock_train(n_users=4, n_items=6)
    genres = sparse.lil_matrix((6, 19), dtype="float32")
    genres[0, 0] = genres[1, 1] = 1.0
    feats = {"genres": genres.tocsr(),
             "tag_docs": pd.DataFrame({"item_idx": np.arange(6),
                                       # tag phải >= 2 ký tự: token_pattern mặc định
                                       # của CountVectorizer là \w\w+
                                       "tag_text": ["noir crime", "crime heist",
                                                    "noir heist"] + [""] * 3})}
    model = Bm25ContentModel(min_df=1, max_df=1.0).fit(train, feats)
    scores = model.score_users(np.array([0, 1]))
    assert scores.shape == (2, 6) and np.isfinite(scores).all()


# ── baseline CF ──────────────────────────────────────────────────────────

def test_cf_item_matrix_is_binary_relevant_and_l2_normed():
    train = _mock_train(n_users=8, n_items=12)
    X = build_cf_item_matrix(train)
    assert X.shape == (12, 8)                                   # [items, users]
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1))).ravel()
    assert np.all((np.abs(norms - 1.0) < 1e-5) | (norms == 0))  # hàng rỗng vẫn được phép
    # Chỉ tương tác relevant được giữ
    relevant = (train.train_csr.data >= paths.RELEVANCE_THRESHOLD).sum()
    assert X.nnz == relevant


def test_als_confidence_drops_non_relevant():
    """c = 1 + alpha chỉ cho tương tác relevant.

    Nếu lưu cả rating thấp với c = 1 thì (c−1)=0 nhưng b += 1·factor vẫn cộng, tức ALS
    học 'user thích phim này' cho đúng những phim user chấm điểm thấp.
    """
    train = _mock_train(n_users=6, n_items=10)
    C = build_confidence_matrix(train, alpha=40.0)
    assert np.allclose(C.data, 41.0)
    assert C.nnz == int((train.train_csr.data >= paths.RELEVANCE_THRESHOLD).sum())
    assert C.shape == train.train_csr.shape
