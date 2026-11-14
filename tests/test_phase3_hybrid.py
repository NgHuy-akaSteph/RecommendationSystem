"""Test hybrid late fusion phase 3 (CB-4)."""

import numpy as np
import pytest

from src.eval.model_registry import build_model
from src.models.base_recommender import BaseRecommender
from src.models.hybrid_content_model import HybridLateFusionModel
from src.models.score_fusion_utils import minmax_row
from tests.conftest import _mock_train


# ── hybrid ───────────────────────────────────────────────────────────────

def _genome_feats(n_items=10, seed=3):
    return {"genome": np.random.default_rng(seed).random((n_items, 8)).astype("float32")}


class _ConstantModel(BaseRecommender):
    """Mô hình giả trả về điểm cố định, dùng để kiểm tra đúng phép hợp nhất."""

    name = "test-constant"
    ROWS: dict[str, np.ndarray] = {}

    def __init__(self, key: str):
        self.key = key

    def fit(self, train, item_features):
        return self

    def score_users(self, user_idx):
        return np.tile(self.ROWS[self.key], (len(user_idx), 1)).astype("float32")


def test_hybrid_single_component_without_prior_keeps_ranking():
    """1 component, λ=0 → chỉ là minmax của mô hình gốc, thứ hạng phải y hệt."""
    train = _mock_train(n_users=5, n_items=10)
    feats = _genome_feats()
    comp = {"model": "cb3-genome", "params": {"standardize": "zscore"}}
    users = np.array([0, 2], dtype="int32")

    base = build_model("cb3-genome", {"standardize": "zscore"}).fit(train, feats)
    hybrid = HybridLateFusionModel([comp]).fit(train, feats)
    for row_b, row_h in zip(base.score_users(users), hybrid.score_users(users)):
        assert np.array_equal(np.argsort(-row_b), np.argsort(-row_h))


def test_hybrid_fuses_on_normalised_scale_not_raw(monkeypatch):
    """Bỏ minmax_row trước khi cộng phải làm test đỏ.

    Hai component cố ý lệch thang: A trong [0,1], B lớn gấp 100 lần. Cộng thô thì B nuốt
    trọn A và thứ hạng bằng đúng thứ hạng của B; cộng sau chuẩn hoá thì A phải có tiếng nói.
    """
    from src.eval import model_registry
    _ConstantModel.ROWS = {
        "a": np.array([0.9, 0.8, 0.1, 0.0]),
        "b": np.array([10.0, 50.0, 90.0, 95.0]),
    }
    monkeypatch.setitem(model_registry.REGISTRY, "test-constant", _ConstantModel)

    train = _mock_train(n_users=3, n_items=4, n_per_user=1)
    comps = [{"model": "test-constant", "params": {"key": "a"}},
             {"model": "test-constant", "params": {"key": "b"}}]
    users = np.array([0], dtype="int32")
    got = HybridLateFusionModel(comps, weights=[0.5, 0.5]).fit(train, {}).score_users(users)

    mask = ~(train.train_csr[users].toarray() > 0)
    a = minmax_row(_ConstantModel.ROWS["a"][None, :].astype("float32"), mask)
    b = minmax_row(_ConstantModel.ROWS["b"][None, :].astype("float32"), mask)
    assert np.allclose(got, 0.5 * a + 0.5 * b, atol=1e-5)

    raw_sum = 0.5 * _ConstantModel.ROWS["a"] + 0.5 * _ConstantModel.ROWS["b"]
    assert not np.array_equal(np.argsort(-got[0]), np.argsort(-raw_sum))


def test_hybrid_pop_prior_shifts_towards_popular_items():
    train = _mock_train(n_users=20, n_items=10)
    feats = _genome_feats()
    comp = {"model": "cb3-genome", "params": {"standardize": "zscore"}}
    users = np.array([0, 1], dtype="int32")

    plain = HybridLateFusionModel([comp]).fit(train, feats).score_users(users)
    biased = HybridLateFusionModel([comp], pop_prior=50.0).fit(train, feats).score_users(users)
    assert not np.array_equal(np.argsort(-plain[0]), np.argsort(-biased[0]))
    # λ rất lớn → prior chi phối. Không đòi thứ hạng khớp tuyệt đối (nhiều item cùng
    # popularity, thế hoà do điểm nội dung phá) mà đòi bất biến mạnh hơn: không item nào
    # popularity THẤP HƠN được xếp trên item popularity CAO HƠN.
    pop_in_rank_order = train.item_popularity[np.argsort(-biased[0])]
    assert np.all(np.diff(pop_in_rank_order) <= 0), pop_in_rank_order


def test_hybrid_validates_construction_and_usage():
    comp = {"model": "cb3-genome", "params": {}}
    assert HybridLateFusionModel([comp, comp], weights=[3.0, 1.0]).weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="ít nhất 1 component"):
        HybridLateFusionModel([])
    with pytest.raises(ValueError, match="trọng số"):
        HybridLateFusionModel([comp, comp], weights=[1.0])
    with pytest.raises(ValueError, match="dương"):
        HybridLateFusionModel([comp], weights=[0.0])
    with pytest.raises(RuntimeError, match="fit"):
        HybridLateFusionModel([comp]).score_users(np.array([0]))
    with pytest.raises(ValueError, match="lồng hybrid"):
        HybridLateFusionModel([{"model": "cb4-hybrid", "params": {}}]).fit(
            _mock_train(3, 5), {"genome": np.zeros((5, 4), dtype="float32")})
