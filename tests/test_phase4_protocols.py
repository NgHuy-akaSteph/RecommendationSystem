"""Test split cold-start item và giao thức leave-one-out lấy mẫu của phase 4."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import paths
from src.data.build_interaction_matrices import Dataset, TrainData
from src.eval import cold_start_item_split as cold
from src.eval.sampled_loo_evaluator import build_loo_candidates, evaluate_sampled
from src.models.base_recommender import BaseRecommender


# ── cold-start item split ────────────────────────────────────────────────

def _item_meta(n=100):
    return pd.DataFrame({"item_idx": np.arange(n),
                         "year": np.repeat([1980.0, 1990.0, 2000.0, 2010.0, 2020.0], n // 5)})


def test_heldout_is_stratified_across_year_buckets():
    """Lấy mẫu đều theo năm: nếu bốc ngẫu nhiên toàn cục, phim mới sẽ bị thiếu đại diện."""
    meta = _item_meta(100)
    held = cold.select_heldout_items(meta, ratio=0.1, seed=42)
    assert len(held) == 10
    per_bucket = [np.sum((held >= lo) & (held < lo + 20)) for lo in range(0, 100, 20)]
    assert per_bucket == [2, 2, 2, 2, 2]


def test_heldout_is_deterministic_and_sorted():
    meta = _item_meta(100)
    a = cold.select_heldout_items(meta, 0.1, seed=42)
    b = cold.select_heldout_items(meta, 0.1, seed=42)
    assert np.array_equal(a, b)
    assert np.array_equal(a, np.sort(a))
    assert not np.array_equal(a, cold.select_heldout_items(meta, 0.1, seed=7))


def test_heldout_ratio_is_validated():
    meta = _item_meta(100)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="holdout_ratio"):
            cold.select_heldout_items(meta, ratio=bad, seed=42)


def test_heldout_handles_items_with_unknown_year():
    meta = _item_meta(100)
    meta.loc[:9, "year"] = np.nan          # 10 phim thiếu năm → nhóm riêng
    held = cold.select_heldout_items(meta, 0.1, seed=42)
    assert len(held) == 10
    assert np.any(held < 10), "nhóm thiếu năm cũng phải có đại diện"


def test_build_split_removes_heldout_from_train_only(tmp_path, monkeypatch):
    train = pd.DataFrame({"user_idx": [0, 0, 1, 1], "item_idx": [0, 1, 1, 2],
                          "rating": [5.0, 4.0, 5.0, 3.0], "ts": [1, 2, 3, 4]})
    test = pd.DataFrame({"user_idx": [0, 1], "item_idx": [2, 0],
                         "rating": [5.0, 4.0], "ts": [5, 6]})
    meta = pd.DataFrame({"item_idx": [0, 1, 2], "year": [1990.0, 2000.0, 2010.0]})
    for name, df in [("train.parquet", train), ("test.parquet", test),
                     ("item_meta.parquet", meta)]:
        df.to_parquet(tmp_path / name, index=False)
    monkeypatch.setattr(paths, "TRAIN_PARQUET", tmp_path / "train.parquet")
    monkeypatch.setattr(paths, "TEST_PARQUET", tmp_path / "test.parquet")
    monkeypatch.setattr(paths, "ITEM_META_PARQUET", tmp_path / "item_meta.parquet")

    cold_train, cold_test, n_original = cold.build_split(np.array([2], dtype="int32"))
    assert 2 not in set(cold_train["item_idx"])        # train sạch item held-out
    assert set(cold_test["item_idx"]) == {2}           # test chỉ còn item held-out
    assert len(cold_train) == 3 and len(cold_test) == 1
    assert n_original == 4      # trả về để caller khỏi hardcode số rating train gốc


# ── sampled leave-one-out ────────────────────────────────────────────────

def _dataset(n_users=4, n_items=30):
    rng = np.random.default_rng(0)
    rows, cols, vals = [], [], []
    for u in range(n_users):
        for i in rng.choice(n_items // 2, size=4, replace=False):
            rows.append(u); cols.append(int(i)); vals.append(5.0)
    train = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype="float32")
    train.sort_indices()
    test_rows = list(range(n_users))
    test_cols = [n_items // 2 + u for u in range(n_users)]
    test_rel = sparse.csr_matrix((np.ones(n_users, dtype="float32"), (test_rows, test_cols)),
                                 shape=(n_users, n_items), dtype="float32")
    train_data = TrainData(train_csr=train, train_df=pd.DataFrame(),
                           n_users=n_users, n_items=n_items)
    return Dataset(train=train_data, test_rel_csr=test_rel,
                   eval_users=np.arange(n_users, dtype="int32"))


def _write_test_parquet(tmp_path, monkeypatch, n_users=4, n_items=30):
    """test.parquet khớp với test_rel_csr của _dataset, thêm một item relevant cũ hơn."""
    rows = []
    for u in range(n_users):
        rows.append({"user_idx": u, "item_idx": n_items // 2 + u, "rating": 5.0, "ts": 100 + u})
        rows.append({"user_idx": u, "item_idx": n_items // 2 + u, "rating": 4.0, "ts": 1})
    df = pd.DataFrame(rows)
    df.loc[1::2, "item_idx"] = n_items - 1     # item cũ hơn, KHÔNG được chọn làm dương
    df.to_parquet(tmp_path / "test.parquet", index=False)
    monkeypatch.setattr(paths, "TEST_PARQUET", tmp_path / "test.parquet")


def test_loo_picks_latest_relevant_item_as_positive(tmp_path, monkeypatch):
    ds = _dataset()
    _write_test_parquet(tmp_path, monkeypatch)
    users, cand = build_loo_candidates(ds, n_negatives=10, seed=42)
    assert len(users) == 4
    assert np.array_equal(cand[:, 0], np.array([15, 16, 17, 18]))   # item ts mới nhất


def test_loo_negatives_exclude_everything_user_touched(tmp_path, monkeypatch):
    ds = _dataset()
    _write_test_parquet(tmp_path, monkeypatch)
    users, cand = build_loo_candidates(ds, n_negatives=10, seed=42)
    for i, u in enumerate(users):
        seen = set(ds.train.train_csr.getrow(u).indices) | set(ds.test_rel_csr.getrow(u).indices)
        assert not (set(cand[i, 1:]) & seen), "negative không được là item user đã tương tác"
        assert len(set(cand[i, 1:])) == 10, "negative phải khác nhau"


def test_loo_candidates_are_deterministic(tmp_path, monkeypatch):
    ds = _dataset()
    _write_test_parquet(tmp_path, monkeypatch)
    a = build_loo_candidates(ds, n_negatives=10, seed=42)[1]
    b = build_loo_candidates(ds, n_negatives=10, seed=42)[1]
    assert np.array_equal(a, b)


class _RankedModel(BaseRecommender):
    """Đặt item dương vào đúng một hạng định trước, không để hoà, để kiểm tra công thức."""

    name = "test-ranked"

    def __init__(self, n_items, positive_rank):
        self.n_items, self.positive_rank = n_items, positive_rank

    def fit(self, train, item_features):
        return self

    def score_users(self, user_idx):
        # Cột 0 của candidates là item dương; cho đúng `positive_rank` item vượt lên trên
        s = np.full((len(user_idx), self.n_items), -1.0, dtype="float32")
        s[:, 0] = 0.0
        s[:, 1: self.positive_rank + 1] = 1.0
        return s


@pytest.mark.parametrize("rank,expect_hit", [(0, True), (5, True), (9, True), (10, False)])
def test_sampled_metrics_match_rank_formula(rank, expect_hit):
    ds = _dataset()
    users = np.arange(4, dtype="int32")
    cand = np.tile(np.arange(11, dtype="int32"), (4, 1))   # 1 dương + 10 âm

    got = evaluate_sampled(_RankedModel(ds.n_items, rank), ds, users, cand)
    assert got["HR@10"] == pytest.approx(1.0 if expect_hit else 0.0)
    expected_ndcg = 1.0 / np.log2(rank + 2.0) if expect_hit else 0.0
    assert got["NDCG@10"] == pytest.approx(expected_ndcg, rel=1e-6)


def test_sampled_ties_are_broken_pessimistically():
    """Mô hình cho điểm hằng số không được ăn HR=1 nhờ hoà.

    Đây đúng là tình huống của những user không có tương tác relevant nào trong train:
    mô hình nội dung trả về vector 0 trên toàn bộ item. Phá hoà lạc quan sẽ cho họ
    HR@10 = 1,0 và thổi phồng mọi thuật toán content.
    """
    ds = _dataset()
    users = np.arange(4, dtype="int32")
    cand = np.tile(np.arange(11, dtype="int32"), (4, 1))

    class _Constant(BaseRecommender):
        name = "const"
        def fit(self, train, item_features): return self
        def score_users(self, user_idx):
            return np.zeros((len(user_idx), ds.n_items), dtype="float32")

    result = evaluate_sampled(_Constant(), ds, users, cand)
    assert result["HR@10"] == pytest.approx(0.0)
    assert result["NDCG@10"] == pytest.approx(0.0)
