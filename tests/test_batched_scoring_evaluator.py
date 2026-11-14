"""Kiểm tra evaluator: xếp hạng, loại item đã xem, và an toàn bộ nhớ.

Evaluator là nơi duy nhất biết nhãn test, nên mọi lỗi ở đây đều làm sai toàn bộ
bảng kết quả mà không gây crash. Riêng phép sao chép điểm số trước khi ghi -inf
từng là lỗi thật: `np.asarray(x, dtype=x.dtype)` trả về chính mảng gốc.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.data.build_interaction_matrices import Dataset, TrainData, _to_csr
from src.eval.batched_scoring_evaluator import evaluate_model, rank_top_items
from src.models.base_recommender import BaseRecommender


class FixedScoreRecommender(BaseRecommender):
    """Mô hình giả: trả về đúng ma trận điểm cho trước, dùng lại cùng một bộ đệm.

    Việc dùng lại bộ đệm là có chủ đích: nếu evaluator ghi -inf trực tiếp lên mảng
    này thì lô sau sẽ nhận điểm đã bị hỏng.
    """

    name = "fixed"

    def __init__(self, scores: np.ndarray):
        self.buffer = np.asarray(scores, dtype="float32")

    def fit(self, train: TrainData, item_features: dict) -> "FixedScoreRecommender":
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return self.buffer[user_idx]


def build_dataset(train_pairs, test_pairs, n_users, n_items, eval_users) -> Dataset:
    train_df = pd.DataFrame(train_pairs, columns=["user_idx", "item_idx", "rating"])
    train_df["ts"] = np.arange(len(train_df))
    train_csr = _to_csr(train_df, train_df["rating"].to_numpy("float32"), (n_users, n_items))

    test_df = pd.DataFrame(test_pairs, columns=["user_idx", "item_idx", "rating"])
    rel = _to_csr(test_df, np.ones(len(test_df), dtype="float32"), (n_users, n_items))

    train = TrainData(train_csr=train_csr, train_df=train_df, n_users=n_users, n_items=n_items)
    return Dataset(train=train, test_rel_csr=rel, eval_users=np.array(eval_users, dtype="int32"))


def test_rank_top_items_sap_theo_diem_giam_dan():
    scores = np.array([[0.1, 0.9, 0.5, 0.3], [0.4, 0.2, 0.8, 0.7]], dtype="float32")

    topk = rank_top_items(scores, k=3)

    assert topk[0].tolist() == [1, 2, 3]
    assert topk[1].tolist() == [2, 3, 0]


def test_item_da_co_trong_train_khong_duoc_goi_y():
    # user 0 đã xem item 0 (điểm cao nhất) -> phải bị loại khỏi danh sách gợi ý
    scores = np.array([[9.0, 1.0, 2.0, 3.0]], dtype="float32")
    dataset = build_dataset(
        train_pairs=[(0, 0, 5.0)],
        test_pairs=[(0, 3, 5.0)],
        n_users=1,
        n_items=4,
        eval_users=[0],
    )

    _, topk = evaluate_model(FixedScoreRecommender(scores), dataset, k_list=(3,))

    assert 0 not in topk[0].tolist()
    assert topk[0][0] == 3  # item điểm cao nhất trong số chưa xem


def test_evaluator_khong_ghi_de_len_mang_cua_model():
    scores = np.array([[5.0, 4.0, 3.0], [5.0, 4.0, 3.0]], dtype="float32")
    model = FixedScoreRecommender(scores)
    original = model.buffer.copy()
    dataset = build_dataset(
        train_pairs=[(0, 0, 5.0), (1, 0, 5.0)],
        test_pairs=[(0, 1, 5.0), (1, 1, 5.0)],
        n_users=2,
        n_items=3,
        eval_users=[0, 1],
    )

    evaluate_model(model, dataset, k_list=(2,))

    assert np.array_equal(model.buffer, original), "evaluator đã ghi -inf lên mảng của mô hình"
    assert np.isfinite(model.buffer).all()


def test_ket_qua_khong_phu_thuoc_kich_thuoc_lo():
    rng = np.random.default_rng(0)
    n_users, n_items = 7, 12
    scores = rng.random((n_users, n_items)).astype("float32")
    dataset = build_dataset(
        train_pairs=[(u, u % n_items, 5.0) for u in range(n_users)],
        test_pairs=[(u, (u + 3) % n_items, 5.0) for u in range(n_users)],
        n_users=n_users,
        n_items=n_items,
        eval_users=list(range(n_users)),
    )

    one_batch, _ = evaluate_model(
        FixedScoreRecommender(scores), dataset, k_list=(5,), batch_size=n_users
    )
    many_batches, _ = evaluate_model(
        FixedScoreRecommender(scores), dataset, k_list=(5,), batch_size=2
    )

    assert np.allclose(one_batch["NDCG@5"], many_batches["NDCG@5"])
    assert np.allclose(one_batch["R@5"], many_batches["R@5"])


def test_bao_loi_khi_model_tra_ve_sai_kich_thuoc():
    dataset = build_dataset(
        train_pairs=[(0, 0, 5.0)],
        test_pairs=[(0, 1, 5.0)],
        n_users=1,
        n_items=4,
        eval_users=[0],
    )
    wrong_shape = FixedScoreRecommender(np.zeros((1, 2), dtype="float32"))

    with pytest.raises(ValueError, match="score_users"):
        evaluate_model(wrong_shape, dataset, k_list=(2,))


def test_to_csr_bao_loi_khi_co_cap_user_item_trung_lap():
    df = pd.DataFrame(
        [(0, 1, 5.0), (0, 1, 3.0)], columns=["user_idx", "item_idx", "rating"]
    )

    with pytest.raises(ValueError, match="trùng lặp"):
        _to_csr(df, df["rating"].to_numpy("float32"), (2, 4))


def test_xep_hang_hoan_hao_cho_metric_toi_da():
    # 2 item relevant nằm đúng 2 vị trí điểm cao nhất trong số chưa xem
    scores = np.array([[0.0, 9.0, 8.0, 1.0]], dtype="float32")
    dataset = build_dataset(
        train_pairs=[(0, 0, 5.0)],
        test_pairs=[(0, 1, 5.0), (0, 2, 5.0)],
        n_users=1,
        n_items=4,
        eval_users=[0],
    )

    per_user, _ = evaluate_model(FixedScoreRecommender(scores), dataset, k_list=(2,))

    assert per_user["NDCG@2"][0] == pytest.approx(1.0)
    assert per_user["R@2"][0] == pytest.approx(1.0)
    assert per_user["P@2"][0] == pytest.approx(1.0)


def test_train_data_khong_lo_nhan_test_cho_model():
    # Hàng rào chống leakage: TrainData không có đường nào tới test_rel_csr
    dataset = build_dataset(
        train_pairs=[(0, 0, 5.0)], test_pairs=[(0, 1, 5.0)], n_users=1, n_items=3, eval_users=[0]
    )

    assert not hasattr(dataset.train, "test_rel_csr")
    assert isinstance(dataset.train.train_csr, sparse.csr_matrix)


def test_item_popularity_chi_dem_rating_relevant():
    dataset = build_dataset(
        train_pairs=[(0, 0, 5.0), (1, 0, 2.0), (0, 1, 4.0)],
        test_pairs=[(0, 2, 5.0)],
        n_users=2,
        n_items=3,
        eval_users=[0],
    )

    popularity = dataset.train.item_popularity

    assert popularity.tolist() == [1.0, 1.0, 0.0]  # rating 2.0 không tính
    assert dataset.train.item_popularity is popularity  # có cache, không tính lại
