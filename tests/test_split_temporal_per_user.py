"""Kiểm tra chia train/test theo thời gian trong từng user.

Rủi ro chính được kiểm ở đây là leakage: mọi tương tác trong test phải xảy ra
sau (hoặc cùng lúc) tương tác cuối cùng của train trong cùng user.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.split_temporal_per_user import select_valid_users, split_ratings


def make_ratings(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating", "ts"])


def test_chia_dung_ty_le_va_lay_tuong_tac_moi_nhat_lam_test():
    # 1 user với 10 tương tác, timestamp tăng dần -> 2 cái cuối vào test
    rows = [(0, i, 5.0, 100 + i) for i in range(10)]
    train, test = split_ratings(make_ratings(rows), test_ratio=0.2)

    assert len(train) == 8 and len(test) == 2
    assert sorted(test["item_idx"]) == [8, 9]
    assert train["ts"].max() < test["ts"].min()


def test_moi_user_luon_co_it_nhat_mot_item_test():
    # user có 3 tương tác: floor(3*0.2) = 0, phải được nâng lên 1
    rows = [(0, i, 4.0, 100 + i) for i in range(3)]
    train, test = split_ratings(make_ratings(rows), test_ratio=0.2)

    assert len(test) == 1 and len(train) == 2
    assert test["item_idx"].iloc[0] == 2


def test_khong_leakage_thoi_gian_tren_nhieu_user():
    rows = [(u, i, 4.0, 1000 * u + i) for u in range(3) for i in range(10)]
    train, test = split_ratings(make_ratings(rows), test_ratio=0.3)

    for user in range(3):
        user_train = train[train["user_idx"] == user]
        user_test = test[test["user_idx"] == user]
        assert user_train["ts"].max() < user_test["ts"].min()
        assert len(user_test) == 3


def test_timestamp_trung_nhau_van_chia_on_dinh():
    # Mọi timestamp giống nhau: kết quả phải ổn định (sắp phụ theo item_idx)
    rows = [(0, i, 4.0, 500) for i in range(5)]
    ratings = make_ratings(rows)

    first_train, first_test = split_ratings(ratings, test_ratio=0.2)
    second_train, second_test = split_ratings(ratings.sample(frac=1.0, random_state=0), test_ratio=0.2)

    assert sorted(first_test["item_idx"]) == sorted(second_test["item_idx"])
    assert len(first_train) == len(second_train) == 4


def test_loai_user_thieu_train_hoac_khong_co_item_relevant():
    train = make_ratings(
        [(0, i, 5.0, i) for i in range(6)]  # user 0: đủ train
        + [(1, i, 5.0, i) for i in range(3)]  # user 1: train quá ngắn
        + [(2, i, 5.0, i) for i in range(6)]  # user 2: đủ train nhưng test toàn rating thấp
    )
    test = make_ratings([(0, 9, 5.0, 99), (1, 9, 5.0, 99), (2, 9, 2.0, 99)])

    valid = select_valid_users(train, test, min_train=5)

    assert valid.tolist() == [0]


def test_user_hop_le_tra_ve_mang_int32_da_sap_xep():
    train = make_ratings([(u, i, 5.0, i) for u in (7, 2, 5) for i in range(6)])
    test = make_ratings([(u, 9, 5.0, 99) for u in (7, 2, 5)])

    valid = select_valid_users(train, test, min_train=5)

    assert valid.dtype == np.int32
    assert valid.tolist() == [2, 5, 7]


@pytest.mark.parametrize("n_interactions,expected_test", [(5, 1), (10, 2), (20, 4), (23, 4)])
def test_so_luong_test_theo_do_dai_lich_su(n_interactions, expected_test):
    rows = [(0, i, 4.0, i) for i in range(n_interactions)]
    _, test = split_ratings(make_ratings(rows), test_ratio=0.2)

    assert len(test) == expected_test
