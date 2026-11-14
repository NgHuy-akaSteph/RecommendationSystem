"""Kiểm tra công thức metric bằng ví dụ tính tay.

Metric sai lệch định nghĩa là rủi ro lớn nhất của đề tài: mọi bảng kết quả và mọi
kết luận đều dựa vào module này, nhưng lỗi lại không gây crash. Vì vậy các giá trị
kỳ vọng dưới đây được tính tay, không lấy từ output của code.
"""

import numpy as np
import pytest
from scipy import sparse

from src.eval import topk_ranking_metrics as m

LOG2 = np.log2


def build_rel_csr(relevant_per_user: list[list[int]], n_items: int) -> sparse.csr_matrix:
    rows = [u for u, items in enumerate(relevant_per_user) for _ in items]
    cols = [i for items in relevant_per_user for i in items]
    data = np.ones(len(rows), dtype="float32")
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(relevant_per_user), n_items))


def test_gather_relevance_danh_dau_dung_vi_tri():
    rel = build_rel_csr([[1, 3], [0], []], n_items=6)
    topk = np.array([[3, 0, 1], [2, 0, 5], [1, 2, 3]], dtype="int32")

    flags = m.gather_relevance(rel, topk)

    # user 0: item 3 relevant (hạng 1), item 0 không, item 1 relevant (hạng 3)
    assert flags[0].tolist() == [1.0, 0.0, 1.0]
    # user 1: chỉ item 0 relevant, nằm ở hạng 2
    assert flags[1].tolist() == [0.0, 1.0, 0.0]
    # user 2 không có item relevant nào
    assert flags[2].tolist() == [0.0, 0.0, 0.0]


def test_metrics_tinh_tay_mot_user():
    # Gợi ý 4 item, item ở hạng 1 và 3 là relevant, user có tổng cộng 3 item relevant
    rel_flags = np.array([[1.0, 0.0, 1.0, 0.0]], dtype="float32")
    n_relevant = np.array([3.0], dtype="float32")

    out = m.compute_metrics(rel_flags, n_relevant, k_list=(2, 4))

    # @2: đúng 1 hit trong 2 vị trí đầu
    assert out["P@2"][0] == pytest.approx(1 / 2)
    assert out["R@2"][0] == pytest.approx(1 / 3)
    assert out["HR@2"][0] == 1.0
    dcg2 = 1 / LOG2(2)
    idcg2 = 1 / LOG2(2) + 1 / LOG2(3)  # min(3 relevant, k=2) = 2 vị trí lý tưởng
    assert out["NDCG@2"][0] == pytest.approx(dcg2 / idcg2, rel=1e-5)

    # @4: 2 hit ở hạng 1 và 3
    assert out["P@4"][0] == pytest.approx(2 / 4)
    assert out["R@4"][0] == pytest.approx(2 / 3)
    dcg4 = 1 / LOG2(2) + 1 / LOG2(4)
    idcg4 = 1 / LOG2(2) + 1 / LOG2(3) + 1 / LOG2(4)  # min(3, 4) = 3
    assert out["NDCG@4"][0] == pytest.approx(dcg4 / idcg4, rel=1e-5)


def test_idcg_gioi_han_boi_k_khi_it_item_relevant():
    # User chỉ có 1 item relevant, nằm đúng hạng 1 -> xếp hạng hoàn hảo, NDCG = 1
    rel_flags = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    out = m.compute_metrics(rel_flags, np.array([1.0]), k_list=(4,))

    assert out["NDCG@4"][0] == pytest.approx(1.0)
    assert out["R@4"][0] == pytest.approx(1.0)
    assert out["P@4"][0] == pytest.approx(0.25)


def test_khong_hit_thi_moi_metric_bang_0():
    rel_flags = np.zeros((1, 5), dtype="float32")
    out = m.compute_metrics(rel_flags, np.array([2.0]), k_list=(5,))

    assert out["P@5"][0] == 0.0
    assert out["R@5"][0] == 0.0
    assert out["HR@5"][0] == 0.0
    assert out["NDCG@5"][0] == 0.0


def test_xep_hang_hoan_hao_cho_ndcg_bang_1():
    # 3 item relevant nằm đúng 3 hạng đầu
    rel_flags = np.array([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype="float32")
    out = m.compute_metrics(rel_flags, np.array([3.0]), k_list=(3, 5))

    assert out["NDCG@3"][0] == pytest.approx(1.0)
    assert out["NDCG@5"][0] == pytest.approx(1.0)
    assert out["R@3"][0] == pytest.approx(1.0)


def test_evaluate_ranking_nhieu_user():
    rel = build_rel_csr([[2], [0, 1]], n_items=4)
    topk = np.array([[2, 0, 1], [3, 1, 0]], dtype="int32")

    out = m.evaluate_ranking(topk, rel, k_list=(3,))

    # user 0: hit ở hạng 1 với 1 item relevant -> NDCG = 1
    # user 1: hit ở hạng 2 và 3 với 2 item relevant
    dcg = 1 / LOG2(3) + 1 / LOG2(4)
    idcg = 1 / LOG2(2) + 1 / LOG2(3)
    assert out["NDCG@3"].tolist() == pytest.approx([1.0, dcg / idcg], rel=1e-5)
    assert out["R@3"].tolist() == pytest.approx([1.0, 1.0])
    assert m.summarize(out)["HR@3"] == pytest.approx(1.0)


def test_k_vuot_qua_so_item_xep_hang_thi_bao_loi():
    with pytest.raises(ValueError):
        m.compute_metrics(np.zeros((1, 3), dtype="float32"), np.array([1.0]), k_list=(10,))


def test_user_khong_co_item_relevant_thi_bao_loi():
    with pytest.raises(ValueError):
        m.compute_metrics(np.zeros((1, 3), dtype="float32"), np.array([0.0]), k_list=(3,))
