"""Kiểm chứng tính đúng của evaluator theo chuẩn Järvelin & Kekäläinen (2002).

Bốn lỗi B1–B4 của protocol cũ trong `src/evaluation/` đã bị **loại bỏ khỏi đường
chạy** (xoá hẳn ở phase 04), không phải "đã sửa". Module này giữ lại bằng chứng
rằng evaluator mới không tái phátm bất kỳ lỗi nào:

    B1  P@k phải là hits/k (Järvelin), không phải hits/len(top_k)
    B2  exception từ score_users phải nổi lên, không bị nuốt thành mảng 0
    B3  ghép cặp thống kê theo tập user phải khớp, không theo thứ tự dict
    B4  không có validate_dataset_split mặc định 0,7 — split kiểm bằng test riêng

Ngoài ra có hai hàng rào mà bản của CB chưa có:

    - item gợi ý nằm ngoài catalog phải raise, không âm thầm thành điểm 0
    - user không có tín hiệu train phải có metric = 0 **một cách có nguyên tắc**,
      và vẫn phải ở lại trong tập đánh giá
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.data.build_interaction_matrices import Dataset, TrainData, _to_csr
from src.eval import topk_ranking_metrics as metrics_mod
from src.eval.batched_scoring_evaluator import (
    evaluate_model, rank_top_items, rows_without_signal,
)
from src.models.base_recommender import BaseRecommender

# ── helper ────────────────────────────────────────────────────────────────

def build_dataset(train_pairs, test_pairs, n_users, n_items, eval_users) -> Dataset:
    train_df = pd.DataFrame(train_pairs, columns=["user_idx", "item_idx", "rating"])
    train_df["ts"] = np.arange(len(train_df))
    train_csr = _to_csr(train_df, train_df["rating"].to_numpy("float32"), (n_users, n_items))

    test_df = pd.DataFrame(test_pairs, columns=["user_idx", "item_idx", "rating"])
    rel = _to_csr(test_df, np.ones(len(test_df), dtype="float32"), (n_users, n_items))

    train = TrainData(train_csr=train_csr, train_df=train_df, n_users=n_users, n_items=n_items)
    return Dataset(train=train, test_rel_csr=rel, eval_users=np.array(eval_users, dtype="int32"))


class FixedScoreRecommender(BaseRecommender):
    """Trả về đúng ma trận điểm cho trước."""

    name = "fixed"

    def __init__(self, scores: np.ndarray):
        self.buffer = np.asarray(scores, dtype="float32")

    def fit(self, train, item_features):
        return self

    def score_users(self, user_idx):
        return self.buffer[user_idx]


class ExplodingRecommender(BaseRecommender):
    """Ném exception khi chấm điểm — mô phỏng mô hình hỏng."""

    name = "exploding"

    def fit(self, train, item_features):
        return self

    def score_users(self, user_idx):
        raise RuntimeError("mô hình hỏng giữa chừng")


# ── B1: P@k = hits/k, không phải hits/len(top_k) ─────────────────────────

def test_b1_precision_la_hits_tren_k_chu_khong_phai_len_danh_sach():
    """Model trả ít item hơn k thì P@k vẫn phải chia cho k.

    Đây là điểm mấu chốt: protocol cũ chia cho `len(top_k)` nên một mô hình chỉ trả 2
    gợi ý nhưng trúng 1 sẽ có P@10 = 0,5 thay vì 0,1 — biến "ít gợi ý" thành "chính
    xác". Người đọc bảng kết quả không có cách nào biết mô hình đó chỉ trả 2 item.
    """
    # 20 item, chỉ 4 item có điểm khác 0 → mô hình "ít tín hiệu" hơn hẳn k=10
    scores = np.zeros((1, 20), dtype="float32")
    scores[0, :4] = [9.0, 8.0, 1.0, 0.5]
    dataset = build_dataset(
        train_pairs=[(0, 0, 5.0)],   # item 0 bị mask vì đã xem
        test_pairs=[(0, 1, 5.0)],   # item 1 relevant
        n_users=1, n_items=20, eval_users=[0],
    )

    per_user, _ = evaluate_model(FixedScoreRecommender(scores), dataset, k_list=(10,))

    # 1 hit trong k=10 -> P@10 = 1/10, KHÔNG phải 1/2 hay 1/3
    assert per_user["P@10"][0] == pytest.approx(0.1)


def test_b1_compute_metrics_dung_cong_thuc_khi_k_lon_hon_so_item_co_danh_sach():
    """Đối chiếu trực tiếp với công thức: P@k = hits/k, R@k = hits/|Rel_u|."""
    rel_flags = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    out = metrics_mod.compute_metrics(rel_flags, np.array([4.0]), k_list=(3,))

    assert out["P@3"][0] == pytest.approx(1 / 3)
    assert out["R@3"][0] == pytest.approx(1 / 4)
    assert out["HR@3"][0] == pytest.approx(1.0)


def test_b1_gia_tri_P_k_khong_bao_gi_o_khi_hits_khong_doi():
    """Cùng số hit nhưng k khác nhau thì P@k phải khác nhau (hits/len_top_k sẽ bằng nhau)."""
    rel_flags = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    out = metrics_mod.compute_metrics(rel_flags, np.array([2.0]), k_list=(2, 4))

    assert out["P@2"][0] == pytest.approx(0.5)
    assert out["P@4"][0] == pytest.approx(0.25)


def test_b1_k_lon_hon_so_item_xep_hang_thi_bao_loi():
    """Guard đảm bảo mẫu số của P@k luôn là k: không cho tính P@10 khi chỉ xếp hạng 4 item.

    Nếu cho phép, `P@10` sẽ phải chia cho 10 trong khi danh sách chỉ có 4 item — và
    đó chính là chỗ dễ rơi về `hits / len(top_k)`.
    """
    with pytest.raises(ValueError, match="vượt quá số item được xếp hạng"):
        metrics_mod.compute_metrics(
            np.zeros((1, 4), dtype="float32"), np.array([1.0]), k_list=(10,))


# ── B2: exception phải nổi lên ────────────────────────────────────────────

def test_b2_exception_tu_model_khong_bi_nuot():
    """Lỗi mô hình phải làm run fail, không được biến thành điểm số 0.

    Nuốt exception thành mảng 0 là kiểu lỗi tệ nhất trong pipeline đánh giá: mô hình
    hỏng vẫn ra một dòng kết quả trông rất bình thường, chỉ là thấp hơn.
    """
    dataset = build_dataset(
        train_pairs=[(0, 0, 5.0)], test_pairs=[(0, 1, 5.0)],
        n_users=1, n_items=3, eval_users=[0],
    )

    with pytest.raises(RuntimeError, match="mô hình hỏng"):
        evaluate_model(ExplodingRecommender(), dataset, k_list=(2,))


def test_b2_diem_nan_vao_diem_luon_bao_loi():
    """Điểm NaN trong score_users phải raise — NaN làm hỏng thứ hạng và biến metric thành NaN."""
    scores = np.array([[1.0, np.nan, 2.0]], dtype="float32")
    dataset = build_dataset(
        train_pairs=[(0, 0, 5.0)], test_pairs=[(0, 2, 5.0)],
        n_users=1, n_items=3, eval_users=[0],
    )

    with pytest.raises(ValueError, match="NaN"):
        evaluate_model(FixedScoreRecommender(scores), dataset, k_list=(2,))


# ── Hàng rào catalog ──────────────────────────────────────────────────────

def test_item_ngoai_catalog_phai_bao_loi():
    """Item gợi ý nằm ngoài [0, n_items) phải raise rõ ràng.

    Không có hàng rào này thì `searchsorted` so sánh luôn False và item sai bị tính
    là "không trúng" — một lỗi ánh xạ item biến thành điểm số thấp hơn mà không ai biết.
    """
    rel = sparse.csr_matrix((np.ones(1, dtype="float32"), ([0], [1])), shape=(1, 4))
    topk = np.array([[0, 1, 99]], dtype="int32")   # 99 >= n_items = 4

    with pytest.raises(ValueError, match="ngoài catalog"):
        metrics_mod.gather_relevance(rel, topk)


def test_item_am_phai_bao_loi():
    rel = sparse.csr_matrix((np.ones(1, dtype="float32"), ([0], [1])), shape=(1, 4))
    topk = np.array([[0, -1, 2]], dtype="int32")

    with pytest.raises(ValueError, match="ngoài catalog"):
        metrics_mod.gather_relevance(rel, topk)


def test_item_trong_catalog_van_binh_thuong():
    rel = sparse.csr_matrix((np.ones(1, dtype="float32"), ([0], [1])), shape=(1, 4))
    topk = np.array([[0, 1, 3]], dtype="int32")

    flags = metrics_mod.gather_relevance(rel, topk)

    assert flags[0].tolist() == [0.0, 1.0, 0.0]


# ── User không có tín hiệu: metric = 0 một cách có nguyên tắc ─────────────

def test_user_khong_co_tinh_hieu_co_metric_bang_0():
    """Mô hình trả điểm bằng nhau trên mọi item (không có tín hiệu) → mọi metric = 0.

    `argpartition` khi đó trả về item theo thứ tự chỉ số của mảng. Danh sách ấy do
    cách lưu trữ quyết định chứ không phải do thuật toán — nếu giữ lại, user có thể
    "trúng" item relevant chỉ vì item đó tình cờ nằm ở chỉ số thấp.
    """
    # user 0: không có tương tác relevant nào trong train -> điểm toàn 0
    # user 1: có tín hiệu -> điểm khác nhau
    scores = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 9.0],
    ], dtype="float32")
    dataset = build_dataset(
        train_pairs=[(1, 0, 5.0)],   # user 0 không có gì trong train
        test_pairs=[(0, 0, 5.0), (1, 3, 5.0)],
        n_users=2, n_items=4, eval_users=[0, 1],
    )

    per_user, _ = evaluate_model(FixedScoreRecommender(scores), dataset, k_list=(2,))

    for metric in ("P@2", "R@2", "NDCG@2", "HR@2"):
        assert per_user[metric][0] == 0.0, f"{metric} của user không tín hiệu phải bằng 0"
    # user 1 vẫn được chấm điểm bình thường
    assert per_user["NDCG@2"][1] > 0.0


def test_user_khong_co_tinh_hieu_van_con_trong_tap_danh_gia():
    """Loại user khỏi tập đánh giá là thao túng mẫu — họ phải ở lại với metric = 0."""
    scores = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 9.0],
    ], dtype="float32")
    dataset = build_dataset(
        train_pairs=[(1, 0, 5.0)],
        test_pairs=[(0, 0, 5.0), (1, 3, 5.0)],
        n_users=2, n_items=4, eval_users=[0, 1],
    )

    per_user, topk = evaluate_model(FixedScoreRecommender(scores), dataset, k_list=(2,))

    assert len(topk) == 2, "user không có tín hiệu không được bị loại khỏi tập đánh giá"
    assert len(per_user["NDCG@2"]) == 2


def test_user_khong_co_tinh_hieu_khong_duoc_diem_ngau_nhien():
    """Danh sách gợi ý của user không tín hiệu phải là 0 về mặt metric, không phải 'trúng'."""
    # item relevant của user 0 nằm ở chỉ số 0 — thứ hạng tự nhiên của nó là 1
    scores = np.array([[0.0, 0.0, 0.0, 0.0]], dtype="float32")
    dataset = build_dataset(
        train_pairs=[], test_pairs=[(0, 0, 5.0)],
        n_users=1, n_items=4, eval_users=[0],
    )

    per_user, _ = evaluate_model(FixedScoreRecommender(scores), dataset, k_list=(2,))

    assert per_user["HR@2"][0] == 0.0
    assert per_user["NDCG@2"][0] == 0.0


def test_rows_without_signal_nhan_dien_dung():
    """Mặt nạ: hàng không có điểm hữu hạn nào, hoặc mọi điểm hữu hạn bằng nhau."""
    scores = np.array([
        [1.0, 2.0, 3.0],     # có tín hiệu
        [5.0, 5.0, 5.0],     # bằng nhau -> không có tín hiệu
        [-np.inf, -np.inf, -np.inf],  # hết item sau khi mask -> không có tín hiệu
        [0.0, 1.0, 1.0],     # có tín hiệu
    ], dtype="float32")

    mask = rows_without_signal(scores)

    assert mask.tolist() == [False, True, True, False]


def test_user_co_tinh_hieu_khong_bi_zero():
    """Guard không được vô tình zero cả những user có xếp hạng hợp lệ."""
    # 2 user, điểm khác nhau ở hàng đầu, hàng sau bị mask hết
    scores = np.array([
        [9.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
    ], dtype="float32")
    dataset = build_dataset(
        train_pairs=[(1, 0, 5.0), (1, 1, 5.0), (1, 2, 5.0)],  # user 1 xem hết
        test_pairs=[(0, 2, 5.0), (1, 0, 5.0)],
        n_users=2, n_items=3, eval_users=[0, 1],
    )

    per_user, _ = evaluate_model(FixedScoreRecommender(scores), dataset, k_list=(2,))

    assert per_user["NDCG@2"][0] > 0.0, "user có tín hiệu không được bị zero"
    assert per_user["NDCG@2"][1] == 0.0


# ── B4: split được kiểm bằng test riêng, không có mặc định 0,7 ────────────

def test_b4_khong_co_ham_validate_split_mac_dinh():
    """Không tồn tại hàm validate_dataset_split với tỷ lệ mặc định 0,7.

    Protocol cũ có `validate_dataset_split(..., train_ratio=0.7)` — một hằng số mặc
    định sai lệch với split 80/20 đang dùng, và hàm đó trả về cảnh báo chứ không
    chặn. Ở đường chạy mới, split được kiểm bằng `tests/test_split_temporal_per_user.py`.
    """
    from src.data import build_interaction_matrices

    assert not hasattr(build_interaction_matrices, "validate_dataset_split")


def test_b4_split_temporal_duoc_kiem_tra_trong_test_rieng():
    """Tỷ lệ 80/20 được chốt bằng test riêng, không bằng hằng số trong runtime."""
    from src.data.split_temporal_per_user import TEST_RATIO

    assert TEST_RATIO == pytest.approx(0.2)


# ── B3: ghép cặp thống kê theo tập user ──────────────────────────────────

def test_b3_bo_sung_khong_ghep_cặp_voi_tap_user_khac():
    """Hai thí nghiệm có metric per-user khác tập user thì không ghép cặp được."""
    from src.eval.statistical_tests import load_all

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        rdir = Path(tmp)
        np.savez(rdir / "a_peruser.npz",
                 eval_users=np.asarray([0, 1, 2], dtype="int32"),
                 **{"NDCG@10": np.asarray([0.1, 0.2, 0.3], dtype="float32")})
        np.savez(rdir / "b_peruser.npz",
                 eval_users=np.asarray([0, 1, 9], dtype="int32"),
                 **{"NDCG@10": np.asarray([0.1, 0.2, 0.3], dtype="float32")})

        with pytest.raises(SystemExit, match="tập user khác"):
            load_all("", "NDCG@10", results_dir=rdir)
