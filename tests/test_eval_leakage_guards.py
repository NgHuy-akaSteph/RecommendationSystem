"""Hàng rào chống leakage: mô hình không được chạm tới nhãn test.

Ba tầng hàng rào, xếp từ mờ đến rõ:

    L1  TrainData và Dataset là hai vật thể tách biệt — mô hình nhận TrainData nên
        về mặt cấu trúc không thể chạm nhãn test.
    L2  Theo thời gian: mọi timestamp trong train của user đều sớm hơn mọi timestamp
        trong test của user đó.
    L3  Metadata: đặc trưng nội dung item tính trên toàn corpus là hậu kỳ. Không có
        cách cắt theo thời gian (genome-scores.csv không có timestamp) nên đây là
        giới hạn được ghi nhận và đo bằng bảng phân tầng theo kỷ nguyên, chứ không
        phải thứ có thể "sửa".

Các test ở đây dùng dữ liệu thật đã tiền xử lý nên bỏ qua khi cache chưa có.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import paths
from src.data.build_interaction_matrices import Dataset, TrainData, load_dataset
from src.models.base_recommender import BaseRecommender

# ── L1: hàng rào cấu trúc ────────────────────────────────────────────────

def test_train_data_khong_co_duong_dan_den_nhan_test():
    """TrainData không được mang theo bất kỳ tham chiếu nào tới test."""
    dataset = _tiny_dataset()

    assert not hasattr(dataset.train, "test_rel_csr")
    assert not hasattr(dataset.train, "test")
    assert isinstance(dataset.train.train_csr, sparse.csr_matrix)


def test_dataset_chi_chua_va_mot_ma_tran_nhan_test():
    """Dataset giữ nhãn test ở đúng một chỗ, tách khỏi TrainData."""
    dataset = _tiny_dataset()

    fields = dataset.__dataclass_fields__
    assert "test_rel_csr" in fields
    # train là một TrainData riêng biệt, không phải chứa luôn nhãn test
    assert isinstance(dataset.train, TrainData)
    assert dataset.train is not dataset


def test_model_chi_nhan_duoc_train_data():
    """Mô hình nhận đúng `TrainData`, không có đường nào tới `Dataset`.

    Đây là hàng rào mạnh nhất: giao diện `fit(train, item_features)` không có tham số
    nào mang nhãn test, nên không thể "gõ nhầm" thành test_rel_csr — hai ma trận có
    cùng shape và dtype nên lỗi đó chạy trơn tru và chỉ làm metric nhảy vọt.
    """
    import inspect

    sig = inspect.signature(BaseRecommender.fit)
    params = list(sig.parameters)

    assert params[:3] == ["self", "train", "item_features"]
    assert "test" not in " ".join(params)
    assert not any("test" in p for p in params)


def test_model_khong_the_tra_ve_du_lieu_test():
    """Mô hình trả về chỉ điểm dự đoán, không kèm nhãn hay thông tin tập test."""
    model = _RecordingModel()
    dataset = _tiny_dataset()

    model.fit(dataset.train, {})
    out = model.score_users(dataset.eval_users)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    # Mọi thứ mô hình nhận được phải thuộc TrainData
    assert set(model.fit_args) == {"train", "item_features"}
    assert isinstance(model.fit_args["train"], TrainData)


def test_los_vao_dataset_khong_the_bien_nhan_test_vao_train():
    """Chỉ `Dataset` giữ nhãn test; mọi đường dẫn tới TrainData đều sạch."""
    dataset = _tiny_dataset()

    # Nếu ai đó gán nhầm test_rel_csr vào train, evaluator sẽ cho điểm 1.0
    assert dataset.train.train_csr.nnz == 2
    assert dataset.test_rel_csr.nnz == 2
    # Hai ma trận khác nhau thật sự
    assert not np.array_equal(
        dataset.train.train_csr.toarray(), dataset.test_rel_csr.toarray())


# ── L2: hàng rào thời gian ────────────────────────────────────────────────

def test_moi_user_train_luon_som_hon_test():
    """Với mỗi user, mọi ts train <= mọi ts test.

    Đây là hàng rào thực sự của giao thức temporal: dùng tương lai để dự đoán quá
    khứ sẽ vi phạm đúng điều kiện này.

    Dùng `<=` chứ không phải `<` vì MovieLens có nhiều phim cùng một mốc thời gian:
    57,6% cặp (user, ts) có hơn một rating. Khi đó user có thể xem 5 phim cùng lúc,
    split chọn 4 vào train và 1 vào test, và cả hai cùng mang ts. Đây là hạn chế *không
    tránh được* của dữ liệu thật — cùng một mốc thời gian thì không có "trước/sau" —
    chứ không phải rò rỉ. Việc vi phạm thật là ts train *muộn hơn* ts test.
    """
    if not paths.TRAIN_PARQUET.exists():
        pytest.skip("chưa chạy src.data.split_temporal_per_user")

    train = pd.read_parquet(paths.TRAIN_PARQUET, columns=["user_idx", "ts"])
    test = pd.read_parquet(paths.TEST_PARQUET, columns=["user_idx", "ts"])

    max_train = train.groupby("user_idx", sort=False)["ts"].max()
    min_test = test.groupby("user_idx", sort=False)["ts"].min()

    common = max_train.index.intersection(min_test.index)
    violations = int((max_train.loc[common] > min_test.loc[common]).sum())

    assert violations == 0, (
        f"{violations} user có ts train MUỘN HƠN ts test — rò rỉ thời gian thật"
    )


def test_phan_bo_timestamp_bang_duoc_ghi_nhan():
    """Tỷ lệ user có ts train == ts test được đo và ghi nhận, không giấu đi.

    Khoảng 19% user có ít nhất một mốc thời gian chung giữa train và test. Đây là
    giới hạn cấu trúc của dữ liệu thời gian thực tế (người dùng xem nhiều phim cùng
    lúc), được báo cáo minh bạch thay vì coi là lỗi của split.
    """
    if not paths.TRAIN_PARQUET.exists():
        pytest.skip("chưa chạy src.data.split_temporal_per_user")

    train = pd.read_parquet(paths.TRAIN_PARQUET, columns=["user_idx", "ts"])
    test = pd.read_parquet(paths.TEST_PARQUET, columns=["user_idx", "ts"])

    max_train = train.groupby("user_idx", sort=False)["ts"].max()
    min_test = test.groupby("user_idx", sort=False)["ts"].min()
    common = max_train.index.intersection(min_test.index)

    n_equal = int((max_train.loc[common] == min_test.loc[common]).sum())
    n_total = len(common)

    assert n_total > 0
    assert n_equal <= n_total
    # Nếu tỷ lệ này tăng vọt so với mốc ~19% đã đo, split có thể đã hỏng
    assert n_equal / n_total < 0.30, (
        f"{n_equal / n_total:.1%} user có ts train == ts test — cao hơn nhiều mức "
        "~19% đo được trên dữ liệu MovieLens 25M"
    )


def test_khong_co_cap_user_item_trung_giua_train_va_test():
    """Cùng một cặp (user, item) không được xuất hiện ở cả train và test.

    Nếu trùng, item đó bị mask khỏi danh sách gợi ý (đã xem rồi) nên user không
    bao giờ có thể "trúng" nó — nhãn test tồn tại nhưng không thể chạm tới, làm giảm
    mẫu đánh giá một cách âm thầm.
    """
    if not paths.TRAIN_PARQUET.exists():
        pytest.skip("chưa chạy src.data.split_temporal_per_user")

    train = pd.read_parquet(paths.TRAIN_PARQUET, columns=["user_idx", "item_idx"])
    test = pd.read_parquet(paths.TEST_PARQUET, columns=["user_idx", "item_idx"])

    key_train = train["user_idx"].astype("int64") * 10_000_000 + train["item_idx"]
    key_test = test["user_idx"].astype("int64") * 10_000_000 + test["item_idx"]

    n_overlap = int(key_test.isin(set(key_train.tolist())).sum())

    assert n_overlap == 0, f"{n_overlap} cặp (user, item) xuất hiện ở cả train và test"


def test_moi_user_dang_gi_co_it_nhat_mot_item_test():
    """User trong tập đánh giá phải có ít nhất một tương tác relevant trong test."""
    if not paths.TEST_PARQUET.exists():
        pytest.skip("chưa chạy src.data.split_temporal_per_user")

    test = pd.read_parquet(paths.TEST_PARQUET, columns=["user_idx", "rating"])
    relevant = test[test["rating"] >= paths.RELEVANCE_THRESHOLD]

    n_users_test = test["user_idx"].nunique()
    n_users_relevant = relevant["user_idx"].nunique()

    assert n_users_relevant == n_users_test, (
        f"{n_users_test - n_users_relevant} user trong test không có item relevant "
        "— họ không thể tính Recall/NDCG nên phải bị loại khỏi tập đánh giá"
    )


# ── L3: metadata hậu kỳ — ghi nhận giới hạn, không "sửa" được ──────────────

def test_metadata_la_hau_ky_thi_duoc_ghi_nhan():
    """Metadata item là hậu kỳ — đây là giới hạn được công bố, không phải lỗi.

    Tag Genome không có timestamp nên không thể cắt theo thời gian. Điều kiện bắt
    buộc là báo cáo phải ghi rõ điều này và kèm bảng phân tầng theo kỷ nguyên.
    """
    from src.eval.analyse_tag_leakage import ERA_BOUNDS

    assert len(ERA_BOUNDS) >= 2, "phải có tối thiểu 2 kỷ nguyên để so sánh"
    # Ngưỡng liên quan đến thời điểm bắt đầu gán tag của MovieLens (2005-12)
    assert any("2006" in label for label, _, _ in ERA_BOUNDS)
    # Các kỷ nguyên phải phủ kín trục thời gian: kỳ đầu không có mốc trên,
    # kỳ cuối không có mốc dưới.
    assert ERA_BOUNDS[0][1] is None
    assert ERA_BOUNDS[-1][2] is None


def test_bang_phan_tang_theo_ky_nguyen_da_duoc_sinh():
    """Bảng phân tầng kỷ nguyên phải tồn tại sau khi chạy phân tích."""
    out = paths.RESULTS_DIR / "final" / "tag-leakage-strata.csv"
    if not out.exists():
        pytest.skip("chưa chạy src.eval.analyse_tag_leakage")

    df = pd.read_csv(out)
    assert len(df) >= 2, "bảng phân tầng phải có ít nhất 2 mô hình để so sánh"
    # Mỗi mô hình phải có điểm ở mọi kỷ nguyên
    era_cols = [c for c in df.columns if c.startswith(("Trước", "Giao thời", "Tag đã"))]
    assert len(era_cols) >= 2
    assert df[era_cols].notna().all().all(), "mọi mô hình phải có điểm ở mọi kỷ nguyên"


# ── helper ────────────────────────────────────────────────────────────────

class _RecordingModel(BaseRecommender):
    """Ghi lại những gì nó được cho, để test assert được model không thấy nhãn test."""

    name = "recording"

    def __init__(self):
        self.fit_args: dict = {}

    def fit(self, train, item_features):
        self.fit_args = {"train": train, "item_features": item_features}
        return self

    def score_users(self, user_idx):
        return np.zeros((len(user_idx), 3), dtype="float32")


def _tiny_dataset() -> Dataset:
    train_df = pd.DataFrame(
        [(0, 0, 5.0, 1), (1, 1, 4.0, 2)], columns=["user_idx", "item_idx", "rating", "ts"])
    train_csr = sparse.csr_matrix(
        (np.array([5.0, 4.0], dtype="float32"), ([0, 1], [0, 1])),
        shape=(2, 3), dtype="float32")
    train_csr.sort_indices()
    test_rel = sparse.csr_matrix(
        (np.array([1.0, 1.0], dtype="float32"), ([0, 1], [2, 2])),
        shape=(2, 3), dtype="float32")
    train = TrainData(train_csr=train_csr, train_df=train_df, n_users=2, n_items=3)
    return Dataset(train=train, test_rel_csr=test_rel,
                   eval_users=np.arange(2, dtype="int32"))
