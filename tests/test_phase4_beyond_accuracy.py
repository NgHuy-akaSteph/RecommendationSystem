"""Test các chỉ số ngoài độ chính xác và kiểm định thống kê của phase 4."""

import numpy as np
import pytest

from src.eval.beyond_accuracy_metrics import (
    coverage, frac_unseen_items, gini, intra_list_diversity, novelty, top_share)
from src.eval.statistical_tests import bootstrap_ci, wilcoxon_against


# ── coverage ─────────────────────────────────────────────────────────────

def test_coverage_counts_distinct_items():
    topk = np.array([[0, 1, 2], [1, 2, 3]])      # 4 item khác nhau trên tổng 10
    assert coverage(topk, n_items=10) == pytest.approx(0.4)


def test_coverage_is_one_when_every_item_recommended():
    assert coverage(np.arange(6).reshape(2, 3), n_items=6) == pytest.approx(1.0)


def test_coverage_minimal_when_all_users_get_same_list():
    """Đây chính là mẫu hình của MostPop: mọi người nhận cùng một danh sách."""
    topk = np.tile(np.array([0, 1, 2]), (50, 1))
    assert coverage(topk, n_items=100) == pytest.approx(0.03)


# ── gini ─────────────────────────────────────────────────────────────────

def test_gini_is_zero_when_load_is_perfectly_even():
    """Mỗi phim được gợi ý đúng một lần → hoàn toàn đều → Gini = 0."""
    topk = np.arange(6).reshape(2, 3)
    assert gini(topk, n_items=6) == pytest.approx(0.0, abs=1e-9)


def test_gini_approaches_one_when_all_load_on_one_item():
    """Dồn hết vào một phim trong catalog lớn → Gini → (n−1)/n."""
    topk = np.zeros((50, 3), dtype="int64")
    assert gini(topk, n_items=1000) == pytest.approx(999 / 1000, abs=1e-9)


def test_gini_counts_items_never_recommended():
    """Phim chưa bao giờ được gợi ý phải được tính vào — đó chính là phần cần đo.

    Nếu chỉ tính trên các phim đã xuất hiện thì một mô hình chỉ biết 2 phim nhưng chia
    đều cho chúng sẽ được Gini = 0, tức được khen là "trải đều" một cách vô lý.
    """
    topk = np.array([[0, 1], [0, 1]])
    assert gini(topk, n_items=2) == pytest.approx(0.0, abs=1e-9)
    assert gini(topk, n_items=100) > 0.97


def test_gini_separates_broad_from_concentrated_at_equal_coverage():
    """Hai mô hình **cùng coverage, cùng số suất** nhưng khác mức tập trung.

    Đây chính là chỗ coverage mù: cả hai chạm đúng 20/100 phim với đúng 40 suất gợi ý,
    nhưng một bên chia đều còn một bên dồn hơn một nửa số suất vào một phim.
    """
    n_items = 100
    broad = np.tile(np.arange(20), 2).reshape(20, 2)            # mỗi phim đúng 2 lần
    concentrated = np.array([0] * 21 + list(range(1, 20))).reshape(20, 2)

    assert coverage(broad, n_items) == coverage(concentrated, n_items) == pytest.approx(0.20)
    assert broad.size == concentrated.size == 40
    assert gini(broad, n_items) < gini(concentrated, n_items)


def test_gini_handles_empty_recommendations():
    assert gini(np.empty((0, 10), dtype="int64"), n_items=50) == pytest.approx(0.0)


# ── novelty ──────────────────────────────────────────────────────────────

def test_novelty_matches_self_information():
    """Làm trơn add-one: p = (pop + 1) / (n_user + 1)."""
    pop = np.array([50.0, 10.0, 1.0])             # trên 100 user
    topk = np.array([[0, 2]])
    expected = (-np.log2(51 / 101) + -np.log2(2 / 101)) / 2
    assert novelty(topk, pop, n_users=100) == pytest.approx(expected, rel=1e-6)


def test_novelty_is_bounded_by_log_n_users():
    """Trần của novelty phải là log2(n+1), không phải hằng số tuỳ tiện từ phép kẹp EPS.

    Kẹp bằng EPS=1e-12 từng gán cho item popularity 0 giá trị 39,9 bit — vô nghĩa và
    chiếm phần lớn giá trị novelty báo cáo ở giao thức cold-start.
    """
    value = novelty(np.array([[0]]), np.array([0.0]), n_users=100)
    assert value == pytest.approx(np.log2(101), rel=1e-6)
    assert value < 7.0


def test_novelty_higher_for_rarer_items():
    pop = np.array([90.0, 1.0])
    popular = novelty(np.array([[0]]), pop, n_users=100)
    rare = novelty(np.array([[1]]), pop, n_users=100)
    assert rare > popular


def test_frac_unseen_reports_cold_start_effect_directly():
    """Tỷ lệ item chưa có tương tác train — đại lượng mà cold-start thực sự quan tâm."""
    pop = np.array([0.0, 5.0, 0.0, 9.0])
    assert frac_unseen_items(np.array([[0, 1]]), pop) == pytest.approx(0.5)
    assert frac_unseen_items(np.array([[1, 3]]), pop) == pytest.approx(0.0)
    assert frac_unseen_items(np.array([[0, 2]]), pop) == pytest.approx(1.0)


# ── intra-list diversity ─────────────────────────────────────────────────

def test_ild_zero_for_identical_items():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype="float32")
    assert intra_list_diversity(np.array([[0, 1, 2]]), emb) == pytest.approx(0.0, abs=1e-6)


def test_ild_one_for_orthogonal_items():
    emb = np.eye(3, dtype="float32")
    assert intra_list_diversity(np.array([[0, 1, 2]]), emb) == pytest.approx(1.0, abs=1e-6)


def test_ild_two_for_opposite_items():
    """cosine = −1 → 1 − cos = 2. Xảy ra thật với genome đã z-score."""
    emb = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype="float32")
    assert intra_list_diversity(np.array([[0, 1]]), emb) == pytest.approx(2.0, abs=1e-6)


def test_ild_matches_manual_pairwise_mean():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((8, 4)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    topk = np.array([[0, 3, 5, 7], [1, 2, 4, 6]])

    manual = []
    for row in topk:
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                manual.append(1.0 - float(emb[row[a]] @ emb[row[b]]))
    assert intra_list_diversity(topk, emb) == pytest.approx(np.mean(manual), rel=1e-5)


def test_ild_batching_does_not_change_result():
    """Vòng lặp theo lô 4000 phải cho đúng kết quả như tính một lần."""
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((20, 5)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    small = np.tile(np.array([0, 1, 2]), (10, 1))
    large = np.tile(np.array([0, 1, 2]), (9000, 1))
    assert intra_list_diversity(small, emb) == pytest.approx(intra_list_diversity(large, emb),
                                                             rel=1e-5)


# ── bootstrap ────────────────────────────────────────────────────────────

def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = np.random.default_rng(2).normal(0.1, 0.05, size=5000).astype("float32")
    lo, hi = bootstrap_ci(values, n_boot=200)
    assert lo < values.mean() < hi
    assert (lo, hi) == bootstrap_ci(values, n_boot=200)     # cùng seed → cùng kết quả


def test_bootstrap_ci_narrows_with_more_users():
    rng = np.random.default_rng(3)
    small = rng.normal(0.1, 0.05, size=200).astype("float32")
    large = rng.normal(0.1, 0.05, size=20000).astype("float32")
    width = lambda v: np.subtract(*reversed(bootstrap_ci(v, n_boot=200)))
    assert width(large) < width(small)


# ── Wilcoxon ─────────────────────────────────────────────────────────────

def test_wilcoxon_detects_consistent_improvement():
    rng = np.random.default_rng(4)
    base = rng.random(500).astype("float32")
    per_user = {"tốt": base + 0.05, "kém": base}
    row = wilcoxon_against(per_user, "tốt").iloc[0]
    assert row["chênh_lệch_trung_bình"] == pytest.approx(0.05, rel=1e-3)
    assert row["tỷ_lệ_user_tốt_hơn"] == pytest.approx(1.0)
    assert row["có_ý_nghĩa_p<0.05"]


def test_wilcoxon_finds_no_difference_for_identical_models():
    values = np.random.default_rng(5).random(300).astype("float32")
    row = wilcoxon_against({"a": values, "b": values.copy()}, "a").iloc[0]
    assert row["p_value"] == pytest.approx(1.0)
    assert not row["có_ý_nghĩa_p<0.05"]


def test_wilcoxon_reports_win_rate_separately_from_mean():
    """Trung bình và tỷ lệ user thắng có thể lệch nhau — cần báo cả hai.

    Ở đây mô hình A thua ở 90% user nhưng thắng đậm ở 10% còn lại nên trung bình vẫn dương.
    """
    base = np.zeros(100, dtype="float32")
    a = base.copy(); a[:90] = -0.01; a[90:] = 0.5
    row = wilcoxon_against({"a": a, "b": base}, "a").iloc[0]
    assert row["chênh_lệch_trung_bình"] > 0
    assert row["tỷ_lệ_user_tốt_hơn"] == pytest.approx(0.1)


# ── top_share ────────────────────────────────────────────────────────────

def test_top_share_counts_most_recommended_items():
    # item 0 xuất hiện 3 lần, item 1 hai lần, item 2 một lần → top-1 = 3/6
    topk = np.array([[0, 1], [0, 1], [0, 2]])
    assert top_share(topk, 1) == pytest.approx(0.5)
    assert top_share(topk, 2) == pytest.approx(5 / 6)
    assert top_share(topk, 100) == pytest.approx(1.0)
