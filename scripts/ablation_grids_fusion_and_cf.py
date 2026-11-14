"""Lưới tham số cho baseline CF, popularity prior, hybrid late fusion và time decay.

Các hằng ``BEST_*`` là ảnh chụp cấu hình thắng cuộc của bước trước trong chuỗi ablation
tuần tự. Đổi chúng thì `run_all_experiments` so params với kết quả đã lưu và tự chạy lại,
không âm thầm giữ số cũ.
"""

from ablation_grids_content_models import TFIDF_BASE

KNN_K_GRID = [20, 50, 100, 200]

# Cấu hình thắng cuộc của phase 2, dùng lại làm component cho pop prior và hybrid
BEST_CB1 = {"model": "cb1-tfidf",
            "params": {**TFIDF_BASE, "scorer": "profile", "weight_mode": "binary_pos"}}
BEST_CB3 = {"model": "cb3-genome",
            "params": {"standardize": "zscore", "threshold": 0.2, "scorer": "item_knn",
                       "weight_mode": "binary_pos", "k_neighbors": 100}}
# λ tăng đơn điệu tới 0.4 nên phải nới: khi λ → ∞ mô hình hội tụ về MostPop (0.0723),
# tức tồn tại cực đại bên trong; grid dừng ở biên là chưa tìm ra nó.
POP_PRIOR_GRID = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
# w=0.75 cho 0.0987 > cả CB-1 đơn lẻ (w=1.0 → 0.0914) nên cực đại nằm giữa 0.75 và 1.0
LATE_FUSION_GRID = [0.25, 0.5, 0.75, 0.85, 0.95]  # trọng số CB-1, còn lại cho CB-3


def pop_prior_configs() -> list[tuple[str, str, dict]]:
    """Trục đối chứng: popularity prior áp lên **từng** mô hình riêng, trước khi hybrid.

    Không có trục này thì khi hybrid thắng sẽ không tách được "hai biểu diễn bổ sung cho
    nhau" khỏi "chỉ cần thêm độ phổ biến". λ=0 chính là kết quả phase 2 nên không chạy lại.
    """
    out = []
    for tag, comp in (("cb1", BEST_CB1), ("cb3", BEST_CB3)):
        for lam in POP_PRIOR_GRID:
            out.append((f"cb4-00-pop-{tag}-lam{round(lam * 100):03d}", "cb4-hybrid",
                        {"components": [comp], "pop_prior": lam}))
    return out


def late_fusion_configs() -> list[tuple[str, str, dict]]:
    """Hybrid CB-1 + CB-3, content thuần (λ_pop = 0) để tách khỏi hiệu ứng popularity."""
    return [
        (f"cb4-01-late-w{round(w * 100):03d}", "cb4-hybrid",
         {"components": [BEST_CB1, BEST_CB3], "weights": [w, 1 - w]})
        for w in LATE_FUSION_GRID
    ]


BEST_FUSION_W = 0.75      # cực đại của late_fusion_configs (0.0987)
# CB-2 BM25 (k1=0.5, b=0) đạt 0.0986 > CB-1 TF-IDF 0.0914 nên nó mới là thành phần nội
# dung mạnh nhất; spec phase-03 bước 6 đã dự trù "thêm BM25 nếu có".
BEST_CB2 = {"model": "cb2-bm25",
            "params": {"k1": 0.5, "b": 0.0, "min_df": 3, "max_df": 0.5,
                       "max_features": 20_000, "scorer": "profile",
                       "weight_mode": "binary_pos"}}
HALF_LIFE_GRID = [90, 365, 1095]   # ngày: 3 tháng / 1 năm / 3 năm


def hybrid_pop_configs() -> list[tuple[str, str, dict]]:
    """Bước cuối: hybrid tốt nhất + popularity prior. Báo cáo cả hai cột."""
    return [
        (f"cb4-02-late-pop-lam{round(lam * 100):04d}", "cb4-hybrid",
         {"components": [BEST_CB1, BEST_CB3],
          "weights": [BEST_FUSION_W, 1 - BEST_FUSION_W], "pop_prior": lam})
        for lam in [0.2, 0.4, 0.8, 1.6]
    ]


def time_decay_configs() -> list[tuple[str, str, dict]]:
    """Suy giảm theo thời gian trên CB-1 (thành phần chi phối của hybrid).

    Đặt trên mô hình đơn lẻ chứ không trên hybrid để tách được đóng góp của time-decay
    khỏi đóng góp của fusion.
    """
    return [
        (f"cb4-03-timedecay-hl{hl:04d}", "cb1-tfidf",
         {**TFIDF_BASE, "scorer": "profile", "weight_mode": "binary_pos",
          "half_life_days": float(hl)})
        for hl in HALF_LIFE_GRID
    ]


def bm25_fusion_configs() -> list[tuple[str, str, dict]]:
    """Fusion với BM25 thay cho TF-IDF, rồi thêm pop prior lên cấu hình tốt nhất."""
    out = [
        (f"cb4-04-late-bm25-w{round(w * 100):03d}", "cb4-hybrid",
         {"components": [BEST_CB2, BEST_CB3], "weights": [w, 1 - w]})
        for w in [0.5, 0.75, 0.85]
    ]
    # BM25 và TF-IDF cùng dựa trên tag nên tương quan cao; thêm một cấu hình 3 thành
    # phần để kiểm chứng xem gộp cả hai có còn biên lợi nào không
    out.append(("cb4-05-late-three-way", "cb4-hybrid",
                {"components": [BEST_CB2, BEST_CB1, BEST_CB3],
                 "weights": [0.45, 0.3, 0.25]}))
    for lam in [0.4, 0.8, 1.6]:
        out.append((f"cb4-06-late-bm25-pop-lam{round(lam * 100):04d}", "cb4-hybrid",
                    {"components": [BEST_CB2, BEST_CB3], "weights": [0.75, 0.25],
                     "pop_prior": lam}))
    return out


def cf_configs() -> list[tuple[str, str, dict]]:
    """Baseline collaborative filtering (phase 3), làm trước Hybrid để có mốc so sánh."""
    out = []
    # ItemKNN-CF dùng lại đúng scorer của CB item-KNN, chỉ đổi biểu diễn item
    for k in [50, 100, 200]:
        out.append((f"cf-itemknn-k{k:03d}", "cf-itemknn", {"k_neighbors": k,
                                                           "weight_mode": "binary_pos"}))
    # ALS: confidence = 1 + alpha trên tương tác relevant (xem als_cf_model).
    # Grid gốc {10, 40} cho alpha=10 thắng, nhưng 10 là cận dưới của grid → nới xuống
    # {1, 5} mới biết đâu là cực trị thật (đúng bài học của phase 2).
    for factors in [64, 128]:
        for alpha in [1.0, 5.0, 10.0, 40.0]:
            out.append((f"cf-als-f{factors:03d}-a{int(alpha):03d}", "cf-als",
                        {"factors": factors, "regularization": 0.01, "iterations": 15,
                         "alpha": alpha}))
    # Quét regularization và iterations trên (factors=64, alpha=5) — cực đại của lưới trên.
    # Không nhằm lật kết luận "CF thắng CB": ALS tinh chỉnh thiếu chỉ làm khoảng cách hẹp
    # lại, tức ước lượng hiện tại đã bảo thủ. Mục đích là đóng câu hỏi "baseline có được
    # tinh chỉnh công bằng không", vốn chắc chắn bị hỏi khi chính baseline thắng.
    # reg=0.01 là rất thấp cho ma trận 162k x 13.8k nên hoàn toàn có thể đang overfit nhẹ.
    for reg, tag in [(0.001, "0p001"), (0.1, "0p1"), (1.0, "1p0")]:
        out.append((f"cf-als-f064-a005-reg{tag}", "cf-als",
                    {"factors": 64, "regularization": reg, "iterations": 15, "alpha": 5.0}))
    # iterations=30 để xác nhận 15 vòng đã đủ hội tụ
    out.append(("cf-als-f064-a005-it030", "cf-als",
                {"factors": 64, "regularization": 0.01, "iterations": 30, "alpha": 5.0}))
    # Lưới đầy đủ reg × iterations (4 × 2 = 8) cho phase 03: CB nhận 103 config còn CF
    # chỉ 18 nên kết luận "CF thắng" dễ bị nghi ngờ là tinh chỉnh thiếu công bằng.
    # Ba ô còn lại của lưới này (reg ∈ {0,001; 0,1; 1,0} × it=30) chưa có ở trên.
    for reg, tag in [(0.001, "0p001"), (0.1, "0p1"), (1.0, "1p0")]:
        out.append((f"cf-als-f064-a005-reg{tag}-it030", "cf-als",
                    {"factors": 64, "regularization": reg, "iterations": 30, "alpha": 5.0}))
    return out
