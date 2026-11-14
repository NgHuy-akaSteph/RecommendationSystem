"""Lưới tham số cho các mô hình nội dung đơn lẻ: CB-1 TF-IDF, CB-2 BM25, CB-3 Genome.

Mỗi hàm trả về danh sách ``(exp_name, model, params)``. Nhiều bình luận giải thích **vì
sao lưới gốc bị nghi ngờ ở biên** — đó là phần đáng giữ hơn cả bản thân các con số.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

# Tham số TF-IDF giữ nguyên qua các bước sau cb1-01 (bước 1 là bước thắng)
TFIDF_BASE = {
    "use_tags": True, "min_df": 3, "max_df": 0.5,
    "max_features": 20_000, "sublinear_tf": True, "ngram_range": [1, 1],
}
KNN_K_GRID = [20, 50, 100, 200]


def cb1_configs() -> list[tuple[str, str, dict]]:
    """CB-1: genres → +tags → α → bigram → centered → item-KNN."""
    out = [
        ("cb1-00-genres-binary-profile", {"use_tags": False, "scorer": "profile",
                                          "weight_mode": "binary_pos"}),
        ("cb1-01-add-tags-tfidf", {**TFIDF_BASE, "scorer": "profile",
                                   "weight_mode": "binary_pos"}),
    ]
    # Bước 2: α. Thêm 0.05 ngoài grid gốc {0.2,0.4,0.6} vì 19 chiều genre dense đấu
    # với ~13k chiều tag thưa (mật độ 0.26%) — α lớn làm số hạng genre áp đảo cosine.
    for alpha in [0.05, 0.2, 0.4, 0.6]:
        out.append((f"cb1-02-genre-weight-alpha{round(alpha * 100):03d}",
                    {**TFIDF_BASE, "genre_weight": alpha, "scorer": "profile",
                     "weight_mode": "binary_pos"}))
    out.append(("cb1-03-bigram", {**TFIDF_BASE, "max_features": 40_000,
                                  "ngram_range": [1, 2], "scorer": "profile",
                                  "weight_mode": "binary_pos"}))
    out.append(("cb1-04-centered-weights", {**TFIDF_BASE, "scorer": "profile",
                                            "weight_mode": "centered"}))
    # Chế độ thứ ba để bộ trọng số đủ cả ba, phục vụ phân tích theo nhóm độ dài lịch sử.
    # `centered` thua `binary_pos` trên **trung bình toàn cục** (0,0674 vs 0,0914) nhưng
    # đó chưa trả lời được câu của cold-start người dùng: user ít lịch sử có thể quá ít
    # phim ≥4 sao để dựng hồ sơ, và khi đó trọng số có dấu lại hơn. Trung bình toàn cục
    # che mất xu hướng trong nhóm con (nghịch lý Simpson dạng nhẹ), nên phải phân tầng.
    out.append(("cb1-04b-rating-weights", {**TFIDF_BASE, "scorer": "profile",
                                           "weight_mode": "rating"}))
    # k=500 nằm ngoài grid gốc {20,50,100,200}: metric vẫn tăng đơn điệu tới k=200 nên
    # phải đẩy thêm một điểm mới biết item-KNN có bao giờ vượt profile hay không.
    for k in KNN_K_GRID + [500]:
        out.append((f"cb1-05-item-knn-k{k:03d}",
                    {**TFIDF_BASE, "scorer": "item_knn", "weight_mode": "binary_pos",
                     "k_neighbors": k}))
    # Bước 6: ItemKNN chuẩn hoá theo Σ|sim| — spec không nêu, đo để biết có đáng không
    out.append(("cb1-06-item-knn-k050-normsim",
                {**TFIDF_BASE, "scorer": "item_knn", "weight_mode": "binary_pos",
                 "k_neighbors": 50, "normalize_sim": True}))
    return [(name, "cb1-tfidf", params) for name, params in out]


def cb3_configs() -> list[tuple[str, str, dict]]:
    """CB-3: raw → chuẩn hoá → threshold → SVD → item-KNN → ridge."""
    profile = {"scorer": "profile", "weight_mode": "binary_pos"}
    out = [
        ("cb3-00-raw-profile", "cb3-genome", {**profile}),
        ("cb3-01-zscore", "cb3-genome", {"standardize": "zscore", **profile}),
        ("cb3-01b-minmax", "cb3-genome", {"standardize": "minmax", **profile}),
    ]
    for tau in [0.1, 0.2]:
        out.append((f"cb3-02-threshold-tau{round(tau * 100):03d}", "cb3-genome",
                    {"standardize": "zscore", "threshold": tau, **profile}))
    for d in [32, 64, 128, 256]:
        out.append((f"cb3-03-svd{d:03d}", "cb3-genome",
                    {"standardize": "zscore", "svd_dim": d, **profile}))
    # 04 = nhánh đối chứng "bỏ SVD": bước 3 làm giảm metric nên phải đo item-KNN khi
    # không có SVD. Nhánh theo đúng luật "giữ tốt nhất bước trước" là 04d bên dưới.
    for k in KNN_K_GRID:
        out.append((f"cb3-04-item-knn-k{k:03d}", "cb3-genome",
                    {"standardize": "zscore", "scorer": "item_knn",
                     "weight_mode": "binary_pos", "k_neighbors": k}))
    # 04b giữ nhánh có SVD để bảng ablation so sánh được hai nhánh cạnh nhau
    out.append(("cb3-04b-item-knn-svd064-k050", "cb3-genome",
                {"standardize": "zscore", "svd_dim": 64, "scorer": "item_knn",
                 "weight_mode": "binary_pos", "k_neighbors": 50}))
    out.append(("cb3-04c-item-knn-k050-normsim", "cb3-genome",
                {"standardize": "zscore", "scorer": "item_knn",
                 "weight_mode": "binary_pos", "k_neighbors": 50, "normalize_sim": True}))
    # 04d là nhánh đúng luật "giữ tốt nhất bước trước": sau khi sửa thứ tự threshold,
    # X tốt nhất của bước 1-3 là zscore + τ=0.2 (0.0507) chứ không phải zscore trần (0.0447).
    for k in KNN_K_GRID:
        out.append((f"cb3-04d-item-knn-tau020-k{k:03d}", "cb3-genome",
                    {"standardize": "zscore", "threshold": 0.2, "scorer": "item_knn",
                     "weight_mode": "binary_pos", "k_neighbors": k}))
    for lam in [1.0, 10.0, 100.0]:
        out.append((f"cb3-05-ridge-lambda{int(lam):03d}", "cb3-ridge",
                    {"svd_dim": 64, "standardize": "zscore", "ridge_lambda": lam,
                     "min_train": 5, "weight_mode": "binary_pos"}))
    return out


def cb2_configs() -> list[tuple[str, str, dict]]:
    """CB-2 BM25: baseline tham số mặc định IR, rồi quét k1 x b.

    Dùng scorer=profile chứ không phải item_knn như spec phase-03 dự kiến: phase 2 đã
    chứng minh item-KNN chỉ là profile bị cắt top-k (k → I thì hai cái trùng nhau), nên
    quét tham số BM25 trên profile mới đo đúng tác động của chính trọng số BM25.
    """
    base = {"min_df": 3, "max_df": 0.5, "max_features": 20_000,
            "scorer": "profile", "weight_mode": "binary_pos"}
    out = [("cb2-00-default", "cb2-bm25", {**base, "k1": 1.2, "b": 0.75})]
    # k1=0.5 thắng ở cận dưới của grid gốc {0.5,1.2,2.0} nên nới xuống {0.1,0.3}.
    # k1 → 0 nghĩa là bão hoà tuyệt đối: w → idf, tức chỉ còn "tag có mặt hay không".
    # b=0 là cận tự nhiên của miền [0,1], không phải biên nhân tạo.
    for k1 in [0.1, 0.3, 0.5, 1.2, 2.0]:
        for b in [0.0, 0.5, 0.75, 1.0]:
            out.append((f"cb2-01-k1-{round(k1 * 10):02d}-b{round(b * 100):03d}", "cb2-bm25",
                        {**base, "k1": k1, "b": b}))
    return out
