"""CB-3: Mô hình nội dung dựa trên Tag Genome (1128 chiều).

Các bước tối ưu hoá (đo bằng ablation):
    1. Raw genome L2-norm (baseline)
    2. Threshold relevance < τ → 0 (sparsify tag yếu)
    3. Chuẩn hoá per tag: z-score hoặc min-max
    4. TruncatedSVD giảm chiều (32, 64, 128, 256) — cache .npy

**Thứ tự bắt buộc: threshold → standardize → SVD → L2-norm.** Threshold phải áp lên
*relevance thô* mới đúng định nghĩa "tag yếu" của spec. Nếu chạy sau z-score thì τ=0.1
không còn là ngưỡng relevance: đo trên dữ liệu thật, nó xoá 76.7% ô thay vì 66.2%, và
15.1% tổng số ô bị xoá oan dù relevance thô ≥ 0.1. Tệ hơn, sau z-score giá trị 0 nghĩa
là "đúng bằng trung bình của tag đó" chứ không phải "không liên quan", nên clamp về 0
là dịch item về mức trung bình chứ không phải làm thưa.

Kết quả: ma trận X [I, D] dense L2-normalised, chuyển cho profile hoặc item-knn scorer.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD

from src import paths
from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.content_item_knn_scorer import ContentItemKnnScorer
from src.models.content_profile_scorer import ContentProfileScorer
from src.models.feature_cache_key import cache_key, matrix_fingerprint
from src.models.vector_normalization import safe_l2_normalize

EPS = 1e-12
STANDARDIZE_MODES = (None, "zscore", "minmax")


def build_genome_item_matrix(
    item_features: dict,
    standardize: str | None = None,
    threshold: float | None = None,
    svd_dim: int | None = None,
) -> np.ndarray:
    """Dựng X [I, D] từ genome. Public để genome_ridge_user_model tái dùng (DRY).

    Chuẩn hoá per-tag fit trên toàn bộ item — không leakage vì không đụng tới rating.
    """
    if standardize not in STANDARDIZE_MODES:
        raise ValueError(f"standardize phải thuộc {STANDARDIZE_MODES}, nhận '{standardize}'")
    if "genome" not in item_features:
        raise KeyError("Thiếu đặc trưng 'genome'. Chạy: python -m src.data.build_item_content_features")

    # PHẢI copy: np.asarray(memmap) CHIA SẺ bộ nhớ với file, threshold in-place bên dưới
    # sẽ ghi đè thẳng vào data/processed/item_genome.npy.
    X = np.array(item_features["genome"], dtype="float32", copy=True)

    if threshold is not None:
        X[X < threshold] = 0.0

    if standardize == "zscore":
        X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), EPS)
    elif standardize == "minmax":
        xmin, xmax = X.min(axis=0), X.max(axis=0)
        X = (X - xmin) / np.maximum(xmax - xmin, EPS)

    if svd_dim is not None:
        X = _apply_svd(X, svd_dim, standardize, threshold)

    return safe_l2_normalize(X)


def _apply_svd(X: np.ndarray, d: int, standardize: str | None, threshold: float | None) -> np.ndarray:
    """TruncatedSVD, cache kết quả cho lần chạy sau.

    Khoá gồm cả vân tay của X đầu vào, không chỉ tham số: nếu code dựng X đổi mà tham
    số không đổi thì cache theo tham số vẫn khớp shape và bị nạp nhầm.
    """
    key = cache_key("genome_svd", {"d": d, "standardize": standardize,
                                   "threshold": threshold, "x": matrix_fingerprint(X)})
    cache = paths.PROCESSED_DIR / f"item_{key}.npy"
    if cache.exists():
        cached = np.load(cache).astype("float32")
        if cached.shape == (X.shape[0], d):
            print(f"  [Genome] Nạp SVD cache: {cache.name}")
            return cached

    svd = TruncatedSVD(n_components=d, random_state=paths.SEED)
    Z = svd.fit_transform(X).astype("float32")
    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, Z)
    print(f"  [Genome] SVD d={d}, explained variance={svd.explained_variance_ratio_.sum():.3f}"
          f", saved {cache.name}")
    return Z


class GenomeContentModel(BaseRecommender):
    """Wrapper: build X từ genome rồi uỷ quyền cho scorer."""

    name = "cb3-genome"

    def __init__(
        self,
        standardize: str | None = None,    # None, "zscore", "minmax"
        threshold: float | None = None,     # τ: relevance thô < τ → 0
        svd_dim: int | None = None,         # None = giữ nguyên 1128 chiều
        scorer: str = "profile",
        weight_mode: str = "binary_pos",
        k_neighbors: int = 50,
        normalize_sim: bool = False,
        half_life_days: float | None = None,
    ):
        self.standardize = standardize
        self.threshold = threshold
        self.svd_dim = svd_dim
        self.scorer_type = scorer
        self.weight_mode = weight_mode
        self.k_neighbors = k_neighbors
        self.normalize_sim = normalize_sim
        self.half_life_days = half_life_days
        self._scorer: BaseRecommender | None = None

    def fit(self, train: TrainData, item_features: dict) -> "GenomeContentModel":
        X = build_genome_item_matrix(
            item_features, self.standardize, self.threshold, self.svd_dim
        )
        features_with_X = {**item_features, "X": X}

        if self.scorer_type == "profile":
            self._scorer = ContentProfileScorer(
                weight_mode=self.weight_mode, half_life_days=self.half_life_days)
        elif self.scorer_type == "item_knn":
            self._scorer = ContentItemKnnScorer(
                k_neighbors=self.k_neighbors,
                weight_mode=self.weight_mode,
                sim_name=self._sim_cache_name(),
                normalize_sim=self.normalize_sim,
                half_life_days=self.half_life_days,
            )
        else:
            raise ValueError(f"scorer phải là 'profile' hoặc 'item_knn', nhận '{self.scorer_type}'")
        self._scorer.fit(train, features_with_X)
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return self._scorer.score_users(user_idx)

    def _sim_cache_name(self) -> str:
        return cache_key("genome", {
            "standardize": self.standardize,
            "threshold": self.threshold,
            "svd_dim": self.svd_dim,
        })
