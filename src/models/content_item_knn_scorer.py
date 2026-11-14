"""Item-KNN scorer trên không gian nội dung: K = topk(X @ Xᵀ); S = W @ Kᵀ.

Khác với profile scorer, KNN scorer không trung bình hoá item thành một profile duy nhất.
Thay vào đó mỗi item ứng viên j được chấm bằng tổng độ tương đồng giữa j và các item
user đã xem, **chỉ tính trên k hàng xóm gần nhất của chính j**:

    S[u, j] = Σ_{i ∈ N(j)} W[u, i] · K[j, i]

Hướng nhân là ``W @ Kᵀ`` chứ không phải ``W @ K``: K bất đối xứng vì top-k được cắt
theo hàng (hàng j = k hàng xóm của j). ``W @ K`` sẽ cho "j nằm trong danh sách hàng
xóm của item user đã xem" — một phép lan truyền khác hẳn, và đo trên dev cho NDCG@10
thấp hơn 16% (CB-1) / 8% (CB-3).

Ma trận tương đồng K dense ~760MB (float32, 13816²) nên tính theo block rồi chỉ giữ
top-k mỗi hàng dưới dạng sparse. Cache .npz theo khoá tham số + k để không tính lại.
"""

from pathlib import Path

import numpy as np
from scipy import sparse

from src import paths
from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender
from src.models.feature_cache_key import matrix_fingerprint
from src.models.user_history_weights import build_user_weights

BLOCK = 2000  # tính sim 2000 item × 13816 item mỗi lô
EPS = 1e-12


class ContentItemKnnScorer(BaseRecommender):
    """Item-KNN content scorer dùng chung cho mọi item representation X."""

    name = "content-item-knn"

    def __init__(self, sim_name: str, k_neighbors: int = 50,
                 weight_mode: str = "binary_pos", normalize_sim: bool = False,
                 half_life_days: float | None = None):
        # sim_name bắt buộc, không có mặc định: một giá trị mặc định kiểu "generic" sẽ
        # khiến mọi lần khởi tạo trực tiếp (notebook, hybrid ở phase 3) dùng chung một
        # file cache cho những X hoàn toàn khác nhau.
        self.k_neighbors = k_neighbors
        self.weight_mode = weight_mode
        self.sim_name = sim_name
        self.half_life_days = half_life_days
        # normalize_sim: chia cho Σ|sim| của các hàng xóm user đã xem (ItemKNN chuẩn).
        # Tắt mặc định để giữ đúng công thức spec phase-02; bật được để đo bằng ablation.
        # Lưu ý: điểm chỉ nằm trong [−1, 1] khi trọng số user không âm (binary_pos). Với
        # weight_mode="rating" mẫu số không bao hàm độ lớn rating nên cận đó không còn đúng.
        self.normalize_sim = normalize_sim
        self._Kt: sparse.csr_matrix | None = None      # [I, I] = Kᵀ đã dựng sẵn
        self._Kt_abs: sparse.csr_matrix | None = None  # |Kᵀ|, chỉ khi normalize_sim
        self._W: sparse.csr_matrix | None = None       # [U, I] sparse weights

    def fit(self, train: TrainData, item_features: dict) -> "ContentItemKnnScorer":
        K = _load_or_build_sim(item_features["X"], self.k_neighbors, self.sim_name)
        # Chuyển vị một lần tại fit thay vì mỗi batch: K.T trên CSR trả về CSC,
        # nhân sparse-sparse với CSC ở vế phải chậm hơn nhiều so với CSR.
        self._Kt = K.T.tocsr()
        if self.normalize_sim:
            self._Kt_abs = abs(self._Kt)
        self._W = build_user_weights(train, self.weight_mode, self.half_life_days)
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        """S = W[batch] @ Kᵀ → [batch_size, n_items]."""
        W_batch = self._W[user_idx]                      # sparse [B, I]
        scores = (W_batch @ self._Kt).toarray().astype("float32")

        if self.normalize_sim:
            # Mẫu số = tổng |sim| của các hàng xóm mà user đã tương tác
            seen = W_batch.copy()
            seen.data = np.ones_like(seen.data)
            denom = (seen @ self._Kt_abs).toarray().astype("float32")
            scores /= np.maximum(denom, EPS)
        return scores


def _cache_path(sim_name: str, k: int, xfp: str) -> Path:
    return paths.PROCESSED_DIR / f"sim_{sim_name}_{xfp}_k{k}.npz"


def _load_or_build_sim(X, k: int, sim_name: str) -> sparse.csr_matrix:
    """Nạp sim cache nếu có, ngược lại tính block-wise và lưu.

    Khoá cache gồm cả vân tay nội dung X: kiểm tra shape thôi là không đủ, hai X khác
    nhau hoàn toàn vẫn có cùng [I, I].
    """
    cache = _cache_path(sim_name, k, matrix_fingerprint(X))
    if cache.exists():
        K = sparse.load_npz(cache)
        if K.shape == (X.shape[0], X.shape[0]):
            print(f"  [KNN] Nạp sim cache: {cache.name}")
            return K
    K = _build_topk_sim_blockwise(X, k)
    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(cache, K)
    print(f"  [KNN] Lưu sim cache: {cache.name} ({K.nnz:,} nnz)")
    return K


def _build_topk_sim_blockwise(X, k: int) -> sparse.csr_matrix:
    """Tính cosine sim block-wise, giữ top-k mỗi hàng (loại self-sim).

    X đã L2-normalised nên X @ Xᵀ chính là cosine. X có thể sparse hoặc dense; chỉ
    **kết quả từng block** [BLOCK, I] mới được densify (110MB), không bao giờ densify X.
    """
    n_items = X.shape[0]
    is_sparse = sparse.issparse(X)
    Xt = X.T.tocsc() if is_sparse else X.T
    k_actual = min(k + 1, n_items)  # +1 vì self-sim sẽ bị loại

    rows_list, cols_list, vals_list = [], [], []
    for start in range(0, n_items, BLOCK):
        end = min(start + BLOCK, n_items)
        block = X[start:end] @ Xt
        sim_block = block.toarray() if sparse.issparse(block) else np.asarray(block)
        sim_block = sim_block.astype("float32", copy=False)

        # Loại self-similarity: gán -2 (thấp hơn cận dưới của cosine là -1) để chắc
        # chắn không lọt vào top-k kể cả khi mọi cosine khác đều bằng -1.
        sim_block[np.arange(end - start), np.arange(start, end)] = -2.0

        # argpartition đảm bảo k_actual phần tử đầu là k_actual giá trị lớn nhất
        topk_idx = np.argpartition(-sim_block, kth=k_actual - 1, axis=1)[:, :k_actual]
        topk_vals = np.take_along_axis(sim_block, topk_idx, axis=1)

        # Sắp giảm dần rồi cắt đúng k: bảo đảm bỏ đúng phần tử nhỏ nhất trong k+1
        order = np.argsort(-topk_vals, axis=1, kind="stable")
        topk_idx = np.take_along_axis(topk_idx, order, axis=1)[:, :k]
        topk_vals = np.take_along_axis(topk_vals, order, axis=1)[:, :k]

        # Khi k ≥ n_items − 1 thì self-sim (−2) vẫn còn sót lại sau khi cắt → lọc nốt
        rows = np.repeat(np.arange(start, end, dtype="int32"), topk_idx.shape[1])
        keep = topk_vals.ravel() > -1.5
        rows_list.append(rows[keep])
        cols_list.append(topk_idx.ravel().astype("int32")[keep])
        vals_list.append(topk_vals.ravel()[keep])

        if start % (BLOCK * 5) == 0:
            print(f"  [KNN] block {start}/{n_items}")

    K = sparse.csr_matrix(
        (np.concatenate(vals_list).astype("float32"),
         (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(n_items, n_items), dtype="float32",
    )
    K.sort_indices()
    return K
