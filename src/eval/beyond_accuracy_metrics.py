"""Các chỉ số ngoài độ chính xác: coverage, novelty, intra-list diversity.

Ba chỉ số này trả lời câu hỏi mà NDCG không trả lời được: hệ thống gợi ý **bao nhiêu**
phim khác nhau, phim được gợi ý có **hiếm** không, và một danh sách 10 phim có **đa dạng**
không. Đây là nơi lọc nội dung thường thắng lọc cộng tác, nên chúng là một nửa luận điểm
của đề tài (nửa kia là cold-start).

    coverage@k = |hợp của mọi danh sách top-k| / tổng số item — *bao nhiêu* phim được chạm
    gini@k     = mức tập trung của tải gợi ý, 0 = trải đều, 1 = dồn một phim
    novelty@k  = trung bình −log2(p_i) làm trơn add-one — phim được gợi ý *hiếm* đến đâu
    ILD@k      = trung bình (1 − cosine) trên mọi cặp trong cùng một danh sách

Coverage và Gini bù nhau: coverage đếm phim khác nhau nhưng mù với mức tập trung, còn
Gini đo đúng mức tập trung đó. Cần cả hai mới mô tả đủ hành vi phân bổ của một mô hình.

Không gian tính ILD là genome đã z-score (1128 chiều): genome thô có cosine trung bình
0,63 giữa hai phim bất kỳ nên gần như không phân biệt được danh sách đa dạng với danh sách
đơn điệu. Đây là cùng biểu diễn mà CB-3 dùng, không phải một lựa chọn riêng cho chỉ số này.

**Loại user không có tín hiệu.** Một số user không có tương tác relevant nào trong train
nên mô hình nội dung với trọng số binary_pos cho điểm 0 trên toàn bộ item; `argpartition`
khi đó trả về item theo thứ tự chỉ số. Danh sách ấy do cách lưu trữ quyết định chứ không
do thuật toán, và nếu giữ lại sẽ làm cả ba chỉ số ở đây sai lệch theo cách không phản ánh
hành vi thật của mô hình.

Chạy: python -m src.eval.beyond_accuracy_metrics [--prefix full_] [--k 10]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths
from src.data.build_interaction_matrices import load_dataset, load_item_features
from src.models.genome_content_model import build_genome_item_matrix

EPS = 1e-12


def users_without_signal(dataset) -> np.ndarray:
    """Mặt nạ user không có tương tác relevant nào trong train (theo thứ tự eval_users)."""
    train_csr = dataset.train.train_csr[dataset.eval_users]
    n_relevant = np.asarray(
        (train_csr >= paths.RELEVANCE_THRESHOLD).sum(axis=1)).ravel()
    return n_relevant == 0


def coverage(topk: np.ndarray, n_items: int) -> float:
    return float(len(np.unique(topk)) / n_items)


def gini(topk: np.ndarray, n_items: int) -> float:
    """Mức **tập trung** của tải gợi ý trên toàn catalog. 0 = trải đều, 1 = dồn một phim.

    Bù đúng chỗ coverage mù: coverage chỉ đếm phim *khác nhau* nên một phim được gợi ý
    một lần tính ngang một phim được gợi ý một triệu lần. Một mô hình hoàn toàn có thể
    phủ 50% catalog mà vẫn dồn 99% số suất cho mười phim.

    Ví dụ thực tế trong đề tài: ALS chỉ chạm 15% catalog (coverage 0,149) so với 47% của
    CB-1, nhưng Gini của ALS lại **thấp hơn** (0,973 vs 0,983) — CB-1 dồn 81% số suất vào
    100 phim, ALS chỉ 47%. Coverage cao của CB-1 đến từ một đuôi dài phim xuất hiện lác đác.

    Đếm trên **toàn bộ** n_items, kể cả phim chưa bao giờ được gợi ý (count = 0): chính
    khối lượng "không bao giờ xuất hiện" là thứ cần đo. Nếu chỉ tính trên các phim đã
    xuất hiện thì Gini sẽ khen một mô hình chỉ biết mười phim nhưng chia đều cho chúng.
    """
    counts = np.bincount(np.asarray(topk).ravel(), minlength=n_items).astype("float64")
    # bincount tự nới mảng khi gặp id ≥ n_items → Gini tính trên catalog sai cỡ, không báo lỗi
    if counts.size != n_items:
        raise ValueError(f"topk chứa item id ≥ n_items ({n_items})")
    total = counts.sum()
    if total <= 0:
        return 0.0
    counts.sort()
    n = len(counts)
    rank = np.arange(1, n + 1, dtype="float64")
    return float((2.0 * np.sum(rank * counts)) / (n * total) - (n + 1.0) / n)


def top_share(topk: np.ndarray, n: int) -> float:
    """Tỷ lệ số suất gợi ý rơi vào n phim được gợi ý nhiều nhất — đọc trực tiếp hơn Gini."""
    counts = np.sort(np.bincount(np.asarray(topk).ravel()))[::-1]
    return float(counts[:n].sum() / counts.sum()) if counts.sum() else 0.0


def novelty(topk: np.ndarray, item_popularity: np.ndarray, n_users: int) -> float:
    """Trung bình self-information −log2(p) của các item được gợi ý, làm trơn add-one.

    Add-one chứ không kẹp bằng EPS. Ở giao thức cold-start **mọi** phim held-out có
    popularity đúng bằng 0, nên kẹp 1e-12 gán cho chúng −log2(1e-12) = 39,9 bit — một hằng
    số tuỳ tiện từng chiếm tới 77% giá trị novelty báo cáo của CB-2. Khi đó cột novelty chỉ
    nói lại điều coverage đã nói ("CB gợi ý phim held-out, CF thì không") trên một thang
    không diễn giải được.

    Với (pop+1)/(n_user+1), trần của novelty là log2(n_user+1) ≈ 17,3 bit và mọi giá trị
    nằm trên cùng một thang "xác suất một user bất kỳ đã thích phim này".
    """
    prob = (item_popularity + 1.0) / (n_users + 1.0)
    return float(np.mean(-np.log2(prob)[topk]))


def frac_unseen_items(topk: np.ndarray, item_popularity: np.ndarray) -> float:
    """Tỷ lệ item được gợi ý chưa hề có tương tác train.

    Báo cáo thẳng thay vì để nó ẩn trong novelty — đây mới là đại lượng mà cold-start
    quan tâm, và nó đọc được ngay mà không cần quy đổi qua log.
    """
    return float(np.mean(item_popularity[topk] == 0))


def intra_list_diversity(topk: np.ndarray, embedding: np.ndarray) -> float:
    """Trung bình (1 − cosine) trên mọi cặp trong cùng danh sách.

    embedding đã L2-norm theo hàng nên cosine = tích vô hướng. Tính theo lô để không dựng
    ma trận [n_users, k, k] cho 158k user cùng lúc.
    """
    k = topk.shape[1]
    if k < 2:
        return 0.0
    n_pairs = k * (k - 1) / 2
    total, batch = 0.0, 4000
    for start in range(0, len(topk), batch):
        vecs = embedding[topk[start: start + batch]]          # [b, k, d]
        sims = np.einsum("bkd,bjd->bkj", vecs, vecs)          # [b, k, k]
        # Chỉ lấy tam giác trên: đường chéo là self-sim, phần dưới lặp lại phần trên
        pair_sum = (sims.sum(axis=(1, 2)) - np.trace(sims, axis1=1, axis2=2)) / 2
        total += float(np.sum(n_pairs - pair_sum))
    return total / (len(topk) * n_pairs)


def compute_all(prefix: str = "full_", k: int = 10,
                results_dir: Path | None = None) -> pd.DataFrame:
    rdir = results_dir or paths.RESULTS_DIR
    topk_files = sorted(rdir.glob(f"{prefix}*_topk.npy"))
    if not topk_files:
        raise SystemExit(f"Không tìm thấy {prefix}*_topk.npy trong {rdir}. "
                         "Chạy scripts/run_final_evaluations.py trước.")

    users = "full" if prefix.endswith("full_") else "dev"
    split_dir = paths.PROCESSED_DIR / "coldstart" if prefix.startswith("coldstart") else None
    dataset = load_dataset(users=users, split_dir=split_dir)
    embedding = build_genome_item_matrix(load_item_features(), standardize="zscore")
    no_signal = users_without_signal(dataset)
    print(f"  loại {int(no_signal.sum()):,}/{len(no_signal):,} user không có tín hiệu train")

    pop = dataset.train.item_popularity
    rows = []
    for path in topk_files:
        raw = np.load(path)
        if raw.shape[0] != len(no_signal):
            raise SystemExit(f"{path.name} có {raw.shape[0]} hàng nhưng dataset có {len(no_signal)}")
        if raw.shape[1] < k:
            raise SystemExit(f"{path.name} chỉ có {raw.shape[1]} cột, cần ít nhất {k}")
        topk = raw[~no_signal, :k]
        if len(topk) == 0:
            raise SystemExit("Mọi user đều bị loại — không tính được chỉ số nào")
        rows.append({
            "exp_name": path.name.removesuffix("_topk.npy"),
            f"coverage@{k}": coverage(topk, dataset.n_items),
            f"gini@{k}": gini(topk, dataset.n_items),
            f"top10_share@{k}": top_share(topk, 10),
            f"top100_share@{k}": top_share(topk, 100),
            f"novelty@{k}": novelty(topk, pop, dataset.train.n_users),
            f"ILD@{k}": intra_list_diversity(topk, embedding),
            f"frac_unseen@{k}": frac_unseen_items(topk, pop),
            "n_users": int(len(topk)),
            "n_excluded": int(no_signal.sum()),
        })
    return pd.DataFrame(rows).sort_values(f"coverage@{k}", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="full_", help="Tiền tố thí nghiệm (full_ / coldstart_)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default=None, help="Lưu CSV (mặc định results/beyond-accuracy.csv)")
    args = parser.parse_args()

    df = compute_all(args.prefix, args.k)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out = Path(args.output) if args.output else paths.RESULTS_DIR / "beyond-accuracy.csv"
    df.to_csv(out, index=False)
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()
