"""Giao thức đánh giá thay thế: leave-one-out + 100 negative lấy mẫu (He et al. 2017).

Mục đích **không** phải để có thêm một bảng số. Mục đích là kiểm chứng một nhận định
phương pháp luận: Krichene & Rendle (2020) chỉ ra rằng chấm điểm trên 101 ứng viên cho
thứ hạng thuật toán **không nhất quán** với xếp hạng đầy đủ, nên nhiều kết quả trong các
bài báo neural không so sánh được với nhau. Dự án này chọn full ranking ngay từ đầu; chạy
thêm giao thức kia rồi đo Kendall τ giữa hai bảng thứ hạng là cách biến lựa chọn đó từ
một câu trích dẫn thành một kết quả thực nghiệm của chính đồ án.

Giao thức: mỗi user lấy 1 item relevant **cuối cùng theo thời gian** trong test làm dương,
100 item user chưa từng tương tác làm âm, chấm điểm trên đúng 101 ứng viên đó, tính
HR@10 và NDCG@10 theo vị trí của item dương.

Chạy: python -m src.eval.sampled_loo_evaluator configs/best-cf-als.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import paths
from src.data.build_interaction_matrices import Dataset, load_dataset, load_item_features
from src.eval.model_registry import build_model
from src.models.base_recommender import BaseRecommender

N_NEGATIVES = 100
K = 10
BATCH = 2000


def build_loo_candidates(dataset: Dataset, n_negatives: int = N_NEGATIVES,
                         seed: int = paths.SEED,
                         test_parquet: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (users, candidates [n_users, 1 + n_negatives]); cột 0 là item dương.

    Item dương = item relevant **mới nhất** trong test của user. Negative lấy trong số item
    user chưa tương tác ở cả train lẫn test — nếu bốc trúng item user đã thích nhưng nằm
    trong test thì nó bị gán nhãn âm sai.

    `test_parquet` phải khớp với split đã dựng nên `dataset`; mặc định là split chuẩn.
    """
    test = pd.read_parquet(test_parquet or paths.TEST_PARQUET)
    relevant = test[test["rating"] >= paths.RELEVANCE_THRESHOLD]
    # Sắp theo (user, ts, item) rồi lấy dòng cuối mỗi user: item_idx phá thế hoà timestamp
    # đúng như split_temporal_per_user, nên "item cuối cùng" là xác định, không phụ thuộc
    # thứ tự đọc file.
    relevant = relevant.sort_values(["user_idx", "ts", "item_idx"], kind="mergesort")
    last = relevant.groupby("user_idx", sort=True).tail(1)

    eval_set = set(dataset.eval_users.tolist())
    last = last[last["user_idx"].isin(eval_set)]
    users = last["user_idx"].to_numpy().astype("int32")
    positives = last["item_idx"].to_numpy().astype("int32")

    train_csr = dataset.train.train_csr
    test_csr = dataset.test_rel_csr
    rng = np.random.default_rng(seed)
    n_items = dataset.n_items

    candidates = np.empty((len(users), 1 + n_negatives), dtype="int32")
    candidates[:, 0] = positives
    for i, u in enumerate(users):
        seen = np.union1d(train_csr.getrow(u).indices, test_csr.getrow(u).indices)
        # Lấy dư rồi lọc: nhanh hơn nhiều so với dựng danh sách bù dài 13.816 mỗi user
        pool = rng.integers(0, n_items, size=n_negatives * 3)
        pool = pool[~np.isin(pool, seen)]
        pool = pd.unique(pool)[:n_negatives]
        if len(pool) < n_negatives:   # cực hiếm, chỉ với user có lịch sử rất dài
            remaining = np.setdiff1d(np.arange(n_items), np.union1d(seen, pool))
            pool = np.concatenate([pool, rng.choice(remaining, n_negatives - len(pool),
                                                    replace=False)])
        candidates[i, 1:] = pool
    return users, candidates


def evaluate_sampled(model: BaseRecommender, dataset: Dataset, users: np.ndarray,
                     candidates: np.ndarray, k: int = K) -> dict[str, float]:
    """HR@k và NDCG@k của item dương trong 101 ứng viên."""
    if candidates.max() >= dataset.n_items:
        raise ValueError(f"candidates chứa item {candidates.max()} ngoài {dataset.n_items} item")
    hits, ndcgs = [], []
    for start in range(0, len(users), BATCH):
        batch_users = users[start: start + BATCH]
        batch_cand = candidates[start: start + BATCH]
        scores = np.take_along_axis(model.score_users(batch_users), batch_cand, axis=1)
        # Hạng của item dương, phá hoà **bi quan**: mọi ứng viên có điểm bằng đều tính là
        # đứng trên. Nếu chỉ đếm strict-greater thì mô hình cho điểm hằng số — chẳng hạn
        # với user không có tương tác relevant nào trong train — sẽ đạt HR@10 = 1,0 dù
        # không hề xếp hạng được gì.
        # Lưu ý khi so với tài liệu: He et al. (2017) và các bài kế thừa dùng strict-greater.
        # Full ranking ở đây phá hoà theo chỉ số item, tức xấp xỉ bi quan ở quy mô 13.816
        # ứng viên (xác suất trúng ~0) nhưng không đồng nhất về mặt định nghĩa. Vì vậy
        # Kendall τ giữa hai giao thức đo lẫn một phần khác biệt về quy ước phá hoà.
        positive = scores[:, :1]
        rank = (scores > positive).sum(axis=1) + (scores == positive).sum(axis=1) - 1
        hits.append(rank < k)
        ndcgs.append(np.where(rank < k, 1.0 / np.log2(rank + 2.0), 0.0))
    return {f"HR@{k}": float(np.concatenate(hits).mean()),
            f"NDCG@{k}": float(np.concatenate(ndcgs).mean()),
            "n_users": len(users)}


def prepare(users: str = "dev") -> tuple[Dataset, np.ndarray, np.ndarray]:
    """Nạp dataset và dựng bộ ứng viên **một lần** cho mọi thuật toán.

    Bộ ứng viên phải giống hệt nhau giữa các thuật toán, nếu không phép so sánh vô nghĩa;
    dựng lại mỗi lần vừa tốn ~36s vừa chỉ đúng nhờ seed cố định.
    """
    dataset = load_dataset(users=users)
    loo_users, candidates = build_loo_candidates(dataset)
    return dataset, loo_users, candidates


def run_config(config_path: Path, users: str = "dev",
               context: tuple[Dataset, np.ndarray, np.ndarray] | None = None) -> dict[str, float]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset, loo_users, candidates = context if context is not None else prepare(users)

    model = build_model(config["model"], config.get("params"))
    model.fit(dataset.train, load_item_features())
    result = evaluate_sampled(model, dataset, loo_users, candidates)
    return {"exp_name": config["exp_name"], "model": config["model"], **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--users", choices=["dev", "full"], default="dev")
    args = parser.parse_args()
    result = run_config(args.config, args.users)
    print(result)


if __name__ == "__main__":
    main()
