"""Đánh giá top-K theo giao thức full ranking, xử lý theo lô user.

Full ranking: mọi item chưa xuất hiện trong train của user đều là ứng viên. Khác với
cách lấy mẫu 100 negative phổ biến trong các bài báo neural, full ranking cho thứ hạng
thuật toán nhất quán (Krichene & Rendle, 2020) và làm Precision/Recall@k có ý nghĩa.

Kết quả lưu tại results/:
    {exp}_metrics.csv   trung bình từng metric + siêu dữ liệu cấu hình
    {exp}_peruser.npz   metric per-user (phase 4 dùng cho Wilcoxon và bootstrap CI)
    {exp}_topk.npy      danh sách gợi ý (tuỳ chọn, phase 4 dùng cho coverage/novelty)
"""

import json
import time

import numpy as np
import pandas as pd

from src import paths
from src.data.build_interaction_matrices import Dataset
from src.eval import topk_ranking_metrics as metrics_mod
from src.models.base_recommender import BaseRecommender

BATCH_SIZE = 2000
K_MAX = 50


def rank_top_items(scores: np.ndarray, k: int) -> np.ndarray:
    """Lấy chỉ số k item điểm cao nhất, đã sắp theo thứ hạng giảm dần."""
    # argpartition O(n) rồi mới sort k phần tử, nhanh hơn nhiều so với argsort toàn bộ
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def rows_without_signal(scores: np.ndarray) -> np.ndarray:
    """Mặt nạ các hàng mà mô hình không phân biệt được item nào.

    Một hàng "không có tín hiệu" khi mọi điểm còn lại (sau khi mask lịch sử train) đều
    bằng nhau — kể cả trường hợp không còn điểm hữu hạn nào. Khi đó mô hình **không có
    thứ hạng nào**; danh sách mà `argpartition` trả về chỉ phản ánh thứ tự chỉ số của
    mảng, tức là do cách lưu trữ quyết định chứ không phải do thuật toán.

    Đây không phải chi tiết hình thức. Ở đây có 21/20.000 user không có tương tác
    relevant nào trong train, và mô hình nội dung cho họ điểm 0 trên toàn bộ item.
    Nếu không chặn, những user đó có cơ hội "trúng" item relevant chỉ vì item đó tình
    cờ nằm ở chỉ số thấp — điểm được ban cho không có lý do. Guard này biến kết quả
    từ "may mắn" thành "0 một cách có nguyên tắc".

    Người dùng không bị loại khỏi tập đánh giá: họ vẫn ở đó, với metric bằng 0. Bỏ họ
    đi sẽ là thao túng mẫu — và ở đây việc loại bỏ đúng là làm điểm trung bình tăng
    lên.
    """
    finite = np.isfinite(scores)
    n_finite = finite.sum(axis=1)
    max_finite = np.where(finite, scores, -np.inf).max(axis=1)
    min_finite = np.where(finite, scores, np.inf).min(axis=1)
    return (n_finite == 0) | (max_finite == min_finite)


def evaluate_model(
    model: BaseRecommender,
    dataset: Dataset,
    k_list=metrics_mod.DEFAULT_K_LIST,
    batch_size: int = BATCH_SIZE,
    save_topk: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Chấm điểm mô hình trên dataset.eval_users.

    Returns:
        (metric per-user, mảng topk [n_eval_users, k]).
    """
    eval_users = dataset.eval_users
    # Luôn giữ sẵn K_MAX item để phase 4 tính được coverage/novelty mà không chạy lại,
    # nhưng không vượt quá số item đang có (quan trọng cho dataset nhỏ trong test).
    k_max = min(max(K_MAX, max(k_list)), dataset.n_items)
    if max(k_list) > k_max:
        raise ValueError(f"k={max(k_list)} vượt quá số item ứng viên ({dataset.n_items})")
    topk_all = np.zeros((len(eval_users), k_max), dtype="int32")
    no_signal = np.zeros(len(eval_users), dtype=bool)

    for start in range(0, len(eval_users), batch_size):
        batch = eval_users[start : start + batch_size]
        raw_scores = model.score_users(batch)
        if raw_scores.shape != (len(batch), dataset.n_items):
            raise ValueError(
                f"{model.name}.score_users trả về {raw_scores.shape}, "
                f"cần {(len(batch), dataset.n_items)}"
            )
        # Bắt buộc sao chép: np.asarray trả về chính mảng gốc khi dtype đã khớp, mà bên
        # dưới ta ghi -inf vào đó. Không copy thì mô hình nào trả về bộ đệm dùng lại sẽ
        # bị nhiễm -inf vĩnh viễn và điểm giảm dần qua từng lô.
        scores = np.array(raw_scores, dtype="float32", copy=True)
        if np.isnan(scores).any():
            raise ValueError(
                f"{model.name}.score_users trả về NaN — điểm NaN sẽ làm hỏng thứ hạng "
                "và biến metric thành NaN mà không báo lỗi."
            )

        # Loại item user đã tương tác trong train: không được gợi ý lại
        seen = dataset.train.train_csr[batch]
        rows, cols = seen.nonzero()
        scores[rows, cols] = -np.inf
        no_signal[start : start + len(batch)] = rows_without_signal(scores)
        topk_all[start : start + len(batch)] = rank_top_items(scores, k_max)

    rel_csr = dataset.test_rel_csr[eval_users]
    per_user = metrics_mod.evaluate_ranking(topk_all, rel_csr, k_list)
    # User không có tín hiệu: điểm bằng 0 một cách có nguyên tắc, thay vì nhận điểm
    # may mắn do thứ tự chỉ số. Họ vẫn ở trong tập đánh giá — loại bỏ sẽ là thao túng mẫu.
    for name in per_user:
        per_user[name][no_signal] = 0.0

    if not save_topk:
        topk_all = topk_all[:, : max(k_list)]
    return per_user, topk_all


def run_and_save(
    model: BaseRecommender,
    dataset: Dataset,
    exp_name: str,
    item_features: dict,
    k_list=metrics_mod.DEFAULT_K_LIST,
    save_topk: bool = False,
    config: dict | None = None,
) -> dict[str, float]:
    """Fit + đánh giá + ghi kết quả. Trả về dict metric trung bình."""
    paths.ensure_dirs()

    fit_start = time.time()
    model.fit(dataset.train, item_features)
    fit_seconds = time.time() - fit_start

    score_start = time.time()
    per_user, topk_all = evaluate_model(model, dataset, k_list, save_topk=save_topk)
    score_seconds = time.time() - score_start

    summary = metrics_mod.summarize(per_user)
    summary["fit_s"] = round(fit_seconds, 2)
    summary["score_s"] = round(score_seconds, 2)
    summary["n_eval_users"] = len(dataset.eval_users)

    # Lưu kèm cấu hình: bảng kết quả ở phase 5 phải truy được tham số đã sinh ra nó
    record = {"exp_name": exp_name, **summary}
    if config is not None:
        record["model"] = config.get("model", model.name)
        record["params"] = json.dumps(config.get("params", {}), ensure_ascii=False)
        record["users"] = config.get("users", "")
        # Định danh giao thức: bằng chứng rằng các dòng trong một bảng so sánh thực sự
        # chạy trên cùng tập ứng viên và cùng split. Không có nó thì tiền tố trong tên
        # file là dấu vết duy nhất — và đổi tên file là mất dấu vết.
        record["split_id"] = config.get("split_id", "")
        record["catalog_id"] = config.get("catalog_id", "")
        record["seed"] = config.get("seed", paths.SEED)
    pd.DataFrame([record]).to_csv(paths.RESULTS_DIR / f"{exp_name}_metrics.csv", index=False)

    np.savez_compressed(
        paths.RESULTS_DIR / f"{exp_name}_peruser.npz",
        eval_users=dataset.eval_users,
        **per_user,
    )
    if save_topk:
        np.save(paths.RESULTS_DIR / f"{exp_name}_topk.npy", topk_all)

    return summary
