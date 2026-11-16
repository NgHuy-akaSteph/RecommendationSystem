"""Lưu / nạp mô hình đã fit ra đĩa để demo khởi động nhanh.

Vấn đề: `fit` trên 158.774 user mất ~15s (ALS) và `load_dataset` mất ~18s. Nếu demo
Streamlit fit lại mỗi lần mở trang thì người xem chờ gần nửa phút — không chấp nhận được.

Cách giải quyết: fit **một lần** ra artifact, rồi demo chỉ nạp. Artifact gồm:

    <out_dir>/<model_key>/
        manifest.json      thông tin mô hình, tham số, split, kích thước
        factors.npz        user_factors + item_factors (mô hình CF)
        state.pkl          trạng thái scorer của mô hình nội dung (BM25/CB-4)

Vì sao không lưu chung một kiểu artifact cho mọi mô hình: CF (ALS) chỉ cần hai
ma trận factor, lưu `.npz` là xong và nạp cực nhanh; còn mô hình nội dung giữ một
ma trận đặc trưng lớn (X, ma trận trọng số user) mà `pickle` xử lý tự nhiên hơn nhiều.

Ràng buộc: module này **không** import `src.eval.*` — evaluator là công cụ offline
(chấm điểm trên nhãn test), còn tầng serving chỉ biết `TrainData` và `score_users`.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.build_interaction_matrices import TrainData
from src.models.base_recommender import BaseRecommender

MANIFEST_NAME = "manifest.json"
FACTORS_NAME = "factors.npz"
STATE_NAME = "state.pkl"


@dataclass
class ModelArtifact:
    """Mô hình đã fit + metadata, nạp từ đĩa."""

    model_key: str
    scorer: BaseRecommender
    manifest: dict

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        """Cùng hợp đồng với `BaseRecommender.score_users`."""
        return self.scorer.score_users(user_idx)


def model_dir(out_dir: Path, model_key: str) -> Path:
    return Path(out_dir) / model_key


def save_artifact(
    model_key: str,
    model: BaseRecommender,
    out_dir: Path,
    manifest_extra: dict | None = None,
) -> Path:
    """Ghi mô hình đã fit xuống `out_dir/<model_key>/`.

    Args:
        model_key: khoá định danh mô hình (tên trong `model_registry`).
        model: mô hình đã gọi `fit`.
        out_dir: thư mục đích.
        manifest_extra: thông tin bổ sung ghi vào manifest (split, n_users, ...).

    Returns:
        Đường dẫn thư mục vừa ghi.
    """
    target = model_dir(out_dir, model_key)
    target.mkdir(parents=True, exist_ok=True)

    user_factors = getattr(model, "_user_factors", None)
    item_factors = getattr(model, "_item_factors", None)
    if user_factors is not None and item_factors is not None:
        np.savez_compressed(target / FACTORS_NAME, user_factors=user_factors,
                            item_factors=item_factors)
        state = None
    else:
        # Mô hình nội dung: scorer giữ ma trận đặc trưng và trọng số user.
        state = _extract_content_state(model)

    if state is not None:
        with (target / STATE_NAME).open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "model_key": model_key,
        "model_class": type(model).__name__,
        "scorer_class": type(state).__name__ if state is not None else type(model).__name__,
        "has_factors": state is None,
        **(manifest_extra or {}),
    }
    (target / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _extract_content_state(model: BaseRecommender) -> object | None:
    """Lấy phần trạng thái đủ để chấm điểm của mô hình nội dung.

    Mô hình nội dung bọc một `scorer` con (`ContentProfileScorer` /
    `ContentItemKnnScorer`). Scorer giữ `X` (ma trận đặc trưng) và `W` (trọng số
    user) — đủ để tính điểm mà không cần train nữa.

    Hybrid (CB-4) là trường hợp đặc biệt: nó giữ **nhiều** scorer con và tự chấm
    điểm, nên lưu chính đối tượng hybrid (pickle xử lý được danh sách scorer con).

    Trả về `None` khi không tìm thấy cấu trúc nào quen thuộc, để `save_artifact` biết
    phải dùng đường lưu factor thay thế.
    """
    scorer = getattr(model, "_scorer", None)
    if scorer is not None:
        return scorer

    if getattr(model, "_models", None):
        return model

    return None


def load_artifact(out_dir: Path, model_key: str) -> ModelArtifact:
    """Nạp artifact đã lưu. Ném `FileNotFoundError` nếu chưa sinh."""
    target = model_dir(out_dir, model_key)
    manifest_path = target / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Thiếu artifact cho '{model_key}' tại {target}. "
            f"Chạy: venv/bin/python scripts/save_demo_artifacts.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    factors_path = target / FACTORS_NAME
    state_path = target / STATE_NAME
    if factors_path.exists():
        scorer = _FactorsScorer(np.load(factors_path))
    elif state_path.exists():
        with state_path.open("rb") as fh:
            scorer = pickle.load(fh)
    else:
        raise FileNotFoundError(
            f"Artifact '{model_key}' không có cả {FACTORS_NAME} lẫn {STATE_NAME} — file bị hỏng."
        )

    return ModelArtifact(model_key=model_key, scorer=scorer, manifest=manifest)


class _FactorsScorer:
    """Bọc hai ma trận factor đã lưu thành đúng hợp đồng `score_users`."""

    def __init__(self, npz):
        self.user_factors = npz["user_factors"]
        self.item_factors = npz["item_factors"]

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return (self.user_factors[user_idx] @ self.item_factors.T).astype("float32")


def artifact_exists(out_dir: Path, model_key: str) -> bool:
    """Kiểm tra artifact đã tồn tại mà không nạp."""
    target = model_dir(out_dir, model_key)
    return (target / MANIFEST_NAME).exists() and (
        (target / FACTORS_NAME).exists() or (target / STATE_NAME).exists()
    )


def save_train_csr(train: TrainData, out_dir: Path) -> Path:
    """Lưu ma trận train thưa để demo tự mask phim user đã xem.

    Ma trận train có 19,8M phần tử — pickle sẽ nặng, nên lưu dạng CSR thô
    (`indptr` + `indices` + `data`) qua `np.savez_compressed`.
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    csr = train.train_csr
    path = target / "train_csr.npz"
    np.savez_compressed(
        path,
        indptr=csr.indptr,
        indices=csr.indices,
        data=csr.data,
        shape=np.asarray(csr.shape, dtype="int64"),
    )
    return path


def load_train_csr(out_dir: Path) -> "sparse.csr_matrix":  # noqa: F821
    """Nạp lại ma trận train đã lưu (dùng cho mask phim đã xem trong demo)."""
    from scipy import sparse

    path = Path(out_dir) / "train_csr.npz"
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {path}. Chạy scripts/save_demo_artifacts.py trước.")
    with np.load(path) as z:
        shape = tuple(int(x) for x in z["shape"])
        return sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
