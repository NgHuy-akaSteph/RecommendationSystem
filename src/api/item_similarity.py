"""Tương đồng item-item cho gợi ý "phim tương tự".

Vì sao không dùng `score_users` của mô hình nội dung: `ContentProfileScorer` tính
hồ sơ user là trung bình có trọng số của các phim đã xem, nên dùng nó để tìm "phim
tương tự phim X" phải giả lập một user chỉ xem đúng X — kết quả là một hồ sơ rất
loãng và hàng xóm bị nhiễu (đo thử: "Star Wars IV" dẫn tới "Forrest Gump").

Giao diện này dùng **cosine trực tiếp giữa hai phim** trên Tag Genome:

    sim(i, j) = cos(genome_i, genome_j)

Tag Genome là 1.128 chiều do GroupLens tính ngoài động cho *mọi* phim, nên phản ánh
đặc trưng nội dung thuần (thể loại, chủ đề, tông màu...). Kết quả: "Madness of King
George" dẫn tới "Elizabeth", "Becket", "Young Victoria" — đúng nghĩa "phim tương tự".

Lưu ý metadata hậu kỳ: `genome-scores.csv` của GroupLens **không có timestamp**,
tức được tính ngoài động trên toàn corpus. Ở đây điều đó không làm sai kết luận
vì so sánh vẫn là giữa các phim cùng điều kiện, nhưng đây là điểm phải nêu khi
trình bày (báo cáo Chương 6 đo định lượng rủi ro này).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


class ItemSimilarityIndex:
    """Chỉ mục cosine trên Tag Genome, dựng một lần rồi trả top-k hàng xóm."""

    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        genome_path = self.processed_dir / "item_genome.npy"
        if not genome_path.exists():
            raise FileNotFoundError(
                f"Thiếu {genome_path}. Chạy: venv/bin/python -m src.data.build_item_content_features"
            )
        raw = np.load(genome_path, mmap_mode="r")
        # z-score theo từng chiều rồi chuẩn hoá L2 — cùng cách CB-3 dùng.
        X = np.array(raw, dtype="float32", copy=True)
        X[X < 0.2] = 0.0
        X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), EPS)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        self.X = np.ascontiguousarray(X / np.maximum(norms, EPS))

        names_path = self.processed_dir / "genome_tag_names.parquet"
        if names_path.exists():
            df = pd.read_parquet(names_path)
            self._tag_names = dict(zip(df["col_idx"].astype(int), df["tag"].astype(str)))
        else:
            self._tag_names = {}

    def neighbors(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Top-n phim tương tự (item_idx, cosine), không gồm chính nó."""
        if not 0 <= item_idx < len(self.X):
            return []
        sims = self.X @ self.X[item_idx]
        sims[item_idx] = -np.inf
        k = min(n, len(sims) - 1)
        top = np.argpartition(-sims, kth=k - 1)[:k]
        order = top[np.argsort(-sims[top], kind="stable")]
        return [(int(i), float(sims[i])) for i in order if np.isfinite(sims[i])]

    def shared_tags(self, item_idx: int, other_idx: int, top_k: int = 5) -> list[str]:
        """Tag có điểm genome cao ở **cả hai** phim — lý do cụ thể cho gợi ý."""
        if not (0 <= item_idx < len(self.X) and 0 <= other_idx < len(self.X)):
            return []
        common = np.minimum(self.X[item_idx], self.X[other_idx])
        if not np.any(common > 0):
            return []
        top_cols = np.argsort(-common)[:top_k]
        labels = [self._tag_names.get(int(c), "") for c in top_cols]
        return [l for l in labels if l]
