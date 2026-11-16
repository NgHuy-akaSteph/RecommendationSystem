"""Tầng serving: biến mô hình đánh giá offline thành API gợi ý dùng được.

Bản đánh giá trả về `item_idx` — chỉ số 0..13815 nội bộ. Người dùng thật cần
`movieId`, tên phim, năm, thể loại. Module này là chỗ duy nhất biết cách ánh xạ
ngược lại, và là nơi áp dụng các quy tắc mà evaluator không cần:

    - mask phim user **đã xem** (trong lịch sử train);
    - trả kèm **lý do** cho gợi ý content-based (tag genome chung);
    - xử lý **user lạ / user mới** bằng fallback thay vì crash.

Ràng buộc quan trọng: service **không** được chạm nhãn test. Nó chỉ đọc
`TrainData` (tức `train_csr`) — cùng dữ liệu mà mô hình đã học.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.api.model_artifacts import ModelArtifact, load_artifact, load_train_csr
from src.models.seen_item_masking import SEEN_ITEM_SCORE


@dataclass
class Recommendation:
    """Một gợi ý đã có đủ thông tin để hiển thị."""

    item_idx: int
    movie_id: int
    title: str
    year: float | None
    genres: str
    score: float
    reason: str | None = None  # lý do giải thích (chỉ có với gợi ý content-based)

    def as_dict(self) -> dict:
        return {
            "item_idx": self.item_idx,
            "movie_id": self.movie_id,
            "title": self.title,
            "year": self.year,
            "genres": self.genres,
            "score": round(float(self.score), 6),
            "reason": self.reason,
        }


class RecommenderService:
    """Cổng gợi ý dùng chung cho demo và (nếu cần) API HTTP.

    Khởi tạo một lần rồi tái dùng: `score_users` chỉ tốn ~0,01s cho 200 user với
    ALS, nên không cần cache kết quả theo user.
    """

    def __init__(
        self,
        artifacts_dir: Path,
        model_keys: list[str],
        processed_dir: Path,
    ):
        """
        Args:
            artifacts_dir: thư mục chứa artifact (mặc định `demo/artifacts`).
            model_keys: danh sách khoá mô hình cần phục vụ, theo thứ tự hiển thị.
            processed_dir: thư mục `data/processed` — nơi có metadata item/user.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.processed_dir = Path(processed_dir)
        self.model_keys = list(model_keys)
        self.artifacts: dict[str, ModelArtifact] = {
            key: load_artifact(self.artifacts_dir, key) for key in self.model_keys
        }
        self.train_csr: sparse.csr_matrix = load_train_csr(self.artifacts_dir)

        self._item_meta = pd.read_parquet(self.processed_dir / "item_meta.parquet")
        self._item_meta = self._item_meta.sort_values("item_idx").reset_index(drop=True)
        self._item_titles = self._item_meta["title"].tolist()
        self._item_movie_ids = self._item_meta["movieId"].tolist()
        self._item_years = self._item_meta["year"].tolist()
        self._item_genres = self._item_meta["genres"].tolist()

        user_map = pd.read_parquet(self.processed_dir / "user_map.parquet")
        self._user_id_to_idx = dict(
            zip(user_map["userId"].astype("int64"), user_map["user_idx"].astype("int64")))
        self._user_idx_to_id = {v: k for k, v in self._user_id_to_idx.items()}

        self._popularity = np.asarray(
            (self.train_csr >= 4.0).astype("float32").sum(axis=0)).ravel()

    # ------------------------------------------------------------------ người dùng

    def known_user_ids(self) -> list[int]:
        """Danh sách userId thực (sắp theo số tương tác giảm dần) — dùng gợi ý trong UI."""
        counts = np.diff(self.train_csr.indptr)
        order = np.argsort(-counts)
        return [self._user_idx_to_id[int(i)] for i in order[:200] if int(i) in self._user_idx_to_id]

    def user_history(self, user_id: int, limit: int = 15) -> list[dict]:
        """Lịch sử xem gần đây của user (theo thứ tự thời gian train)."""
        idx = self._user_id_to_idx.get(int(user_id))
        if idx is None:
            return []
        row = self.train_csr.getrow(idx)
        return [
            {
                "movie_id": self._item_movie_ids[item],
                "title": self._item_titles[item],
                "year": self._item_years[item],
                "genres": self._item_genres[item],
                "rating": float(rating),
            }
            for item, rating in zip(row.indices, row.data)
        ][:limit]

    def recommend_by_user(
        self,
        user_id: int,
        model_key: str | None = None,
        n: int = 10,
        exclude_seen: bool = True,
        with_reason: bool = True,
    ) -> list[Recommendation]:
        """Top-N gợi ý cho một userId.

        User không tồn tại hoặc không có lịch sử train nào → fallback sang
        MostPop, kèm lý do để UI hiển thị minh bạch.
        """
        model_key = model_key or self.model_keys[0]
        artifact = self.artifacts[model_key]
        idx = self._user_id_to_idx.get(int(user_id))

        if idx is None:
            return self._fallback_popular(n, reason=f"User {user_id} không có trong dữ liệu")
        if self.train_csr.getrow(idx).nnz == 0:
            return self._fallback_popular(n, reason=f"User {user_id} chưa có lịch sử tương tác")

        scores = np.asarray(artifact.score_users(np.array([idx], dtype="int64")))[0]
        if exclude_seen:
            scores = self._mask_seen(scores, idx)

        ranked = self._rank(scores, n)
        history = {item["movie_id"]: item["title"] for item in self.user_history(user_id, limit=50)}
        return [
            self._to_recommendation(
                item_idx, float(scores[item_idx]),
                reason=self._reason_for(item_idx, history, model_key) if with_reason else None,
            )
            for item_idx in ranked
        ]

    # -------------------------------------------------------------------- phim

    def similar_movies(
        self,
        movie_id: int,
        model_key: str = "cb2-bm25",
        n: int = 10,
        with_reason: bool = True,
    ) -> list[Recommendation]:
        """Các phim tương tự theo nội dung, kèm tag genome chung làm lý do.

        Dùng chỉ mục cosine Tag Genome (`ItemSimilarityIndex`) chứ không phải
        `score_users` của mô hình nội dung: profile scorer gộp mọi phim user đã xem
        thành một hồ sơ trung bình, dùng để tìm "phim tương tự" thì ra kết quả nhiễu.
        """
        item_idx = self._movie_id_to_item_idx(movie_id)
        if item_idx is None:
            return []

        index = self._similarity_index()
        neighbors = index.neighbors(item_idx, n=n)
        return [
            self._to_recommendation(
                idx, sim,
                reason=(
                    "Tag chung: " + ", ".join(index.shared_tags(item_idx, idx))
                    if with_reason and index.shared_tags(item_idx, idx)
                    else None
                ),
            )
            for idx, sim in neighbors
        ]

    def search_movies(self, query: str, limit: int = 50) -> list[dict]:
        """Tìm phim theo tên (không phân biệt hoa thường)."""
        if not query:
            return []
        needle = query.strip().lower()
        hits = []
        for idx, title in enumerate(self._item_titles):
            if needle in str(title).lower():
                hits.append({
                    "item_idx": idx,
                    "movie_id": self._item_movie_ids[idx],
                    "title": self._item_titles[idx],
                    "year": self._item_years[idx],
                    "genres": self._item_genres[idx],
                })
                if len(hits) >= limit:
                    break
        return hits

    # ---------------------------------------------------------------- cold-start

    def coldstart_items(self, limit: int = 200) -> list[dict]:
        """Danh sách phim từ tập held-out của giao thức cold-start (1.381 phim)."""
        path = self.processed_dir / "coldstart" / "heldout_items.npy"
        if not path.exists():
            return []
        heldout = np.load(path)
        return [
            {
                "item_idx": int(idx),
                "movie_id": self._item_movie_ids[int(idx)],
                "title": self._item_titles[int(idx)],
                "year": self._item_years[int(idx)],
                "genres": self._item_genres[int(idx)],
            }
            for idx in heldout[:limit]
        ]

    def coldstart_movies_for(
        self,
        item_idx: int,
        n: int = 10,
    ) -> dict:
        """So sánh CF và CB trên **cùng một phim cold-start**.

        Đây là màn hình trung tâm khi bảo vệ. Câu hỏi mô hình được hỏi: *"với người
        dùng u, hệ thống có xếp hạng được phim cold-start này không?"*

        **Vì sao phải dùng mô hình fit trên split cold-start:** nếu dùng mô hình fit
        trên split `full`, nó **đã học** tương tác của phim held-out và sẽ xếp hạng
        phim đó bình thường — sai lệch hoàn toàn so với giao thức mà báo cáo đo.
        Ở split cold-start, tương tác của 1.381 phim bị xoá khỏi train, nên:
            - ALS: hàng factor của phim đó bằng 0 → điểm mọi user với phim đó = 0;
            - BM25: hồ sơ vẫn dựng được từ tag → vẫn xếp hạng được.

        Vì vậy artifact cold-start được sinh riêng bằng
        `scripts/save_demo_artifacts.py --split coldstart` và nằm ở
        `demo/artifacts/coldstart/`.
        """
        cold = self._coldstart_service()
        if cold is None:
            return {
                "item": self._coldstart_item_info(item_idx),
                "models": {},
                "available": False,
            }

        return cold._coldstart_comparison(item_idx, n=n)

    def _coldstart_item_info(self, item_idx: int) -> dict:
        return {
            "item_idx": item_idx,
            "movie_id": self._item_movie_ids[item_idx],
            "title": self._item_titles[item_idx],
            "year": self._item_years[item_idx],
            "genres": self._item_genres[item_idx],
        }

    def _coldstart_comparison(self, item_idx: int, n: int = 10) -> dict:
        """Đo trực tiếp: mô hình có đưa phim cold-start này vào top-N của user không?

        Cùng một user, cùng một phim held-out, hai mô hình khác nhau:
            - ALS (CF): hàng factor của phim held-out bằng 0 (không có tương tác
              train nào) → **không thể** xếp hạng, không bao giờ vào top-N.
            - BM25 (CB): hồ sơ dựng từ tag của chính phim đó → xếp hạng được.

        Đây chính là phép đo mà báo cáo quy về NDCG@10 = 0,0000 (CF) vs 0,0306 (CB)
        ở giao thức `coldstart` — ở đây ta thấy **từng user** thay vì trung bình.
        """
        out: dict = {
            "item": self._coldstart_item_info(item_idx),
            "models": {},
            "available": True,
        }

        # Chọn user có lịch sử train dày để hồ sơ CB đủ tín hiệu.
        counts = np.diff(self.train_csr.indptr)
        user_idx = int(np.argsort(-counts)[0])
        user_id = self._user_idx_to_id.get(user_idx)

        for key, artifact in self.artifacts.items():
            scores = np.asarray(artifact.score_users(np.array([user_idx], dtype="int64")))[0]
            scores = self._mask_seen(scores, user_idx)

            # Vị trí của phim cold-start trong bảng xếp hạng của user này.
            item_score = float(scores[item_idx])
            ranked = self._rank(scores, n)
            in_top_n = item_idx in ranked
            rank = self._rank_position(scores, item_idx)

            out["models"][key] = {
                "count": len(ranked),
                "items": [
                    self._to_recommendation(i, float(scores[i])).as_dict() for i in ranked
                ],
                "cold_item_score": item_score,
                "cold_item_in_top_n": in_top_n,
                "cold_item_rank": rank,
                "user_id": user_id,
            }
        return out

    def _rank_position(self, scores: np.ndarray, item_idx: int) -> int | None:
        """Thứ hạng 1-based của một item trong bảng xếp hạng; None nếu không hạnh được."""
        usable = np.isfinite(scores) & (scores > SEEN_ITEM_SCORE)
        if not usable[item_idx]:
            return None
        better = int((scores[usable] > scores[item_idx]).sum())
        return better + 1

    def _coldstart_service(self):
        """Service riêng dùng artifact fit trên split cold-start (nếu đã sinh)."""
        if not hasattr(self, "_cold_service"):
            from src.api.model_artifacts import artifact_exists

            cold_dir = self.artifacts_dir / "coldstart"
            ready = all(artifact_exists(cold_dir, key) for key in self.model_keys)
            self._cold_service = (
                RecommenderService(cold_dir, self.model_keys, self.processed_dir)
                if ready else None
            )
        return self._cold_service

    # ------------------------------------------------------------------ nội bộ

    def _mask_seen(self, scores: np.ndarray, user_idx: int) -> np.ndarray:
        scores = scores.copy()
        seen = self.train_csr.getrow(user_idx).indices
        scores[seen] = SEEN_ITEM_SCORE
        return scores

    def _rank(self, scores: np.ndarray, n: int) -> list[int]:
        """Lấy n item có điểm cao nhất, bỏ item đã bị mask (−1e9 / −inf)."""
        finite = np.isfinite(scores) & (scores > SEEN_ITEM_SCORE)
        if not finite.any():
            return []
        masked = np.where(finite, scores, -np.inf)
        k = min(n, int(finite.sum()))
        part = np.argpartition(-masked, kth=k - 1)[:k]
        order = part[np.argsort(-masked[part], kind="stable")]
        return [int(i) for i in order]

    def _to_recommendation(
        self, item_idx: int, score: float, reason: str | None = None
    ) -> Recommendation:
        return Recommendation(
            item_idx=item_idx,
            movie_id=int(self._item_movie_ids[item_idx]),
            title=str(self._item_titles[item_idx]),
            year=self._item_years[item_idx],
            genres=str(self._item_genres[item_idx]),
            score=score,
            reason=reason,
        )

    def _fallback_popular(self, n: int, reason: str) -> list[Recommendation]:
        """Gợi ý phổ biến nhất — dùng khi không có tín hiệu nào cho user."""
        top = np.argsort(-self._popularity)[:n]
        return [
            self._to_recommendation(int(i), float(self._popularity[i]), reason=reason)
            for i in top
        ]

    def _movie_id_to_item_idx(self, movie_id: int) -> int | None:
        try:
            position = self._item_movie_ids.index(int(movie_id))
        except ValueError:
            return None
        return position

    def _shared_genome_tags(self, item_idx: int, other_idx: int, top_k: int = 5) -> str | None:
        """Các tag genome **chung** giữa hai phim — dùng làm lý do giải thích.

        Chỉ những tag mà **cả hai** phim đều có điểm genome cao mới được tính là
        "chung"; nếu chỉ lấy tag của phim gợi ý thì lý do đó là vô nghĩa (mọi phim
        đều có tag riêng, không giải thích được vì sao phim này *giống* phim kia).
        """
        genome = self._genome()
        if genome is None:
            return None
        source = genome[item_idx]
        target = genome[other_idx]
        if not np.any(source) or not np.any(target):
            return None
        # Điểm chung = min của hai phim (giống quy ước "tag chung" của Tag Genome).
        common = np.minimum(source, target)
        if not np.any(common > 0):
            return None
        top_cols = np.argsort(-common)[:top_k]
        names = self._genome_tag_names()
        labels = [names.get(int(c), "") for c in top_cols]
        labels = [l for l in labels if l]
        if not labels:
            return None
        return "Tag chung: " + ", ".join(labels)

    def _genome(self) -> np.ndarray | None:
        if not hasattr(self, "_genome_cache"):
            path = self.processed_dir / "item_genome.npy"
            self._genome_cache = np.load(path, mmap_mode="r") if path.exists() else None
        return self._genome_cache

    def _similarity_index(self):
        """Chỉ mục tương đồng item-item, dựng một lần rồi dùng lại."""
        if not hasattr(self, "_sim_index"):
            from src.api.item_similarity import ItemSimilarityIndex

            self._sim_index = ItemSimilarityIndex(self.processed_dir)
        return self._sim_index

    def _genome_tag_names(self) -> dict[int, str]:
        if not hasattr(self, "_tag_names_cache"):
            path = self.processed_dir / "genome_tag_names.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                self._tag_names_cache = dict(zip(df["col_idx"].astype(int), df["tag"].astype(str)))
            else:
                self._tag_names_cache = {}
        return self._tag_names_cache

    def _reason_for(
        self, item_idx: int, history: dict[int, str], model_key: str
    ) -> str | None:
        """Lý do gợi ý cho user.

        Với mô hình nội dung, lý do là các tag genome mà phim gợi ý **và** các phim
        user đã xem cùng có. Với mô hình CF không có lý do nội tại (nó chỉ biết ma
        trận factor) nên trả None — thừa nhận giới hạn thay vì bịa lý do.
        """
        if model_key not in ("cb1-tfidf", "cb2-bm25", "cb3-genome", "cb4-hybrid"):
            return None
        if not history:
            return None

        genome = self._genome()
        if genome is None:
            return None
        target = genome[item_idx]
        if not np.any(target):
            return None

        # Cộng điểm genome của các phim user đã xem, rồi lấy các tag mà cả phim
        # gợi ý lẫn lịch sử đều có.
        best_shared: list[tuple[float, str]] = []
        names = self._genome_tag_names()
        for movie_id in list(history)[:20]:
            source_idx = self._movie_id_to_item_idx(movie_id)
            if source_idx is None:
                continue
            common = np.minimum(genome[source_idx], target)
            if not np.any(common > 0):
                continue
            for col in np.argsort(-common)[:3]:
                score = float(common[col])
                if score <= 0:
                    continue
                label = names.get(int(col), "")
                if label:
                    best_shared.append((score, label))
        if not best_shared:
            return None
        best_shared.sort(reverse=True)
        seen_labels: list[str] = []
        for _, label in best_shared:
            if label not in seen_labels:
                seen_labels.append(label)
            if len(seen_labels) == 5:
                break
        return "Giống phim đã xem: " + ", ".join(seen_labels)
