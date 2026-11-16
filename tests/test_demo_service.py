"""Test tầng serving — `src/api/`.

Bộ test này **không** cần artifact đã fit: dựng dữ liệu nhỏ trong bộ nhớ để kiểm
tra đúng *hợp đồng* của service (mask phim đã xem, fallback user lạ, ánh xạ
movieId, cold-start chỉ trả phim held-out).

Bộ test *có* cần artifact (`test_demo_artifacts_roundtrip.py`) thì đánh dấu `slow`
và bỏ qua nếu chưa sinh — vì artifact là 120 MB, không nên bắt CI tải về.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.recommender_service import RecommenderService  # noqa: E402
from src.models.seen_item_masking import SEEN_ITEM_SCORE  # noqa: E402

N_ITEMS = 12
N_USERS = 6
COLD_ITEM_IDX = 10  # item held-out của split cold-start


# Lớp giả phải để ở **cấp module** — `pickle` không serialize được lớp định nghĩa
# bên trong hàm (`local object`), mà artifact của mô hình nội dung lưu bằng pickle.
class FakeCFScorer:
    name = "cf-als"

    def score_users(self, user_idx):
        # Điểm tăng theo chỉ số item: user 0 thích item 3, user 1 thích item 4.
        out = np.zeros((len(user_idx), N_ITEMS), dtype="float32")
        for row, user in enumerate(user_idx):
            out[row] = (np.arange(N_ITEMS) == (3 + int(user) % 4)).astype("float32")
        return out


class FakeContentScorer:
    name = "cb2-bm25"

    def score_users(self, user_idx):
        out = np.zeros((len(user_idx), N_ITEMS), dtype="float32")
        for row, user in enumerate(user_idx):
            out[row] = (np.arange(N_ITEMS) == (6 + int(user) % 4)).astype("float32")
        return out


class FakeCounterScorer:
    """Scorer trả điểm = chỉ số item, dùng để kiểm tra artifact roundtrip."""

    def score_users(self, user_idx):
        return np.tile(np.arange(N_ITEMS, dtype="float32"), (len(user_idx), 1))


class FakeColdCFScorer:
    """CF fit trên split cold-start: item held-out có **factor bằng 0** → điểm 0."""

    name = "cf-als"

    def score_users(self, user_idx):
        out = np.zeros((len(user_idx), N_ITEMS), dtype="float32")
        for row, user in enumerate(user_idx):
            liked = 3 + int(user) % 4
            out[row, liked] = 1.0
        return out


class FakeColdContentScorer:
    """CB fit trên split cold-start: item held-out **vẫn có điểm** (nhờ nội dung)."""

    name = "cb2-bm25"

    def score_users(self, user_idx):
        out = np.zeros((len(user_idx), N_ITEMS), dtype="float32")
        for row, user in enumerate(user_idx):
            out[row, 6 + int(user) % 4] = 1.0
            # Phim cold-start vẫn được xếp hạng nhờ nội dung (thể loại + tag).
            out[row, COLD_ITEM_IDX] = 0.5
        return out


@pytest.fixture
def fake_processed(tmp_path: Path) -> Path:
    """Dựng `data/processed` giả: 12 phim, 6 user, có genome và tag names."""
    processed = tmp_path / "processed"
    processed.mkdir()

    pd.DataFrame({
        "item_idx": range(N_ITEMS),
        "movieId": [100 + i for i in range(N_ITEMS)],
        "title": [f"Phim {i}" for i in range(N_ITEMS)],
        "year": [1990 + i for i in range(N_ITEMS)],
        "genres": ["Drama"] * N_ITEMS,
    }).to_parquet(processed / "item_meta.parquet")

    pd.DataFrame({
        "user_idx": range(N_USERS),
        "userId": [1 + i for i in range(N_USERS)],
    }).to_parquet(processed / "user_map.parquet")

    pd.DataFrame({
        "col_idx": [0, 1],
        "tagId": [1, 2],
        "tag": ["space opera", "war"],
    }).to_parquet(processed / "genome_tag_names.parquet")

    rng = np.random.default_rng(42)
    genome = rng.random((N_ITEMS, 2)).astype("float32")
    np.save(processed / "item_genome.npy", genome)

    cold = processed / "coldstart"
    cold.mkdir()
    np.save(cold / "heldout_items.npy", np.array([10, 11], dtype="int64"))
    pd.DataFrame({"user_idx": [], "item_idx": []}).to_parquet(cold / "train.parquet")

    return processed


@pytest.fixture
def fake_artifacts(tmp_path: Path, fake_processed: Path) -> Path:
    """Artifact giả: một mô hình CF (factor) và một mô hình nội dung (pickle)."""
    from src.api.model_artifacts import save_artifact, save_train_csr

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    class _Wrapper:
        """Bọc scorer để khớp đường lưu của mô hình nội dung (`_scorer`)."""

        def __init__(self, inner):
            self._scorer = inner
            self.name = inner.name

    for key, scorer in (("cf-als", FakeCFScorer()), ("cb2-bm25", FakeContentScorer())):
        save_artifact(key, _Wrapper(scorer), artifacts, manifest_extra={"split": "full"})

    # Bộ artifact thứ hai: fit trên split cold-start, nơi phim held-out bị xoá
    # khỏi train. Đây là bộ mà tab cold-start của demo dùng.
    cold_dir = artifacts / "coldstart"
    cold_dir.mkdir()
    cold_scorers = {
        "cf-als": FakeColdCFScorer(),
        "cb2-bm25": FakeColdContentScorer(),
    }
    for key, scorer in cold_scorers.items():
        save_artifact(key, _Wrapper(scorer), cold_dir,
                      manifest_extra={"split": "coldstart"})

    train_csr = sparse.csr_matrix(
        (np.array([4.5, 3.0, 5.0], dtype="float32"), ([0, 0, 1], [1, 2, 3])),
        shape=(N_USERS, N_ITEMS),
    )
    from src.data.build_interaction_matrices import TrainData

    save_train_csr(
        TrainData(train_csr=train_csr, train_df=pd.DataFrame(), n_users=N_USERS, n_items=N_ITEMS),
        artifacts,
    )
    # Train của split cold-start không chứa phim held-out (10, 11).
    cold_csr = sparse.csr_matrix(
        (np.array([4.5, 3.0, 5.0], dtype="float32"), ([0, 0, 1], [1, 2, 3])),
        shape=(N_USERS, N_ITEMS),
    )
    save_train_csr(
        TrainData(train_csr=cold_csr, train_df=pd.DataFrame(), n_users=N_USERS, n_items=N_ITEMS),
        cold_dir,
    )
    return artifacts


@pytest.fixture
def service(fake_artifacts: Path, fake_processed: Path) -> RecommenderService:
    return RecommenderService(fake_artifacts, ["cf-als", "cb2-bm25"], fake_processed)


# ------------------------------------------------------------------- ranh giới

def test_serving_layer_never_imports_eval():
    """`src/api/` không được import `src.eval.*` — hàng rào chống rò rỉ ở tầng cấu trúc.

    Evaluator chấm điểm trên **nhãn test**; tầng serving chỉ được biết `TrainData`.
    Nếu ai đó import `src.eval.*` vào serving, demo có thể vô tình chạm nhãn.
    """
    import ast

    api_dir = Path(__file__).resolve().parent.parent / "src" / "api"
    offenders: list[str] = []
    for py_file in sorted(api_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.startswith("src.eval") or name.startswith("src.eval."):
                    offenders.append(f"{py_file.name}:{node.lineno} -> {name}")
    assert not offenders, (
        "Tầng serving không được import evaluator: " + "; ".join(offenders))


# ------------------------------------------------------------------ người dùng

def test_recommend_by_user_returns_expected_shape(service: RecommenderService):
    recs = service.recommend_by_user(user_id=1, model_key="cf-als", n=5)
    assert recs, "phải trả ít nhất một gợi ý"
    # Mô hình giả đặt điểm 1.0 cho item 3 với user_idx 0 -> phim đứng đầu là "Phim 3".
    assert recs[0].movie_id == 100 + 3
    assert recs[0].title == "Phim 3"
    assert recs[0].year == 1993
    assert recs[0].genres == "Drama"
    # Mọi gợi ý phải có đủ trường để hiển thị, và thứ hạng điểm không tăng.
    assert all(r.movie_id and r.title and r.genres for r in recs)
    assert recs == sorted(recs, key=lambda r: -r.score)


def test_recommend_excludes_already_seen_items(service: RecommenderService):
    # User 1 (user_idx 0) đã xem item 1 và 2 trong train.
    recs = service.recommend_by_user(user_id=1, model_key="cf-als", n=10)
    seen = {h["movie_id"] for h in service.user_history(1, limit=999)}
    assert seen == {101, 102}
    assert not seen.intersection({r.movie_id for r in recs})


def test_masked_items_do_not_leak_into_ranking(service: RecommenderService):
    """Item đã xem phải bị đẩy xuống đáy, không phải chỉ bị loại khỏi top-N."""
    scores = np.zeros(N_ITEMS, dtype="float32")
    scores[1] = 0.9
    masked = service._mask_seen(scores, 0)
    assert masked[1] == SEEN_ITEM_SCORE


def test_unknown_user_falls_back_to_popular_without_crashing(service: RecommenderService):
    recs = service.recommend_by_user(user_id=999_999, model_key="cf-als", n=3)
    assert len(recs) == 3
    assert all(r.reason and "không có trong dữ liệu" in r.reason for r in recs)


def test_user_without_history_falls_back(service: RecommenderService):
    # user_id 4 (user_idx 3) không có dòng nào trong train_csr.
    recs = service.recommend_by_user(user_id=4, model_key="cf-als", n=3)
    assert len(recs) == 3
    assert all(r.reason and "chưa có lịch sử" in r.reason for r in recs)


def test_user_history_lists_seen_movies(service: RecommenderService):
    history = service.user_history(1)
    assert [h["movie_id"] for h in history] == [101, 102]
    assert history[0]["rating"] == pytest.approx(4.5)


def test_known_user_ids_sorted_by_activity(service: RecommenderService):
    ids = service.known_user_ids()
    assert 1 in ids and 2 in ids
    # User 1 có 2 tương tác, user 2 có 1 -> user 1 phải đứng trước.
    assert ids.index(1) < ids.index(2)


# ----------------------------------------------------------------------- phim

def test_similar_movies_excludes_the_source_movie(service: RecommenderService):
    similar = service.similar_movies(movie_id=100 + 6, model_key="cb2-bm25", n=5)
    assert all(s.movie_id != 100 + 6 for s in similar)


def test_similar_movies_reports_shared_tags(service: RecommenderService):
    similar = service.similar_movies(movie_id=100, model_key="cb2-bm25", n=3)
    assert similar
    assert similar[0].reason is None or "Tag chung" in similar[0].reason


def test_search_movies_matches_title_case_insensitively(service: RecommenderService):
    hits = service.search_movies("phim 1")
    assert any(h["movie_id"] == 101 for h in hits)


def test_search_movies_empty_query_returns_nothing(service: RecommenderService):
    assert service.search_movies("") == []


def test_unknown_movie_id_returns_empty(service: RecommenderService):
    assert service.similar_movies(movie_id=999_999, n=5) == []


# ------------------------------------------------------------------ cold-start

def test_coldstart_items_come_from_heldout_file(service: RecommenderService):
    items = service.coldstart_items()
    assert {i["movie_id"] for i in items} == {110, 111}


def test_coldstart_comparison_reports_both_models(service: RecommenderService):
    result = service.coldstart_movies_for(item_idx=COLD_ITEM_IDX, n=5)
    assert set(result["models"]) == {"cf-als", "cb2-bm25"}
    assert result["item"]["movie_id"] == 110
    for payload in result["models"].values():
        assert "count" in payload
        assert payload["count"] == len(payload["items"])


def test_coldstart_cf_scores_zero_while_content_ranks_the_item(service: RecommenderService):
    """Điểm cốt lõi của demo cold-start: CF không có tín hiệu, CB thì có."""
    result = service.coldstart_movies_for(item_idx=COLD_ITEM_IDX, n=5)
    assert result["available"] is True

    cf = result["models"]["cf-als"]
    cb = result["models"]["cb2-bm25"]

    # CF: phim held-out có factor bằng 0 -> điểm 0, không vào top-N.
    assert cf["cold_item_score"] == 0.0
    assert cf["cold_item_in_top_n"] is False

    # CB: hồ sơ nội dung dựng được không cần tương tác -> vẫn xếp hạng được.
    assert cb["cold_item_score"] > 0.0
    assert cb["cold_item_in_top_n"] is True
    assert cb["cold_item_rank"] is not None and cb["cold_item_rank"] <= 5


# ------------------------------------------------------------------- artifact

def test_artifact_roundtrip_preserves_scores(tmp_path: Path, fake_processed: Path):
    """Lưu rồi nạp artifact phải cho **cùng** điểm — không lệch."""
    from src.api.model_artifacts import load_artifact, save_artifact

    class _Model:
        name = "cb2-bm25"

        def __init__(self):
            self._scorer = FakeCounterScorer()

    out = tmp_path / "artifacts"
    save_artifact("cb2-bm25", _Model(), out)
    loaded = load_artifact(out, "cb2-bm25")
    assert np.array_equal(
        loaded.score_users(np.array([0, 1], dtype="int64")),
        FakeCounterScorer().score_users(np.array([0, 1], dtype="int64")),
    )
    assert loaded.manifest["model_key"] == "cb2-bm25"


def test_load_artifact_missing_raises_clear_error(tmp_path: Path):
    from src.api.model_artifacts import load_artifact

    with pytest.raises(FileNotFoundError, match="save_demo_artifacts"):
        load_artifact(tmp_path, "cf-als")
