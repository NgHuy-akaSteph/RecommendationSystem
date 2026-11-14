"""Đường dẫn dùng chung cho toàn bộ pipeline.

Mọi module khác import từ đây thay vì hard-code path (yêu cầu tái lập của phase 6).

Khác biệt so với bản gốc của `recommendation-cb`: `RAW_DIR` trỏ tới `data/raw/ml-25m`
thay vì `ml-25m` vì `RecommendationSystem` đã có sẵn dữ liệu thô ở vị trí đó.
"""

from pathlib import Path

# src/paths.py -> src/ -> project root
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw" / "ml-25m"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "configs"

# File raw
RAW_RATINGS = RAW_DIR / "ratings.csv"
RAW_MOVIES = RAW_DIR / "movies.csv"
RAW_TAGS = RAW_DIR / "tags.csv"
RAW_GENOME_SCORES = RAW_DIR / "genome-scores.csv"
RAW_GENOME_TAGS = RAW_DIR / "genome-tags.csv"

# File cache sau tiền xử lý
RATINGS_PARQUET = PROCESSED_DIR / "ratings_genome.parquet"
ITEM_META_PARQUET = PROCESSED_DIR / "item_meta.parquet"
USER_MAP_PARQUET = PROCESSED_DIR / "user_map.parquet"
ITEM_GENRES_NPZ = PROCESSED_DIR / "item_genres.npz"
ITEM_TAG_DOCS_PARQUET = PROCESSED_DIR / "item_tag_docs.parquet"
ITEM_GENOME_NPY = PROCESSED_DIR / "item_genome.npy"
GENOME_TAG_NAMES_PARQUET = PROCESSED_DIR / "genome_tag_names.parquet"

# File sau khi chia train/test
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
DEV_USERS_NPY = PROCESSED_DIR / "dev_users.npy"

FINAL_DIR = RESULTS_DIR / "final"
FIGURES_DIR = RESULTS_DIR / "figures"


def beyond_accuracy_csv(protocol: str) -> Path:
    """Đường dẫn bảng chỉ số ngoài độ chính xác của một giao thức.

    Bản ở `results/final/` là bản được commit (nằm trong git); bản ở gốc `results/` là
    artifact trung gian (bị gitignore) do các script cũ ghi. Ưu tiên bản trong `final/`,
    fallback về bản gốc để vẫn chạy được với pipeline cũ.
    """
    final = FINAL_DIR / f"beyond-accuracy-{protocol}.csv"
    if final.exists():
        return final
    return RESULTS_DIR / f"beyond-accuracy-{protocol}.csv"

# Hằng số dùng chung
SEED = 42
RELEVANCE_THRESHOLD = 4.0  # rating >= 4 được coi là "relevant"
N_DEV_USERS = 20_000


def ensure_dirs() -> None:
    """Tạo các thư mục output nếu chưa có."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
