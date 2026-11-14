"""Tạo split cold-start item: giữ lại 10% phim, xoá mọi tương tác train của chúng.

Đây là thí nghiệm then chốt của đồ án. Lọc cộng tác mô tả một phim bằng "những ai đã
thích nó"; phim không có tương tác train nào thì mô tả đó rỗng, nên ItemKNN-CF cho điểm
đúng bằng 0 và ALS chỉ còn vector khởi tạo. Lọc nội dung thì không đổi gì cả — đặc trưng
của phim đến từ thể loại, tag và genome, không đến từ hành vi. Chênh lệch ở đây chính là
luận điểm chính của đề tài.

Phân tầng theo năm phát hành (5 nhóm phân vị) thay vì lấy mẫu ngẫu nhiên đều: phim mới
vốn ít tương tác hơn, nếu bốc ngẫu nhiên thì tập held-out sẽ lệch về phim cũ nhiều rating
và bài toán trở nên dễ hơn thực tế.

Item **không** bị loại khỏi ma trận: số cột giữ nguyên 13.816 nên mọi thuật toán vẫn xếp
hạng trên cùng tập ứng viên, và điểm 0 của CF vẫn được đặt cạnh điểm thật của CB.

Chạy: python -m src.eval.cold_start_item_split [--holdout-ratio 0.1] [--force]
"""

import argparse

import numpy as np
import pandas as pd

from src import paths

COLDSTART_DIR = paths.PROCESSED_DIR / "coldstart"
HELDOUT_ITEMS_NPY = COLDSTART_DIR / "heldout_items.npy"
N_YEAR_BUCKETS = 5


def select_heldout_items(item_meta: pd.DataFrame, ratio: float, seed: int) -> np.ndarray:
    """Chọn `ratio` phim mỗi nhóm năm phát hành. Phim thiếu năm (18 phim) thành nhóm riêng."""
    year = item_meta["year"]
    bucket = pd.qcut(year, N_YEAR_BUCKETS, labels=False, duplicates="drop")
    bucket = bucket.fillna(-1).astype("int8")   # -1 = không rõ năm

    if not 0 < ratio <= 1:
        raise ValueError(f"holdout_ratio phải trong (0, 1], nhận {ratio}")

    rng = np.random.default_rng(seed)
    chosen = []
    for b in np.unique(bucket):
        idx = item_meta.loc[bucket == b, "item_idx"].to_numpy()
        n_hold = min(len(idx), max(1, int(round(len(idx) * ratio))))
        chosen.append(rng.choice(idx, size=n_hold, replace=False))
        print(f"  nhóm năm {b}: {len(idx):5,} phim → giữ lại {n_hold:4,}")
    return np.sort(np.concatenate(chosen)).astype("int32")


def build_split(heldout: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """train = train gốc bỏ mọi rating trên phim held-out; test = chỉ giữ rating trên chúng.

    Trả thêm số rating train gốc để caller báo cáo tỷ lệ mất mát mà không hardcode con số.
    """
    train = pd.read_parquet(paths.TRAIN_PARQUET)
    test = pd.read_parquet(paths.TEST_PARQUET)
    held = np.zeros(len(pd.read_parquet(paths.ITEM_META_PARQUET, columns=["item_idx"])), dtype=bool)
    held[heldout] = True

    cold_train = train[~held[train["item_idx"].to_numpy()]].copy()
    cold_test = test[held[test["item_idx"].to_numpy()]].copy()
    return cold_train, cold_test, len(train)


def main(holdout_ratio: float = 0.1, seed: int = paths.SEED, force: bool = False) -> None:
    if HELDOUT_ITEMS_NPY.exists() and not force:
        print("Đã có split cold-start, bỏ qua (dùng --force để tạo lại).")
        return
    if not paths.TRAIN_PARQUET.exists():
        raise FileNotFoundError("Chạy src.data.split_temporal_per_user trước.")

    COLDSTART_DIR.mkdir(parents=True, exist_ok=True)
    item_meta = pd.read_parquet(paths.ITEM_META_PARQUET, columns=["item_idx", "year"])

    heldout = select_heldout_items(item_meta, holdout_ratio, seed)
    cold_train, cold_test, n_train_original = build_split(heldout)

    np.save(HELDOUT_ITEMS_NPY, heldout)
    cold_train.to_parquet(COLDSTART_DIR / "train.parquet", index=False)
    cold_test.to_parquet(COLDSTART_DIR / "test.parquet", index=False)

    relevant = cold_test[cold_test["rating"] >= paths.RELEVANCE_THRESHOLD]
    dev_users = np.load(paths.DEV_USERS_NPY)
    eligible_dev = np.intersect1d(dev_users, relevant["user_idx"].unique())

    print(f"\nPhim held-out: {len(heldout):,}/{len(item_meta):,} ({len(heldout)/len(item_meta):.1%})")
    print(f"Train: {len(cold_train):,} rating ({len(cold_train) / n_train_original - 1:+.1%} so với gốc)")
    print(f"Test:  {len(cold_test):,} rating, {len(relevant):,} relevant")
    print(f"Dev user đánh giá được: {len(eligible_dev):,}/{len(dev_users):,}")
    print(f"\nĐã lưu vào {COLDSTART_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-ratio", type=float, default=0.1, help="Tỷ lệ phim giữ lại")
    parser.add_argument("--seed", type=int, default=paths.SEED)
    parser.add_argument("--force", action="store_true", help="Tạo lại kể cả khi đã có")
    main(**vars(parser.parse_args()))
