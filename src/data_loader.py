from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class ML25MLoader:
    """Loader for MovieLens 25M dataset with memory optimization"""

    def __init__(self, data_dir='data/raw/ml-25m'):
        self.data_dir = Path(data_dir)

    def load_ratings(self, sample_frac=None):
        """
        Load ratings.csv with dtype optimization

        Args:
            sample_frac: Float (0-1) for sample random subset, None for load all

        Returns:
            pd.DataFrame columns: userId, movieId, rating, timestamp
        """
        dtype_dict = {
            'userId': 'int32',
            'movieId': 'int32',
            'rating': 'float16',
            'timestamp': 'int32'
        }

        ratings_path = self.data_dir / 'ratings.csv'

        if sample_frac is not None:
            n = sum(1 for _ in open(ratings_path)) - 1  # Total rows minus header
            skip_idx = np.random.choice(range(1, n + 1),
                                        size=int(n * (1 - sample_frac)),
                                        replace=False)
            ratings = pd.read_csv(ratings_path,
                                  dtype=dtype_dict,
                                  skiprows=skip_idx)
        else:
            ratings = pd.read_csv(ratings_path, dtype=dtype_dict)

        return ratings

    def load_movies(self):
        """
        Load movies.csv

        Returns:
            pd.DataFrame columns: movieId, title, genres
        """
        movies_path = self.data_dir / 'movies.csv'
        return pd.read_csv(movies_path)

    def load_tags(self):
        """Load tags.csv"""
        tags_path = self.data_dir / 'tags.csv'
        return pd.read_csv(tags_path)


class DataPreprocessor:
    """Preprocessing utilities for data"""

    @staticmethod
    def temporal_split(ratings_df, train_ratio=0.7):
        """
        Split data by timestamp (temporal split)

        Args:
            ratings_df: DataFrame -> timestamp column
            train_ratio

        Returns:
            train_df, test_df
        """
        # Sort by timestamp
        sorted_df = ratings_df.sort_values('timestamp').reset_index(drop=True)

        # Calculate split index
        split_idx = int(len(sorted_df) * train_ratio)

        train = sorted_df.iloc[:split_idx].copy()
        test = sorted_df.iloc[split_idx:].copy()

        return train, test

    @staticmethod
    def temporal_user_split(ratings_df, val_ratio=0.2):
        """
        Per-user temporal split: trong moi user, lay val_ratio rating MOI NHAT lam val.

        Tranh temporal leakage (val rating co timestamp cu hon train_sub rating)
        khi dung per-user random split. Val luon nam "trong tuong lai" cua train_sub.

        Args:
            ratings_df: DataFrame co userId, movieId, rating, timestamp
            val_ratio: Ty le val (0.2 = 20% rating moi nhat)

        Returns:
            (train_sub_df, val_df, n_singletons)
            - train_sub_df: Phan train (rating cu hon)
            - val_df: Phan val (rating moi hon)
            - n_singletons: So user co 1 rating -> giu nguyen trong train_sub
        """
        if val_ratio <= 0 or val_ratio >= 1:
            raise ValueError(f"val_ratio phai nam trong (0, 1), nhan {val_ratio}")

        required_cols = {'userId', 'timestamp'}
        missing = required_cols - set(ratings_df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # Sort theo (userId, timestamp) de groupby khong can sort lai.
        # Them row_index de dam bao deterministic split khi nhieu rating cung timestamp.
        sorted_df = ratings_df.copy()
        sorted_df['_orig_idx'] = np.arange(len(sorted_df))
        sorted_df = sorted_df.sort_values(['userId', 'timestamp', '_orig_idx']).reset_index(drop=True)

        train_parts, val_parts = [], []
        n_singletons = 0
        for _uid, _grp in sorted_df.groupby('userId', sort=False):
            n = len(_grp)
            if n < 2:
                # Singleton: 1 rating -> giu nguyen trong train_sub
                train_parts.append(_grp.drop(columns=['_orig_idx']))
                n_singletons += 1
                continue
            n_val_target = max(1, int(round(n * val_ratio)))
            if n_val_target >= n:
                # Edge case: n_val_target == n -> dat it nhat 1 row vao train_sub
                n_val_target = n - 1

            # Neu co nhieu rating cung timestamp tai bien phan chia,
            # mo rong val de gom toan bo cluster cung timestamp.
            ts_arr = _grp['timestamp'].to_numpy()
            n_train = n - n_val_target
            # Neu timestamp tai n_train-1 == timestamp tai n_train, expand n_train
            while n_train > 0 and n_train < n and ts_arr[n_train - 1] == ts_arr[n_train]:
                n_train -= 1
            # Dam bao val co it nhat 1 row
            if n_train == n:
                n_train = n - 1
            n_val = n - n_train

            train_parts.append(_grp.iloc[:n_train].drop(columns=['_orig_idx']))
            val_parts.append(_grp.iloc[n_train:].drop(columns=['_orig_idx']))

        train_sub = pd.concat(train_parts, ignore_index=True) if train_parts else \
            ratings_df.iloc[0:0].copy()
        val = pd.concat(val_parts, ignore_index=True) if val_parts else \
            ratings_df.iloc[0:0].copy()

        return train_sub, val, n_singletons

    @staticmethod
    def filter_cold_start(train_df, test_df):
        """
        Remove cold-start users/items từ test set

        Args:
            train_df, test_df: DataFrames -> userId, movieId

        Returns:
            test_df_warm: Test set only has warm users/items
        """
        train_users = set(train_df['userId'].unique())
        train_movies = set(train_df['movieId'].unique())

        test_warm = test_df[
            (test_df['userId'].isin(train_users)) &
            (test_df['movieId'].isin(train_movies))
            ].copy()

        removed = len(test_df) - len(test_warm)
        print(f"Removed {removed:,} cold-start interactions "
              f"({removed / len(test_df) * 100:.2f}% of test)")

        return test_warm

    @staticmethod
    def to_implicit(ratings_df, threshold=4.0):
        """
        Convert explicit ratings to implicit feedback

        Args:
            ratings_df: DataFrame rating column
            threshold: Rating >= threshold = positive interaction

        Returns:
            DataFrame with implicit interactions only
        """
        implicit = ratings_df[ratings_df['rating'] >= threshold].copy()
        print(f"Implicit conversion: {len(implicit):,} / {len(ratings_df):,} "
              f"({len(implicit) / len(ratings_df) * 100:.1f}%) positive interactions")
        return implicit

    @staticmethod
    def build_user_item_matrix(interactions_df, value_column=None, confidence_alpha=None):
        """Build a user-by-item CSR matrix.

        Exactly ONE of ``value_column`` / ``confidence_alpha`` may be supplied
        (passing both raises ``ValueError``).

        Three branches:

        * ``value_column="rating"`` — preserve explicit values (1.0–5.0),
          used for Popularity, User-CF (weighted cosine) and Item-CF
          (adjusted cosine requires mean-centering by rating).
        * ``confidence_alpha=1.0`` — cell = ``1 + α · rating`` (Hu/Koren/Volinsky
          2008), used for Implicit ALS.
        * Both ``None``            — every cell = 1.0 (binary "seen"), used for
          any model that only needs presence (e.g. BPR).
        """
        if value_column is not None and confidence_alpha is not None:
            raise ValueError("Use value_column or confidence_alpha, not both")
        if value_column is not None and value_column not in interactions_df:
            raise KeyError(f"Missing value column: {value_column}")
        if confidence_alpha is not None and 'rating' not in interactions_df:
            raise KeyError("confidence_alpha requires a rating column")

        user_ids = np.sort(interactions_df['userId'].unique())
        movie_ids = np.sort(interactions_df['movieId'].unique())
        user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}
        df = interactions_df.copy()
        df['user_idx'] = df['userId'].map(user_id_map)
        df['movie_idx'] = df['movieId'].map(movie_id_map)

        if confidence_alpha is not None:
            data = (1.0 + confidence_alpha * df['rating']).astype(np.float32)
        elif value_column is not None:
            data = df[value_column].astype(np.float32)
        else:
            data = np.ones(len(df), dtype=np.float32)

        sparse_matrix = csr_matrix(
            (data, (df['user_idx'], df['movie_idx'])),
            shape=(len(user_ids), len(movie_ids)), dtype=np.float32
        )

        sparsity = 100 * sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
        print(f"User-item matrix: {sparse_matrix.shape}, "
              f"sparsity: {sparsity:.4f}%, "
              f"nnz: {sparse_matrix.nnz:,}")

        return sparse_matrix, user_id_map, movie_id_map


def save_processed_data(train, test, save_dir='data/processed'):
    """Save processed train/test splits"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(save_dir / 'train_ml25m.csv', index=False)
    test.to_csv(save_dir / 'test_ml25m.csv', index=False)

    print(f"Saved to {save_dir}:")
    print(f"  - train_ml25m.csv: {len(train):,} rows")
    print(f"  - test_ml25m.csv: {len(test):,} rows")
