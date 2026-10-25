import pandas as pd
import pickle
from pathlib import Path


class _RankedRecommender:
    """Shared ranking state and seen-item filtering for popularity models."""

    def __init__(self) -> None:
        self.ranked_items_: list[int] | None = None
        self.scores_: dict[int, float] | None = None
        # Giữ tên cũ để tương thích hướng dẫn triển khai trước đó.
        self.popular_items: list[int] | None = None
        self.score_by_item: dict[int, float] | None = None

    def _set_ranking(self, scores: pd.Series) -> None:
        ranking = (
            scores.rename("score")
            .rename_axis("movieId")
            .reset_index()
            .sort_values(
                ["score", "movieId"],
                ascending=[False, True],
                kind="mergesort",
            )
        )
        self.ranked_items_ = [
            int(movie_id) for movie_id in ranking["movieId"]
        ]
        self.scores_ = {
            int(movie_id): float(score)
            for movie_id, score in scores.items()
        }
        self.popular_items = self.ranked_items_
        self.score_by_item = self.scores_

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        seen: set[int] | None = None,
    ) -> list[int]:
        """Trả tối đa ``n`` item xếp hạng toàn cục, loại bỏ item đã xem."""
        if self.ranked_items_ is None:
            raise RuntimeError("Call fit() first")
        if n <= 0:
            return []
        if not seen:
            return self.ranked_items_[:n]
        return [
            movie_id
            for movie_id in self.ranked_items_
            if movie_id not in seen
        ][:n]

    def save(self, path: str):
        """Save model to pickle file.
        
        Args:
            path: Output file path (e.g., 'models/popularity.pkl')
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'ranked_items_': self.ranked_items_,
                'scores_': self.scores_,
            }, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model from pickle file.
        
        Args:
            path: Input file path
        
        Returns:
            Loaded recommender instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls()
        obj.ranked_items_ = data['ranked_items_']
        obj.scores_ = data['scores_']
        obj.popular_items = obj.ranked_items_
        obj.score_by_item = obj.scores_
        
        print(f"Model loaded from {path}")
        
        return obj


def _validate_ratings(train_df: pd.DataFrame) -> None:
    required_columns = {"movieId", "rating"}
    missing_columns = required_columns.difference(train_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"train_df is missing required columns: {missing}"
        )
    if train_df.empty:
        raise ValueError("train_df must not be empty")


def _validate_implicit(implicit_df: pd.DataFrame) -> None:
    required_columns = {"userId", "movieId"}
    missing_columns = required_columns.difference(implicit_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"implicit_df is missing required columns: {missing}"
        )
    if implicit_df.empty:
        raise ValueError("implicit_df must not be empty")


class PopularityRecommender(_RankedRecommender):
    """Xếp hạng item theo số lượt rating, loại bỏ item đã xem."""

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        _validate_ratings(train_df)
        counts = train_df.groupby("movieId").size().astype(float)
        self._set_ranking(counts)
        return self

class BayesianPopularityRecommender(_RankedRecommender):
    """IMDb-style weighted rating, co về global mean.

    Prior mặc định là mean số rating trên mỗi item (trong tập train).
    """

    def __init__(self, m: float | None = None) -> None:
        super().__init__()
        if m is not None and m <= 0:
            raise ValueError("m must be greater than zero")
        self.m = m

    def fit(self, train_df: pd.DataFrame) -> "BayesianPopularityRecommender":
        _validate_ratings(train_df)
        item_stats = train_df.groupby("movieId")["rating"].agg(
            ["count", "mean"]
        )
        votes = item_stats["count"].astype(float)
        item_mean = item_stats["mean"].astype(float)
        global_mean = float(train_df["rating"].mean())
        prior_strength = (
            float(self.m) if self.m is not None else float(votes.mean())
        )
        scores = (votes / (votes + prior_strength)) * item_mean
        scores += (prior_strength / (votes + prior_strength)) * global_mean
        self._set_ranking(scores)
        return self

    def save(self, path: str):
        """Save model with hyperparameter m."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'ranked_items_': self.ranked_items_,
                'scores_': self.scores_,
                'm': self.m,
            }, f)
        
        print(f"BayesianPopularity model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BayesianPopularityRecommender":
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(m=data['m'])
        obj.ranked_items_ = data['ranked_items_']
        obj.scores_ = data['scores_']
        obj.popular_items = obj.ranked_items_
        obj.score_by_item = obj.scores_
        
        print(f"BayesianPopularity model loaded from {path}")
        
        return obj

class WilsonLowerBoundRecommender(_RankedRecommender):
    """Wilson 95% lower bound của tỉ lệ rating dương (`rating >= 4`)."""

    def __init__(self, z: float = 1.96) -> None:
        super().__init__()
        if z <= 0:
            raise ValueError("z must be greater than zero")
        self.z = float(z)

    def fit(self, train_df: pd.DataFrame) -> "WilsonLowerBoundRecommender":
        _validate_ratings(train_df)
        votes = train_df.groupby("movieId").size().astype(float)
        positives = (
            (train_df["rating"] >= 4)
            .groupby(train_df["movieId"])
            .sum()
        )
        positive_rate = (
            positives.reindex(votes.index, fill_value=0).astype(float)
            / votes
        )
        z_squared = self.z ** 2
        denominator = 1.0 + z_squared / votes
        centre = positive_rate + z_squared / (2.0 * votes)
        margin = self.z * (
            positive_rate * (1.0 - positive_rate) / votes
            + z_squared / (4.0 * votes ** 2)
        ).pow(0.5)
        scores = (
            (centre - margin) / denominator
        ).clip(lower=0.0, upper=1.0)
        self._set_ranking(scores)
        return self

    def save(self, path: str):
        """Save model with hyperparameter z."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'ranked_items_': self.ranked_items_,
                'scores_': self.scores_,
                'z': self.z,
            }, f)
        
        print(f"WilsonLowerBound model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "WilsonLowerBoundRecommender":
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(z=data['z'])
        obj.ranked_items_ = data['ranked_items_']
        obj.scores_ = data['scores_']
        obj.popular_items = obj.ranked_items_
        obj.score_by_item = obj.scores_

        print(f"WilsonLowerBound model loaded from {path}")

        return obj


class ImplicitPopularityRecommender(_RankedRecommender):
    """Xếp hạng item theo số lượt implicit interaction, loại bỏ item đã xem.

    Đầu vào là ``implicit_df`` đã được lọc bởi ``DataPreprocessor.to_implicit``
    (chỉ còn positive interaction, ``rating >= threshold``) — đồng bộ với
    pipeline ALS/BPR. Cột ``rating`` là tuỳ chọn: nếu có, chỉ dùng để đếm
    số row (confidence bằng 1 cho mỗi row); nếu không có thì đếm theo
    ``userId, movieId``.
    """

    def fit(self, implicit_df: pd.DataFrame) -> "ImplicitPopularityRecommender":
        _validate_implicit(implicit_df)
        counts = implicit_df.groupby("movieId").size().astype(float)
        self._set_ranking(counts)
        return self


class ImplicitBayesianPopularityRecommender(_RankedRecommender):
    """Bayesian weighted rating trên implicit data, có về global mean.

    - Nếu ``implicit_df`` có cột ``rating``: score là IMDb-style Bayesian mean
      của rating, với prior là mean rating toàn cục. Trọng số là số row
      (mỗi implicit interaction đóng góp 1 quan sát).
    - Nếu không có ``rating``: fallback về Bayesian smoothing trên interaction
      count (prior = mean count toàn cục) — tương đương smoothed popularity.
    Prior mặc định là mean số rating trên mỗi item (trong tập train).
    """

    def __init__(self, m: float | None = None) -> None:
        super().__init__()
        if m is not None and m <= 0:
            raise ValueError("m must be greater than zero")
        self.m = m

    def fit(self, implicit_df: pd.DataFrame) -> "ImplicitBayesianPopularityRecommender":
        _validate_implicit(implicit_df)
        counts = implicit_df.groupby("movieId").size().astype(float)
        if "rating" in implicit_df.columns:
            item_stats = implicit_df.groupby("movieId")["rating"].agg(
                ["count", "mean"]
            )
            votes = item_stats["count"].astype(float)
            item_mean = item_stats["mean"].astype(float)
            global_mean = float(implicit_df["rating"].mean())
        else:
            votes = counts
            item_mean = counts.copy()
            global_mean = float(votes.mean())
        prior_strength = (
            float(self.m) if self.m is not None else float(votes.mean())
        )
        scores = (votes / (votes + prior_strength)) * item_mean
        scores += (prior_strength / (votes + prior_strength)) * global_mean
        self._set_ranking(scores)
        return self

    def save(self, path: str):
        """Save model with hyperparameter m."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'ranked_items_': self.ranked_items_,
                'scores_': self.scores_,
                'm': self.m,
            }, f)

        print(f"ImplicitBayesianPopularity model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ImplicitBayesianPopularityRecommender":
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        obj = cls(m=data['m'])
        obj.ranked_items_ = data['ranked_items_']
        obj.scores_ = data['scores_']
        obj.popular_items = obj.ranked_items_
        obj.score_by_item = obj.scores_

        print(f"ImplicitBayesianPopularity model loaded from {path}")

        return obj


class ImplicitWilsonLowerBoundRecommender(_RankedRecommender):
    """Wilson 95% lower bound của tỉ lệ positive interaction trên implicit data.

    - Nếu ``implicit_df`` có cột ``rating``: positives = số row có
      ``rating >= 4.0`` (khớp với ngưỡng của ``DataPreprocessor.to_implicit``),
      N = tổng số row trên item.
    - Nếu không có ``rating`` (pure binary implicit): fallback về
      user-coverage Wilson — positives = số user đã tương tác với item,
      N = tổng số user trong tập implicit.
    """

    def __init__(self, z: float = 1.96) -> None:
        super().__init__()
        if z <= 0:
            raise ValueError("z must be greater than zero")
        self.z = float(z)

    def fit(self, implicit_df: pd.DataFrame) -> "ImplicitWilsonLowerBoundRecommender":
        _validate_implicit(implicit_df)
        if "rating" in implicit_df.columns:
            votes = implicit_df.groupby("movieId").size().astype(float)
            positives = (
                (implicit_df["rating"] >= 4.0)
                .groupby(implicit_df["movieId"])
                .sum()
                .astype(float)
            )
        else:
            user_per_item = implicit_df.groupby("movieId")["userId"].nunique().astype(float)
            votes = user_per_item.copy()
            positives = user_per_item.copy()
        positive_rate = (
            positives.reindex(votes.index, fill_value=0).astype(float)
            / votes
        )
        z_squared = self.z ** 2
        denominator = 1.0 + z_squared / votes
        centre = positive_rate + z_squared / (2.0 * votes)
        margin = self.z * (
            positive_rate * (1.0 - positive_rate) / votes
            + z_squared / (4.0 * votes ** 2)
        ).pow(0.5)
        scores = (
            (centre - margin) / denominator
        ).clip(lower=0.0, upper=1.0)
        self._set_ranking(scores)
        return self

    def save(self, path: str):
        """Save model with hyperparameter z."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'ranked_items_': self.ranked_items_,
                'scores_': self.scores_,
                'z': self.z,
            }, f)

        print(f"ImplicitWilsonLowerBound model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ImplicitWilsonLowerBoundRecommender":
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        obj = cls(z=data['z'])
        obj.ranked_items_ = data['ranked_items_']
        obj.scores_ = data['scores_']
        obj.popular_items = obj.ranked_items_
        obj.score_by_item = obj.scores_

        print(f"ImplicitWilsonLowerBound model loaded from {path}")

        return obj

