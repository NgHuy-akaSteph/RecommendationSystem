"""Evaluation pipeline orchestrator for recommender systems.

This module provides the EvaluationPipeline class that orchestrates the entire
evaluation process: ground truth construction, seen items extraction, per-user
evaluation, and metrics aggregation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, List, Any
from collections import defaultdict

from .recommender_evaluator import RecommenderEvaluator
from .evaluation_protocols import Recommender


class EvaluationPipeline:
    """
    Pipeline for evaluating recommender systems.

    Handles:
    - Ground truth construction (ratings >= threshold)
    - Seen items extraction (from train set)
    - Per-user evaluation
    - Aggregation across all test users

    Attributes:
        train_df: Training DataFrame with columns [userId, movieId, rating, ...]
        test_df: Test DataFrame with columns [userId, movieId, rating, ...]
        positive_threshold: Rating >= threshold is relevant (default: 4.0)
        evaluator: RecommenderEvaluator instance
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        positive_threshold: float = 4.0,
        k_values: List[int] = None
    ):
        """
        Initialize evaluation pipeline.

        Args:
            train_df: Training data
            test_df: Test data (warm users/items only)
            positive_threshold: Minimum rating to be considered relevant
            k_values: K values for metrics (default: [5, 10, 20])
        """
        # Validate inputs
        required_cols = {'userId', 'movieId', 'rating'}
        if not required_cols.issubset(train_df.columns):
            missing = required_cols - set(train_df.columns)
            raise ValueError(f"train_df missing columns: {missing}")
        if not required_cols.issubset(test_df.columns):
            missing = required_cols - set(test_df.columns)
            raise ValueError(f"test_df missing columns: {missing}")

        if train_df.empty:
            raise ValueError("train_df is empty")
        if test_df.empty:
            raise ValueError("test_df is empty")

        if positive_threshold < 0 or positive_threshold > 5:
            raise ValueError("positive_threshold must be in [0, 5]")

        self.train_df = train_df
        self.test_df = test_df
        self.positive_threshold = positive_threshold
        self.evaluator = RecommenderEvaluator(k_values=k_values)

        # Cache ground truth and seen items
        self._ground_truth = None
        self._seen_items = None

    def prepare_ground_truth(self) -> Dict[int, Set[int]]:
        """
        Build per-user ground truth from test set.

        Ground truth = movies rated >= positive_threshold in test set

        Returns:
            Dict mapping userId -> Set of relevant movieIds

        Example:
            Test data:
            userId | movieId | rating
            1      | 10      | 5.0
            1      | 20      | 3.0
            1      | 30      | 4.5
            2      | 10      | 4.0

            With threshold=4.0:
            {
                1: {10, 30},  # ratings 5.0, 4.5
                2: {10}       # rating 4.0
            }

        Note:
            - Users without any relevant items are EXCLUDED from ground truth
            - This ensures all evaluated users have at least 1 relevant item
        """
        if self._ground_truth is not None:
            return self._ground_truth

        ground_truth = {}

        # Filter test set to positive ratings only
        positive_test = self.test_df[
            self.test_df['rating'] >= self.positive_threshold
        ]

        # Group by user
        for user_id, group in positive_test.groupby('userId'):
            relevant_items = set(group['movieId'].astype(int))
            if relevant_items:  # Only include users with relevant items
                ground_truth[int(user_id)] = relevant_items

        self._ground_truth = ground_truth

        print(f"Ground truth prepared: {len(ground_truth):,} users with relevant items")
        print(f"Avg relevant items per user: {np.mean([len(v) for v in ground_truth.values()]):.2f}")

        return ground_truth

    def prepare_seen_items(self) -> Dict[int, Set[int]]:
        """
        Extract items seen by each user in training set.

        These items will be excluded from recommendations to ensure
        we only recommend unseen items.

        Returns:
            Dict mapping userId -> Set of seen movieIds

        Example:
            Train data:
            userId | movieId | rating
            1      | 5       | 4.0
            1      | 15      | 3.0
            2      | 5       | 5.0

            Result:
            {
                1: {5, 15},
                2: {5}
            }
        """
        if self._seen_items is not None:
            return self._seen_items

        seen_items = (
            self.train_df
            .groupby('userId')['movieId']
            .apply(lambda x: set(x.astype(int)))
            .to_dict()
        )

        # Convert keys to int
        seen_items = {int(k): v for k, v in seen_items.items()}

        self._seen_items = seen_items

        print(f"Seen items prepared: {len(seen_items):,} users")
        print(f"Avg seen items per user: {np.mean([len(v) for v in seen_items.values()]):.1f}")

        return seen_items

    def evaluate_model(
        self,
        model: Recommender,
        model_name: str = "Model",
        n_recommendations: int = 20,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single recommender model.

        Args:
            model: Recommender instance with recommend() method
            model_name: Model name for logging
            n_recommendations: Number of recommendations to generate (>= max(K))
            verbose: Print progress

        Returns:
            Dict with:
            - 'model_name': str
            - 'metrics': Dict[str, float] - aggregated metrics
            - 'user_metrics': List[Dict] - per-user metrics
            - 'num_users': int - number of evaluated users
            - 'num_failed': int - number of users with empty recommendations

        Process:
            For each user in ground truth:
            1. Get seen items from train
            2. Generate recommendations (exclude seen)
            3. Compute metrics vs ground truth
            4. Aggregate across all users
        """
        ground_truth = self.prepare_ground_truth()
        seen_items = self.prepare_seen_items()

        # Storage for per-user metrics
        all_user_metrics = []
        failed_users = 0

        # Evaluate each user
        for idx, (user_id, relevant) in enumerate(ground_truth.items()):
            if verbose and (idx + 1) % 1000 == 0:
                print(f"  Evaluated {idx + 1:,}/{len(ground_truth):,} users...")

            # Get seen items for this user
            seen = seen_items.get(user_id, set())

            # Generate recommendations
            try:
                recommended = model.recommend(
                    user_id=user_id,
                    n=n_recommendations,
                    seen=seen
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to recommend for user {user_id}: {e}")
                recommended = []

            # Handle empty recommendations
            if not recommended:
                failed_users += 1
                # Assign zero metrics
                user_metrics = {
                    f'{metric}@{k}': 0.0
                    for metric in ['Precision', 'Recall', 'NDCG']
                    for k in self.evaluator.k_values
                }
            else:
                # Validate recommendations
                if len(recommended) != len(set(recommended)):
                    raise ValueError(
                        f"Duplicate recommendations for user {user_id}: {recommended}"
                    )

                # Check no seen items in recommendations
                overlap = set(recommended) & seen
                if overlap:
                    raise ValueError(
                        f"Recommended seen items for user {user_id}: {overlap}"
                    )

                # Compute metrics
                user_metrics = self.evaluator.evaluate_user(recommended, relevant)

            # Store
            user_metrics['userId'] = user_id
            all_user_metrics.append(user_metrics)

        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_user_metrics)

        if verbose:
            print(f"\n{model_name} Evaluation Complete:")
            print(f"  Users evaluated: {len(all_user_metrics):,}")
            print(f"  Users with no recommendations: {failed_users:,}")
            for k in self.evaluator.k_values:
                print(f"  Precision@{k}: {aggregated[f'Precision@{k}']:.4f}")
                print(f"  Recall@{k}: {aggregated[f'Recall@{k}']:.4f}")
                print(f"  NDCG@{k}: {aggregated[f'NDCG@{k}']:.4f}")

        return {
            'model_name': model_name,
            'metrics': aggregated,
            'user_metrics': all_user_metrics,
            'num_users': len(all_user_metrics),
            'num_failed': failed_users
        }

    def _aggregate_metrics(
        self,
        user_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate per-user metrics to model-level metrics.

        Args:
            user_metrics: List of dicts with per-user metrics

        Returns:
            Dict with mean, std, median for each metric

        Example output:
            {
                'Precision@5_mean': 0.123,
                'Precision@5_std': 0.045,
                'Precision@5_median': 0.100,
                'Recall@5_mean': 0.089,
                ...
            }
        """
        aggregated = {}

        # Get all metric names (exclude 'userId')
        metric_names = [k for k in user_metrics[0].keys() if k != 'userId']

        for metric_name in metric_names:
            values = [u[metric_name] for u in user_metrics]

            aggregated[f'{metric_name}_mean'] = float(np.mean(values))
            aggregated[f'{metric_name}_std'] = float(np.std(values))
            aggregated[f'{metric_name}_median'] = float(np.median(values))

            # For convenience, also store mean with short name
            aggregated[metric_name] = float(np.mean(values))

        return aggregated
