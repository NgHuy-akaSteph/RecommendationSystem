"""Recommender system metrics calculator.

This module provides the RecommenderEvaluator class for computing standard
recommender system metrics including Precision@K, Recall@K, NDCG@K, and Hit Rate@K.
"""

from math import log2
from typing import List, Set, Dict


class RecommenderEvaluator:
    """
    Evaluator cho recommender systems với Precision@K, Recall@K, NDCG@K, HR@K.

    Attributes:
        k_values: List of K values to evaluate. Default: [5, 10, 20]
    """

    def __init__(self, k_values: List[int] = None):
        """
        Initialize evaluator.

        Args:
            k_values: List of K values to evaluate. Default: [5, 10, 20]
        """
        self.k_values = k_values if k_values is not None else [5, 10, 20]

        # Validate K values
        if not self.k_values:
            raise ValueError("k_values must not be empty")
        if any(k <= 0 for k in self.k_values):
            raise ValueError("All K values must be positive integers")

        # Sort K values for consistent output
        self.k_values = sorted(self.k_values)

    @staticmethod
    def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Compute Precision@K.

        Precision@K = (# relevant items in top-K) / K

        Args:
            recommended: List of recommended item IDs (ordered by score)
            relevant: Set of relevant item IDs (ground truth)
            k: Cutoff position

        Returns:
            Precision score in [0, 1]

        Edge Cases:
            - k > len(recommended): Use len(recommended) as denominator
            - k <= 0: Return 0.0
            - empty recommended: Return 0.0
            - empty relevant: Return 0.0
        """
        if k <= 0:
            return 0.0

        if not recommended or not relevant:
            return 0.0

        # Take top-K recommendations
        top_k = recommended[:k]

        # Count hits
        hits = len(set(top_k) & relevant)

        # Precision = hits / K (or len(top_k) if fewer than K items)
        return hits / len(top_k)

    @staticmethod
    def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Compute Recall@K.

        Recall@K = (# relevant items in top-K) / (total # relevant items)

        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k: Cutoff position

        Returns:
            Recall score in [0, 1]

        Edge Cases:
            - k <= 0: Return 0.0
            - empty recommended: Return 0.0
            - empty relevant: Return 0.0 (avoid division by zero)
        """
        if k <= 0:
            return 0.0

        if not recommended or not relevant:
            return 0.0

        # Take top-K
        top_k = recommended[:k]

        # Count hits
        hits = len(set(top_k) & relevant)

        # Recall = hits / total relevant
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Compute NDCG@K (Normalized Discounted Cumulative Gain).

        DCG@K = Σ(i=1 to K) [rel_i / log2(i+1)]
        IDCG@K = DCG@K for perfect ranking (all relevant items at top)
        NDCG@K = DCG@K / IDCG@K

        Discount factor: 1 / log2(position + 2)
        - Position 0 (rank 1): discount = 1 / log2(2) = 1.0
        - Position 1 (rank 2): discount = 1 / log2(3) ≈ 0.631
        - Position 4 (rank 5): discount = 1 / log2(6) ≈ 0.387

        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k: Cutoff position

        Returns:
            NDCG score in [0, 1]

        Edge Cases:
            - k <= 0: Return 0.0
            - empty recommended: Return 0.0
            - empty relevant: Return 0.0
            - no relevant items in top-K: DCG=0, NDCG=0
        """
        if k <= 0:
            return 0.0

        if not recommended or not relevant:
            return 0.0

        # Compute DCG@K
        dcg = 0.0
        for i, item_id in enumerate(recommended[:k]):
            if item_id in relevant:
                # Position i (0-indexed) → rank i+1
                # Discount: 1 / log2(rank + 1) = 1 / log2(i + 2)
                dcg += 1.0 / log2(i + 2)

        # Compute IDCG@K (ideal DCG with perfect ranking)
        # Perfect ranking: all relevant items at top positions
        idcg = 0.0
        num_relevant_in_topk = min(len(relevant), k)
        for i in range(num_relevant_in_topk):
            idcg += 1.0 / log2(i + 2)

        # Avoid division by zero
        if idcg == 0.0:
            return 0.0

        # NDCG = DCG / IDCG
        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Compute Hit Rate@K (binary indicator).

        Hit Rate@K = 1.0 if at least one relevant item in top-K, else 0.0

        This is a binary metric per user. When averaged across users,
        it represents the fraction of users who got at least one hit.

        Args:
            recommended: List of recommended item IDs (ordered by score)
            relevant: Set of relevant item IDs (ground truth)
            k: Cutoff position

        Returns:
            1.0 if hit, 0.0 if no hit

        Edge Cases:
            - k <= 0: Return 0.0
            - empty recommended: Return 0.0
            - empty relevant: Return 0.0
        """
        if k <= 0:
            return 0.0

        if not recommended or not relevant:
            return 0.0

        # Take top-K recommendations
        top_k = recommended[:k]

        # Check if any item in top-K is relevant
        return 1.0 if any(item in relevant for item in top_k) else 0.0

    def evaluate_user(
            self,
            recommended: List[int],
            relevant: Set[int]
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user across all K values.

        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs

        Returns:
            Dict with metrics for each K value:
            {
                'Precision@5': 0.4,
                'Recall@5': 0.2,
                'NDCG@5': 0.65,
                'Precision@10': 0.3,
                ...
            }
        """
        metrics = {}

        for k in self.k_values:
            metrics[f'Precision@{k}'] = self.precision_at_k(recommended, relevant, k)
            metrics[f'Recall@{k}'] = self.recall_at_k(recommended, relevant, k)
            metrics[f'NDCG@{k}'] = self.ndcg_at_k(recommended, relevant, k)

        return metrics
