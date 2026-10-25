"""Recommender systems implementations.

This module provides various recommender algorithms including:
- ItemCFRecommender: Item-based collaborative filtering
- ALSRecommender: Implicit Alternating Least Squares matrix factorization
- BPRRecommender: Bayesian Personalized Ranking for implicit feedback
- PopularityRecommender: Popularity-based (count)
- BayesianPopularityRecommender: Bayesian weighted popularity
- WilsonLowerBoundRecommender: Wilson score confidence interval
"""

from .item_cf import ItemCFRecommender
from .als_recommender import ALSRecommender
from .bpr_recommender import BPRRecommender
from .popularity import (
    PopularityRecommender,
    BayesianPopularityRecommender,
    WilsonLowerBoundRecommender,
)

__all__ = [
    "ItemCFRecommender",
    "ALSRecommender",
    "BPRRecommender",
    "PopularityRecommender",
    "BayesianPopularityRecommender",
    "WilsonLowerBoundRecommender",
]

__version__ = "0.1.0"
