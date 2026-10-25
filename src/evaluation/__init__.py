"""Evaluation module for recommender systems.

This module provides a complete evaluation pipeline for recommender systems,
including metrics computation, ground truth preparation, and result aggregation.

Main components:
- RecommenderEvaluator: Compute Precision@K, Recall@K, NDCG@K metrics
- EvaluationPipeline: Orchestrate evaluation across users and models
- Recommender: Protocol interface for model compatibility
"""

from .recommender_evaluator import RecommenderEvaluator
from .evaluation_pipeline import EvaluationPipeline
from .evaluation_protocols import Recommender

__all__ = [
    "RecommenderEvaluator",
    "EvaluationPipeline",
    "Recommender",
]

__version__ = "0.1.0"
