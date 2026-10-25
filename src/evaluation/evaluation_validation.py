"""Validation utilities for evaluation datasets and metrics.

This module provides validation functions to ensure data quality and
metric correctness during the evaluation process.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def validate_dataset_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    expected_ratio: float = 0.7,
    tolerance: float = 0.02
) -> None:
    """
    Validate train/test split is correct.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        expected_ratio: Expected train ratio (default: 0.7 for 70/30 split)
        tolerance: Acceptable deviation from expected ratio (default: 0.02)

    Raises:
        AssertionError: If validation fails
    """
    total = len(train_df) + len(test_df)
    actual_ratio = len(train_df) / total

    assert abs(actual_ratio - expected_ratio) < tolerance, (
        f"Split ratio {actual_ratio:.3f} differs from expected {expected_ratio:.3f}"
    )

    print(f"✓ Dataset split validated: {actual_ratio:.1%} train")

    # Check cold-start filtering
    train_users = set(train_df['userId'].unique())
    train_items = set(train_df['movieId'].unique())

    test_cold_users = set(test_df['userId'].unique()) - train_users
    test_cold_items = set(test_df['movieId'].unique()) - train_items

    if test_cold_users:
        print(f"⚠ Warning: {len(test_cold_users)} cold users in test")
    if test_cold_items:
        print(f"⚠ Warning: {len(test_cold_items)} cold items in test")

    if not test_cold_users and not test_cold_items:
        print("✓ No cold-start users/items in test")


def validate_metrics(
    results: Dict[str, Any],
    min_users: int = 1000
) -> None:
    """
    Validate evaluation results are reasonable.

    Args:
        results: Result dict from evaluate_model()
        min_users: Minimum users for statistical validity (default: 1000)

    Raises:
        AssertionError: If validation fails
    """
    metrics = results['metrics']

    # Check sufficient users
    assert results['num_users'] >= min_users, (
        f"Too few users: {results['num_users']} < {min_users}"
    )
    print(f"✓ Sufficient users: {results['num_users']:,}")

    # Check metric ranges [0, 1]
    for metric_name, value in metrics.items():
        if not metric_name.endswith('_std') and not metric_name.endswith('_median'):
            assert 0.0 <= value <= 1.0, f"Metric {metric_name}={value} out of range [0,1]"
    print("✓ All metrics in valid range [0, 1]")

    # Check monotonic properties
    k_values = [5, 10, 20]

    # Precision typically decreases with K
    p_values = [metrics[f'Precision@{k}'] for k in k_values]
    if not all(p_values[i] >= p_values[i+1] - 0.01 for i in range(len(p_values)-1)):
        print("⚠ Warning: Precision not monotonically decreasing")

    # Recall typically increases with K
    r_values = [metrics[f'Recall@{k}'] for k in k_values]
    if not all(r_values[i] <= r_values[i+1] + 0.01 for i in range(len(r_values)-1)):
        print("⚠ Warning: Recall not monotonically increasing")

    print("✓ Metrics validation passed")


def check_data_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Check for potential data leakage between train and test sets.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
    """
    # Create interaction tuples (userId, movieId)
    train_interactions = set(
        zip(train_df['userId'].values, train_df['movieId'].values)
    )
    test_interactions = set(
        zip(test_df['userId'].values, test_df['movieId'].values)
    )

    # Check for overlap
    overlap = train_interactions & test_interactions

    if overlap:
        print(f"WARNING: {len(overlap)} identical interactions in train/test!")
        print(f"  This indicates DATA LEAKAGE - results will be inflated")
        print(f"  Example leaked interactions: {list(overlap)[:5]}")
    else:
        print("✓ No identical interactions in train/test")
        print("✓ No data leakage detected")


def validate_positive_threshold(
    ratings_df: pd.DataFrame,
    positive_threshold: float = 4.0
) -> None:
    """
    Validate positive threshold produces reasonable positive ratio.

    Args:
        ratings_df: DataFrame with rating column
        positive_threshold: Rating threshold for positive interactions
    """
    total_ratings = len(ratings_df)
    positive_ratings = len(ratings_df[ratings_df['rating'] >= positive_threshold])
    positive_ratio = positive_ratings / total_ratings

    print(f"Positive ratio: {positive_ratio:.1%} (ratings >= {positive_threshold})")

    # Warn if ratio is too extreme
    if positive_ratio < 0.1:
        print(f"Warning: Very few positive interactions ({positive_ratio:.1%})")
        print(f"  Consider lowering threshold below {positive_threshold}")
    elif positive_ratio > 0.9:
        print(f"Warning: Almost all interactions are positive ({positive_ratio:.1%})")
        print(f"  Consider raising threshold above {positive_threshold}")
    else:
        print("✓ Positive threshold is reasonable")
