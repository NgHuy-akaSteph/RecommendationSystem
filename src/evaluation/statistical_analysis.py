"""Statistical analysis tools for model comparison.

This module provides functions for statistical significance testing and
confidence interval computation for comparing recommender system models.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List


def compute_statistical_significance(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
    metric: str = 'NDCG@10',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test if difference between two models is statistically significant.

    Uses paired t-test (same users evaluated for both models).

    Args:
        result_a: Result dict from model A (from evaluate_model())
        result_b: Result dict from model B (from evaluate_model())
        metric: Metric to compare (e.g., 'NDCG@10', 'Precision@10')
        alpha: Significance level (default: 0.05)

    Returns:
        Dict with:
        - 't_statistic': float
        - 'p_value': float
        - 'significant': bool (p < alpha)
        - 'mean_diff': float (A - B)
        - 'mean_a': float
        - 'mean_b': float
        - 'model_a': str
        - 'model_b': str

    Example:
        >>> sig = compute_statistical_significance(als_results, usercf_results)
        >>> if sig['significant']:
        ...     print(f"ALS significantly better than UserCF (p={sig['p_value']:.4f})")

    Note:
        - Paired t-test assumes both models evaluated on same users
        - p_value < 0.05 typically considered significant
        - Positive mean_diff means model A is better
    """
    # Extract per-user metrics
    values_a = [u[metric] for u in result_a['user_metrics']]
    values_b = [u[metric] for u in result_b['user_metrics']]

    if len(values_a) != len(values_b):
        raise ValueError(
            f"Models evaluated on different number of users: "
            f"{len(values_a)} vs {len(values_b)}"
        )

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values_a, values_b)

    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'mean_diff': float(mean_a - mean_b),
        'mean_a': float(mean_a),
        'mean_b': float(mean_b),
        'model_a': result_a['model_name'],
        'model_b': result_b['model_name'],
        'metric': metric,
        'alpha': alpha
    }


def compute_confidence_intervals(
    result: Dict[str, Any],
    metric: str = 'NDCG@10',
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for a metric.

    Args:
        result: Result dict from evaluate_model()
        metric: Metric name (e.g., 'NDCG@10', 'Precision@10')
        confidence: Confidence level (default: 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        Dict with:
        - 'mean': float - observed mean
        - 'lower': float - lower bound (e.g., 2.5th percentile for 95% CI)
        - 'upper': float - upper bound (e.g., 97.5th percentile for 95% CI)
        - 'confidence': float - confidence level
        - 'n_bootstrap': int - number of bootstrap samples

    Example:
        >>> ci = compute_confidence_intervals(als_results, metric='NDCG@10')
        >>> print(f"NDCG@10: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    Note:
        - Bootstrap resampling provides non-parametric confidence intervals
        - Useful when distribution is unknown or non-normal
        - CI not containing 0 suggests significant difference from baseline
    """
    values = [u[metric] for u in result['user_metrics']]

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return {
        'mean': float(np.mean(values)),
        'lower': float(np.percentile(bootstrap_means, lower_percentile)),
        'upper': float(np.percentile(bootstrap_means, upper_percentile)),
        'confidence': confidence,
        'n_bootstrap': n_bootstrap,
        'metric': metric,
        'model': result['model_name']
    }


def compare_multiple_models(
    results: List[Dict[str, Any]],
    metric: str = 'NDCG@10',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform pairwise statistical comparison of multiple models.

    Args:
        results: List of result dicts from evaluate_model()
        metric: Metric to compare
        alpha: Significance level

    Returns:
        Dict with:
        - 'comparisons': List of pairwise comparison results
        - 'ranking': List of model names sorted by metric (best first)
        - 'best_model': str - name of best performing model

    Example:
        >>> results = [pop_results, usercf_results, als_results]
        >>> comparison = compare_multiple_models(results, metric='NDCG@10')
        >>> print(f"Best model: {comparison['best_model']}")
        >>> for comp in comparison['comparisons']:
        ...     if comp['significant']:
        ...         print(f"{comp['model_a']} > {comp['model_b']} (p={comp['p_value']:.4f})")
    """
    if len(results) < 2:
        raise ValueError("Need at least 2 models to compare")

    # Perform all pairwise comparisons
    comparisons = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            comp = compute_statistical_significance(
                results[i], results[j], metric=metric, alpha=alpha
            )
            comparisons.append(comp)

    # Rank models by mean metric
    ranking = sorted(
        results,
        key=lambda r: np.mean([u[metric] for u in r['user_metrics']]),
        reverse=True
    )

    return {
        'comparisons': comparisons,
        'ranking': [r['model_name'] for r in ranking],
        'best_model': ranking[0]['model_name'],
        'metric': metric,
        'alpha': alpha
    }
