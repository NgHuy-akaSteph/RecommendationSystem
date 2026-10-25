"""Export utilities for evaluation results and configurations.

This module provides functions to save experiment configurations and
export evaluation results in multiple formats (CSV, JSON, Markdown).
"""

import json
import pandas as pd
import numpy as np
import scipy
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def save_experiment_config(
    output_path: str,
    train_ratio: float = 0.7,
    positive_threshold: float = 4.0,
    k_values: List[int] = None,
    model_params: Dict[str, Any] = None
) -> None:
    """
    Save experiment configuration for reproducibility.

    Args:
        output_path: Path to save JSON config
        train_ratio: Train/test split ratio
        positive_threshold: Threshold for positive ratings
        k_values: K values for evaluation
        model_params: Dict of model hyperparameters

    Example:
        >>> save_experiment_config(
        ...     'results/config.json',
        ...     train_ratio=0.7,
        ...     model_params={'als': {'factors': 100}}
        ... )
        Configuration saved to: results/config.json
    """
    if k_values is None:
        k_values = [5, 10, 20]

    try:
        import implicit
        implicit_version = implicit.__version__
    except ImportError:
        implicit_version = "N/A"

    config = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': 42,
        'dataset': 'MovieLens-25M',
        'evaluation': {
            'train_ratio': train_ratio,
            'positive_threshold': positive_threshold,
            'k_values': k_values,
            'metrics': ['Precision@K', 'Recall@K', 'NDCG@K']
        },
        'packages': {
            'python': '3.11',
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'implicit': implicit_version
        },
        'model_params': model_params or {}
    }

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {output_path}")


def export_results(
    results_df: pd.DataFrame,
    output_dir: str = 'results',
    experiment_name: str = 'evaluation'
) -> None:
    """
    Export evaluation results to multiple formats.

    Args:
        results_df: DataFrame with model comparison
        output_dir: Directory to save results
        experiment_name: Name prefix for files

    Example:
        >>> comparison_df = pd.DataFrame([...])
        >>> export_results(comparison_df, output_dir='results', experiment_name='popularity_baselines')
        CSV saved: results/popularity_baselines_20260910_160530.csv
        JSON saved: results/popularity_baselines_20260910_160530.json
        Markdown saved: results/popularity_baselines_20260910_160530.md
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # CSV export
    csv_path = output_dir / f'{experiment_name}_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # JSON export
    json_path = output_dir / f'{experiment_name}_{timestamp}.json'
    results_df.to_json(json_path, orient='records', indent=2)
    print(f"JSON saved: {json_path}")

    # Markdown table
    md_path = output_dir / f'{experiment_name}_{timestamp}.md'
    with open(md_path, 'w') as f:
        f.write(f"# Evaluation Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(results_df.to_markdown(index=False))
    print(f"Markdown saved: {md_path}")


def format_results_for_comparison(
    results_list: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Format multiple model results into comparison DataFrame.

    Args:
        results_list: List of result dicts from evaluate_model()

    Returns:
        DataFrame with model comparison, sorted by NDCG@10

    Example:
        >>> results = [pop_results, usercf_results, als_results]
        >>> comparison_df = format_results_for_comparison(results)
        >>> print(comparison_df[['Model', 'NDCG@10', 'Precision@10']])
    """
    rows = []

    for result in results_list:
        row = {
            'Model': result['model_name'],
            'N_Users': result['num_users'],
            'Failed': result['num_failed']
        }

        # Add all metrics (mean values only, not std/median)
        for metric_name, value in result['metrics'].items():
            if not metric_name.endswith('_std') and not metric_name.endswith('_median'):
                row[metric_name] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by NDCG@10 (primary metric)
    if 'NDCG@10' in df.columns:
        df = df.sort_values('NDCG@10', ascending=False).reset_index(drop=True)

    return df


def save_per_user_metrics(
    result: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save detailed per-user metrics to CSV.

    Args:
        result: Result dict from evaluate_model()
        output_path: Path to save CSV file

    Example:
        >>> save_per_user_metrics(als_results, 'results/als_per_user.csv')
        Per-user metrics saved: results/als_per_user.csv (145234 users)
    """
    user_metrics_df = pd.DataFrame(result['user_metrics'])

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    user_metrics_df.to_csv(output_path, index=False)
    print(f"Per-user metrics saved: {output_path} ({len(user_metrics_df)} users)")
