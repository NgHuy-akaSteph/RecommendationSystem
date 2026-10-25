"""Export trained models to pickle files for deployment.

This script trains and exports the best models:
- Item-based Collaborative Filtering
- ALS (Alternating Least Squares)
- Bayesian Popularity baseline

Usage:
    python scripts/export_models.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data_loader import DataPreprocessor
from src.recommenders.item_cf import ItemCFRecommender
from src.recommenders.als_recommender import ALSRecommender
from src.recommenders import BayesianPopularityRecommender


def main():
    print("="*80)
    print("Model Export Script")
    print("="*80)
    
    # Create output directory
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_csv(PROJECT_ROOT / 'data/processed/train_ml25m.csv')
    print(f"   Loaded {len(train_df):,} training ratings")
    
    preprocessor = DataPreprocessor()
    
    # Train and export Item-CF
    print("\n2. Training Item-CF...")
    user_item_csr, user_map, movie_map = preprocessor.build_user_item_matrix(
        train_df, value_column='rating'
    )
    item_cf = ItemCFRecommender()
    item_cf.fit(user_item_csr, user_map, movie_map)
    
    output_path = models_dir / 'item_cf_best.pkl'
    item_cf.save(str(output_path))
    print(f"   ✓ Exported: {output_path}")
    
    # Train and export ALS
    print("\n3. Training ALS...")
    train_implicit = preprocessor.to_implicit(train_df, threshold=4.0)
    als = ALSRecommender(
        factors=64,
        regularization=0.05,
        iterations=15,
        alpha=1.0
    )
    als.fit(train_implicit)
    
    output_path = models_dir / 'als_best.pkl'
    als.save(str(output_path))
    print(f"   ✓ Exported: {output_path}")
    
    # Train and export Popularity baseline
    print("\n4. Training Bayesian Popularity...")
    pop = BayesianPopularityRecommender()
    pop.fit(train_df)
    
    output_path = models_dir / 'popularity_weighted.pkl'
    pop.save(str(output_path))
    print(f"   ✓ Exported: {output_path}")
    
    print("\n" + "="*80)
    print("Export complete!")
    print("\nTo load models:")
    print("  from src.recommenders import ItemCFRecommender, ALSRecommender")
    print("  item_cf = ItemCFRecommender.load('models/item_cf_best.pkl')")
    print("  als = ALSRecommender.load('models/als_best.pkl')")
    print("="*80)


if __name__ == '__main__':
    main()
