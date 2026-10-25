"""
BPR Experiments Script - Test logic before notebook integration
This script can be run directly to verify BPR implementation
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ML25MLoader, DataPreprocessor
from src.recommenders import BPRRecommender, ALSRecommender

# Config
RANDOM_SEED = 42
POSITIVE_THRESHOLD = 4.0
np.random.seed(RANDOM_SEED)

print("✓ Dependencies imported successfully")


# =============================================================================
# Helper Functions
# =============================================================================

def temporal_leave_one_positive_out(
    ratings_df: pd.DataFrame, 
    threshold: float = 4.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out each user's final positive rating for BPR evaluation.
    
    Protocol:
    - Sort by (userId, timestamp)
    - Find last positive rating (rating >= threshold) per user
    - Require >=2 positive ratings in history
    - Split: everything before last positive → train, last positive → test
    
    Args:
        ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
        threshold: Minimum rating to consider as positive interaction
        
    Returns:
        train_bpr: All positive ratings before last positive per user
        test_positive: One positive test item per user
    """
    print(f"Original ratings: {len(ratings_df):,}")
    
    # Sort by user and time
    sorted_df = ratings_df.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    
    train_parts = []
    test_rows = []
    
    for user_id, user_ratings in sorted_df.groupby("userId", sort=False):
        # Find positive ratings
        positive_mask = user_ratings["rating"] >= threshold
        positive_indices = np.flatnonzero(positive_mask)
        
        # Need at least 2 positives (1 for train, 1 for test)
        if len(positive_indices) < 2:
            continue
        
        # Last positive index in this user's group
        test_idx = positive_indices[-1]
        
        # Everything before test item
        history = user_ratings.iloc[:test_idx].copy()
        
        # Check if history has at least 1 positive
        if (history["rating"] >= threshold).sum() == 0:
            continue
        
        # Only keep positive ratings in train
        train_parts.append(history[history["rating"] >= threshold])
        test_rows.append(user_ratings.iloc[test_idx])
    
    train_bpr = pd.concat(train_parts, ignore_index=True)
    test_positive = pd.DataFrame(test_rows).reset_index(drop=True)
    
    print(f"Train BPR: {len(train_bpr):,} positive interactions")
    print(f"Test positive: {len(test_positive):,} users")
    print(f"Unique users in train: {train_bpr['userId'].nunique():,}")
    print(f"Unique movies in train: {train_bpr['movieId'].nunique():,}")
    
    return train_bpr, test_positive


def build_ranking_candidates(
    train_positive: pd.DataFrame,
    test_positive: pd.DataFrame,
    all_movie_ids: np.ndarray,
    negatives_per_user: int = 99,
    random_seed: int = 42,
) -> dict[int, np.ndarray]:
    """
    Create sampled ranking candidates: 1 positive + N negatives per user.
    
    Protocol:
    - For each test user, take their held-out positive item
    - Sample N unseen items (not in train) as negatives
    - Return candidates array: [positive_item, negative_1, ..., negative_N]
    
    Args:
        train_positive: Training interactions
        test_positive: Test interactions (one per user)
        all_movie_ids: Full catalogue of movie IDs
        negatives_per_user: Number of negative samples (default: 99)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary: {user_id: candidates_array}
    """
    rng = np.random.default_rng(random_seed)
    
    # Build seen items per user from training data
    seen = train_positive.groupby("userId")["movieId"].agg(set).to_dict()
    
    # Full catalogue
    catalog = set(map(int, all_movie_ids))
    
    candidates = {}
    
    for row in test_positive.itertuples(index=False):
        user_id = int(row.userId)
        positive_item = int(row.movieId)
        
        # Items to exclude: seen in train + the test positive item
        excluded = seen[user_id] | {positive_item}
        
        # Pool of potential negatives
        pool = np.fromiter(catalog - excluded, dtype=np.int32)
        
        # Sample negatives (or take all if fewer than requested)
        sample_size = min(negatives_per_user, len(pool))
        negatives = rng.choice(pool, size=sample_size, replace=False)
        
        # Candidates: positive item first, then negatives
        candidates[user_id] = np.concatenate(([positive_item], negatives))
    
    print(f"Built ranking candidates for {len(candidates):,} users")
    print(f"Candidates per user: 1 positive + {negatives_per_user} negatives")
    
    return candidates


def evaluate_ranking_at_k(
    model,  # BPR or ALS model
    candidate_map: dict[int, np.ndarray],
    k: int = 10,
) -> dict[str, float]:
    """
    Evaluate model with HitRate@K and NDCG@K using sampled ranking protocol.
    
    Protocol:
    - For each user, rank their candidate set (1 positive + N negatives)
    - Find rank of the positive item (candidates[0])
    - HitRate@K: 1 if positive in top-K, else 0
    - NDCG@K: DCG with position discount if positive in top-K, else 0
    
    Args:
        model: Trained BPR or ALS recommender with .recommend() method
        candidate_map: {user_id: candidates_array} from build_ranking_candidates()
        k: Cutoff for evaluation metrics
        
    Returns:
        Dictionary with hit_rate, ndcg, and users count
    """
    hits = []
    ndcgs = []
    
    for user_id, candidates in candidate_map.items():
        # Get recommendations from model (returns list of movieIds)
        # Request enough to cover all candidates
        try:
            recommended = model.recommend(
                user_id=user_id,
                n=len(candidates),
                seen=None  # Model already filters train items
            )
        except Exception as e:
            print(f"Warning: Failed to recommend for user {user_id}: {e}")
            recommended = []
        
        if not recommended:
            hits.append(0.0)
            ndcgs.append(0.0)
            continue
        
        # Find rank of positive item (candidates[0])
        positive_item = candidates[0]
        
        # Filter recommendations to only those in candidate set
        candidate_set = set(candidates)
        filtered_recs = [mid for mid in recommended if mid in candidate_set]
        
        # Find position
        try:
            rank = filtered_recs.index(positive_item)
        except ValueError:
            rank = len(filtered_recs)  # Not found
        
        # Compute metrics
        if rank < k:
            hits.append(1.0)
            # NDCG discount: 1 / log2(rank + 2)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
    
    return {
        "hit_rate": float(np.mean(hits)),
        "ndcg": float(np.mean(ndcgs)),
        "users": len(hits),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("="*70)
    print("BPR vs ALS Experiment on MovieLens 25M")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    loader = ML25MLoader()
    ratings = loader.load_ratings(sample_frac=0.1)  # 10% for testing
    print(f"Loaded {len(ratings):,} ratings")
    
    # Split data
    print("\n2. Temporal leave-one-positive-out split...")
    train_bpr, test_positive = temporal_leave_one_positive_out(
        ratings, threshold=POSITIVE_THRESHOLD
    )
    
    # Filter cold-start
    warm_users = set(train_bpr["userId"].unique())
    warm_movies = set(train_bpr["movieId"].unique())
    test_positive = test_positive[
        test_positive["userId"].isin(warm_users) & 
        test_positive["movieId"].isin(warm_movies)
    ].copy()
    print(f"Test positive (warm): {len(test_positive):,} users")
    
    # Train BPR
    print("\n3. Training BPR model...")
    start_time = time.perf_counter()
    bpr = BPRRecommender(
        factors=64,
        learning_rate=0.05,
        regularization=0.01,
        iterations=30,  # Reduced for testing
        random_state=RANDOM_SEED
    )
    bpr.fit(train_bpr)
    bpr_train_time = time.perf_counter() - start_time
    print(f"BPR training time: {bpr_train_time:.2f}s")
    
    # Train ALS
    print("\n4. Training ALS model...")
    start_time = time.perf_counter()
    als = ALSRecommender(
        factors=64,
        regularization=0.01,
        iterations=15,
        alpha=1.0,
        random_state=RANDOM_SEED
    )
    als.fit(train_bpr)
    als_train_time = time.perf_counter() - start_time
    print(f"ALS training time: {als_train_time:.2f}s")
    
    # Build ranking candidates
    print("\n5. Building ranking candidates...")
    item_ids = np.sort(train_bpr["movieId"].unique())
    ranking_candidates = build_ranking_candidates(
        train_bpr, test_positive, item_ids, negatives_per_user=99
    )
    
    # Evaluate both models
    print("\n6. Evaluating models...")
    results = {"BPR": {}, "ALS": {}}
    
    for k in [5, 10, 20]:
        print(f"\n  Evaluating @ K={k}...")
        
        # BPR
        bpr_result = evaluate_ranking_at_k(bpr, ranking_candidates, k=k)
        results["BPR"][k] = bpr_result
        print(f"    BPR - HitRate@{k}: {bpr_result['hit_rate']:.4f}, "
              f"NDCG@{k}: {bpr_result['ndcg']:.4f}")
        
        # ALS
        als_result = evaluate_ranking_at_k(als, ranking_candidates, k=k)
        results["ALS"][k] = als_result
        print(f"    ALS - HitRate@{k}: {als_result['hit_rate']:.4f}, "
              f"NDCG@{k}: {als_result['ndcg']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for k in [5, 10, 20]:
        print(f"\n@ K={k}:")
        print(f"  BPR - HitRate: {results['BPR'][k]['hit_rate']:.4f}, "
              f"NDCG: {results['BPR'][k]['ndcg']:.4f}")
        print(f"  ALS - HitRate: {results['ALS'][k]['hit_rate']:.4f}, "
              f"NDCG: {results['ALS'][k]['ndcg']:.4f}")
    
    print(f"\nTraining Time:")
    print(f"  BPR: {bpr_train_time:.2f}s")
    print(f"  ALS: {als_train_time:.2f}s")
    print("="*70)
    
    # Save models
    print("\n7. Saving models...")
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    bpr.save(str(models_dir / "bpr_best.pkl"))
    print("✓ All experiments complete!")


if __name__ == "__main__":
    main()
