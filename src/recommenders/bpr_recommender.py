"""Bayesian Personalized Ranking (BPR) Recommender for Implicit Feedback.
"""

import pickle
from pathlib import Path
from typing import Set, List, Optional

import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking


class BPRRecommender:
    """Bayesian Personalized Ranking for implicit feedback data.
    
    BPR learns user and item latent factors by optimizing pairwise ranking:
    for each user, observed items should rank higher than unobserved items.
    This is trained using SGD on sampled triplets (user, positive_item, negative_item).
    
    Unlike ALS which minimizes reconstruction error, BPR directly optimizes
    ranking metrics, making it well-suited for Top-N recommendation tasks.
    
    Attributes:
        factors: Number of latent factors (embedding dimensions)
        learning_rate: SGD learning rate for BPR optimization
        regularization: L2 regularization strength
        iterations: Number of training iterations
        verify_negative_samples: Ensure sampled negatives are truly unobserved
        random_state: Random seed for reproducibility
        model: Fitted BayesianPersonalizedRanking model from implicit library
        user_map: Dict mapping original userId -> matrix index
        movie_map: Dict mapping original movieId -> matrix index
        user_map_inv: Dict mapping matrix index -> original userId
        movie_map_inv: Dict mapping matrix index -> original movieId
        train_matrix: User×item binary matrix used for training (1=observed, 0=unobserved)
    """
    
    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        iterations: int = 100,
        verify_negative_samples: bool = True,
        random_state: int = 42
    ):
        """Initialize BPR recommender.
        
        Args:
            factors: Number of latent factors (default: 64)
            learning_rate: SGD learning rate (default: 0.05)
            regularization: L2 regularization strength (default: 0.01)
            iterations: Number of training iterations (default: 100)
                       BPR typically needs more iterations than ALS
            verify_negative_samples: Check that negatives are unobserved (default: True)
            random_state: Random seed (default: 42)
        """
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.verify_negative_samples = verify_negative_samples
        self.random_state = random_state
        
        # Initialize implicit BPR model
        self.model = BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
            verify_negative_samples=verify_negative_samples,
            random_state=random_state
        )
        
        self.user_map = None
        self.movie_map = None
        self.user_map_inv = None
        self.movie_map_inv = None
        self.train_matrix = None
    
    def fit(self, train_implicit_df: pd.DataFrame) -> "BPRRecommender":
        """Fit the BPR model on implicit feedback data.
        
        BPR uses binary feedback (observed vs unobserved), so it doesn't
        need confidence weighting like ALS. Each observed interaction gets
        value 1.0 in the matrix.
        
        Args:
            train_implicit_df: DataFrame with columns [userId, movieId, rating]
                               containing only positive interactions (rating >= threshold)
        
        Returns:
            self
        """
        # Build binary matrix: each cell = 1.0 for observed interactions
        from src.data_loader import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        user_item_csr, user_map, movie_map = preprocessor.build_user_item_matrix(
            train_implicit_df,
            value_column=None,  # Binary matrix
            confidence_alpha=None  # No confidence weighting
        )
        
        # Store mappings
        self.user_map = user_map
        self.movie_map = movie_map
        self.user_map_inv = {idx: uid for uid, idx in user_map.items()}
        self.movie_map_inv = {idx: mid for mid, idx in movie_map.items()}
        self.train_matrix = user_item_csr
        
        # Fit model
        print(f"Training BPR: factors={self.factors}, lr={self.learning_rate}, "
              f"reg={self.regularization}, iters={self.iterations}")
        
        self.model.fit(user_item_csr, show_progress=True)
        
        print(f"BPR training complete")
        
        return self
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        seen: Optional[Set[int]] = None
    ) -> List[int]:
        """Generate top-N item recommendations for a user.
        
        BPR ranks items by learned latent factors. Items with higher scores
        are predicted to be preferred by the user.
        
        Args:
            user_id: Original user ID
            n: Number of recommendations to return
            seen: Set of movieIds already seen by user (will be excluded)
                  If None, only training items are filtered
        
        Returns:
            List of recommended movieIds, ordered by score (highest first)
        
        Raises:
            RuntimeError: If model not fitted
        
        Note:
            - If user not in training data, returns empty list (cold-start)
            - filter_already_liked_items=True excludes training interactions
            - BPR scores are relative rankings, not probabilities or ratings
        """
        if self.train_matrix is None:
            raise RuntimeError("Call fit() first")
        
        if n <= 0:
            return []
        
        # Check if user exists in training data
        if user_id not in self.user_map:
            return []
        
        user_idx = self.user_map[user_id]
        
        # Get user's item interactions from training matrix
        user_items = self.train_matrix[user_idx]
        
        # Generate recommendations (implicit library handles filtering)
        try:
            indices, scores = self.model.recommend(
                userid=user_idx,
                user_items=user_items,
                N=n,
                filter_already_liked_items=True
            )
        except Exception as e:
            print(f"Warning: BPR recommend failed for user {user_id}: {e}")
            return []
        
        # Map indices back to original movieIds
        recommendations = [self.movie_map_inv[idx] for idx in indices]
        
        # Additional filtering for seen items (if provided and different from train)
        if seen:
            recommendations = [mid for mid in recommendations if mid not in seen][:n]
        
        return recommendations
    
    def save(self, path: str):
        """Save model to pickle file.
        
        Args:
            path: Output file path (e.g., 'models/bpr_best.pkl')
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'factors': self.factors,
                'learning_rate': self.learning_rate,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'verify_negative_samples': self.verify_negative_samples,
                'random_state': self.random_state,
                'user_map': self.user_map,
                'movie_map': self.movie_map,
                'user_map_inv': self.user_map_inv,
                'movie_map_inv': self.movie_map_inv,
                'train_matrix': self.train_matrix,
            }, f)
        
        print(f"BPR model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BPRRecommender":
        """Load model from pickle file.
        
        Args:
            path: Input file path
        
        Returns:
            Loaded BPRRecommender instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(
            factors=data['factors'],
            learning_rate=data['learning_rate'],
            regularization=data['regularization'],
            iterations=data['iterations'],
            verify_negative_samples=data['verify_negative_samples'],
            random_state=data['random_state']
        )
        
        obj.model = data['model']
        obj.user_map = data['user_map']
        obj.movie_map = data['movie_map']
        obj.user_map_inv = data['user_map_inv']
        obj.movie_map_inv = data['movie_map_inv']
        obj.train_matrix = data['train_matrix']
        
        print(f"BPR model loaded from {path}")
        
        return obj
