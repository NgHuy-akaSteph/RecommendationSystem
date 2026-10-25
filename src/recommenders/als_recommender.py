"""Alternating Least Squares (ALS) Recommender for Implicit Feedback.
"""

import pickle
from pathlib import Path
from typing import Set, List, Optional

import pandas as pd
from implicit.als import AlternatingLeastSquares


class ALSRecommender:
    """Alternating Least Squares for implicit feedback data.
    
    This recommender uses matrix factorization to learn user and item latent factors.
    It converts explicit ratings to implicit confidence values using:
        c_ui = 1 + alpha * rating
    
    where higher ratings indicate stronger confidence in the preference.
    
    Attributes:
        factors: Number of latent factors (embedding dimensions)
        regularization: L2 regularization strength
        iterations: Number of ALS iterations
        alpha: Confidence scaling parameter
        random_state: Random seed for reproducibility
        model: Fitted AlternatingLeastSquares model from implicit library
        user_map: Dict mapping original userId -> matrix index
        movie_map: Dict mapping original movieId -> matrix index
        user_map_inv: Dict mapping matrix index -> original userId
        movie_map_inv: Dict mapping matrix index -> original movieId
        train_matrix: User×item confidence matrix used for training
    """
    
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 1.0,
        random_state: int = 42
    ):
        """Initialize ALS recommender.
        
        Args:
            factors: Number of latent factors (default: 64)
            regularization: L2 regularization strength (default: 0.01)
            iterations: Number of ALS iterations (default: 15)
            alpha: Confidence scaling parameter (default: 1.0)
            random_state: Random seed (default: 42)
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        
        # Initialize implicit ALS model
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
            calculate_training_loss=True
        )
        
        self.user_map = None
        self.movie_map = None
        self.user_map_inv = None
        self.movie_map_inv = None
        self.train_matrix = None
    
    def fit(self, train_implicit_df: pd.DataFrame) -> "ALSRecommender":
        """Fit the ALS model on implicit feedback data.
        
        Args:
            train_implicit_df: DataFrame with columns [userId, movieId, rating]
                               containing only positive interactions (rating >= threshold)
        
        Returns:
            self
        """
        # Build confidence matrix: c_ui = 1 + alpha * rating
        from src.data_loader import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        user_item_csr, user_map, movie_map = preprocessor.build_user_item_matrix(
            train_implicit_df,
            confidence_alpha=self.alpha
        )
        
        # Store mappings
        self.user_map = user_map
        self.movie_map = movie_map
        self.user_map_inv = {idx: uid for uid, idx in user_map.items()}
        self.movie_map_inv = {idx: mid for mid, idx in movie_map.items()}
        self.train_matrix = user_item_csr
        
        # Fit model (implicit library expects item×user matrix)
        print(f"Training ALS: factors={self.factors}, reg={self.regularization}, "
              f"iters={self.iterations}, alpha={self.alpha}")
        
        self.model.fit(user_item_csr)
        
        print(f"ALS training complete")

        return self
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        seen: Optional[Set[int]] = None
    ) -> List[int]:
        """Generate top-N item recommendations for a user.
        
        Args:
            user_id: Original user ID
            n: Number of recommendations to return
            seen: Set of movieIds already seen by user (will be excluded)
        
        Returns:
            List of recommended movieIds, ordered by score (highest first)
        
        Raises:
            RuntimeError: If model not fitted
        
        Note:
            - If user not in training data, returns empty list
            - filter_already_liked_items=True automatically excludes train items
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
        # Note: recommend() returns (indices, scores)
        try:
            indices, scores = self.model.recommend(
                userid=user_idx,
                user_items=user_items,
                N=n,
                filter_already_liked_items=True
            )
        except Exception as e:
            print(f"Warning: ALS recommend failed for user {user_id}: {e}")
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
            path: Output file path (e.g., 'models/als_best.pkl')
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'factors': self.factors,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'alpha': self.alpha,
                'random_state': self.random_state,
                'user_map': self.user_map,
                'movie_map': self.movie_map,
                'user_map_inv': self.user_map_inv,
                'movie_map_inv': self.movie_map_inv,
                'train_matrix': self.train_matrix,
            }, f)
        
        print(f"ALS model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ALSRecommender":
        """Load model from pickle file.
        
        Args:
            path: Input file path
        
        Returns:
            Loaded ALSRecommender instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(
            factors=data['factors'],
            regularization=data['regularization'],
            iterations=data['iterations'],
            alpha=data['alpha'],
            random_state=data['random_state']
        )
        
        obj.model = data['model']
        obj.user_map = data['user_map']
        obj.movie_map = data['movie_map']
        obj.user_map_inv = data['user_map_inv']
        obj.movie_map_inv = data['movie_map_inv']
        obj.train_matrix = data['train_matrix']
        
        print(f"ALS model loaded from {path}")
        
        return obj
