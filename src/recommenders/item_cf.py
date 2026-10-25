"""Item-based Collaborative Filtering Recommender.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Set, List, Optional
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class ItemCFRecommender:
    """Item-based Collaborative Filtering using cosine similarity.
    
    This recommender computes item-item similarity based on user rating patterns.
    For each target user, it finds items similar to their rated items and
    generates recommendations based on weighted similarity scores.
    
    Attributes:
        normalized_item_user: L2-normalized item×user matrix (CSR format)
        user_item: Original user×item matrix (CSR format)
        user_map: Dict mapping original userId -> matrix index
        movie_map: Dict mapping original movieId -> matrix index
        movie_map_inv: Dict mapping matrix index -> original movieId
    """
    
    def __init__(self):
        """Initialize empty ItemCF recommender."""
        self.normalized_item_user = None
        self.user_item = None
        self.user_map = None
        self.movie_map = None
        self.movie_map_inv = None
    
    def fit(
        self,
        user_item_csr: csr_matrix,
        user_map: Dict[int, int],
        movie_map: Dict[int, int]
    ) -> "ItemCFRecommender":
        """Fit the item-based CF model.
        
        Args:
            user_item_csr: User×item CSR matrix with explicit ratings
            user_map: Mapping from original userId to matrix row index
            movie_map: Mapping from original movieId to matrix column index
        
        Returns:
            self
        """
        # Store original matrix
        self.user_item = user_item_csr
        self.user_map = user_map
        self.movie_map = movie_map
        self.movie_map_inv = {idx: mid for mid, idx in movie_map.items()}
        
        # Transpose to item×user and normalize by L2
        item_user = user_item_csr.T.tocsr()
        self.normalized_item_user = normalize(item_user, norm='l2', axis=1)
        
        print(f"ItemCF fitted: {len(movie_map)} items, {len(user_map)} users")
        
        return self
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        seen: Optional[Set[int]] = None
    ) -> List[int]:
        """Generate top-N item recommendations for a user.
        
        Algorithm:
        1. Get user's rated items (profile) from seen set
        2. For each rated item, compute similarity with all other items
        3. Aggregate similarity scores weighted by user's ratings
        4. Exclude seen items and return top-N
        
        Args:
            user_id: Original user ID
            n: Number of recommendations to return
            seen: Set of movieIds already seen by user (will be excluded)
        
        Returns:
            List of recommended movieIds, ordered by score (highest first)
        
        Raises:
            RuntimeError: If model not fitted
        
        """
        if self.normalized_item_user is None:
            raise RuntimeError("Call fit() first")
        
        if n <= 0:
            return []
        
        # Check if user exists in training data
        if user_id not in self.user_map:
            return []
        
        user_idx = self.user_map[user_id]
        
        # Get user's rating profile
        user_ratings = self.user_item[user_idx].toarray().flatten()
        
        # If user has no ratings, return empty
        if user_ratings.sum() == 0:
            return []
        
        # Compute item scores
        # For each item j: score[j] = sum_i(sim(i,j) * rating_i) for all rated items i
        scores = self.normalized_item_user @ (self.normalized_item_user.T @ user_ratings)
        
        # Exclude seen items
        if seen:
            for movie_id in seen:
                if movie_id in self.movie_map:
                    movie_idx = self.movie_map[movie_id]
                    scores[movie_idx] = -np.inf
        
        # Get top-N indices
        top_indices = np.argsort(scores)[::-1][:n]
        
        # Filter out -inf scores (all seen items)
        top_indices = [idx for idx in top_indices if scores[idx] > -np.inf]
        
        # Map back to original movieIds
        recommendations = [self.movie_map_inv[idx] for idx in top_indices]
        
        return recommendations
    
    def save(self, path: str):
        """Save model to pickle file.
        
        Args:
            path: Output file path (e.g., 'models/item_cf_best.pkl')
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'normalized_item_user': self.normalized_item_user,
                'user_item': self.user_item,
                'user_map': self.user_map,
                'movie_map': self.movie_map,
                'movie_map_inv': self.movie_map_inv,
            }, f)
        
        print(f"ItemCF model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ItemCFRecommender":
        """Load model from pickle file.
        
        Args:
            path: Input file path
        
        Returns:
            Loaded ItemCFRecommender instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls()
        obj.normalized_item_user = data['normalized_item_user']
        obj.user_item = data['user_item']
        obj.user_map = data['user_map']
        obj.movie_map = data['movie_map']
        obj.movie_map_inv = data['movie_map_inv']
        
        print(f"ItemCF model loaded from {path}")
        
        return obj
