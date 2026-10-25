"""Protocol definitions for recommender systems evaluation.

This module defines the Protocol interface that all recommender models must implement
to be compatible with the EvaluationPipeline.
"""

from typing import Protocol, List, Set, Optional


class Recommender(Protocol):
    """Protocol interface for all recommender models.
    
    Any recommender model that implements this protocol can be evaluated
    using the EvaluationPipeline. This uses structural subtyping (duck typing)
    so models don't need to explicitly inherit from this class.
    """
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        seen: Optional[Set[int]] = None
    ) -> List[int]:
        """Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            n: Number of recommendations to return (default: 10)
            seen: Set of item IDs already seen by user. These items
                  should be excluded from recommendations (default: None)
        
        Returns:
            List of recommended item IDs, ordered by score (highest first).
            Length should be <= n. No duplicates. No items from seen set.
            
        Raises:
            RuntimeError: If model not fitted (if applicable)
        """
        ...
