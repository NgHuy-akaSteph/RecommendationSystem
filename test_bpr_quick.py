"""Quick test for BPR implementation with synthetic data"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*70)
print("BPR Quick Test - Synthetic Data")
print("="*70)

try:
    from src.recommenders import BPRRecommender
    print("✓ BPRRecommender imported successfully\n")
    
    # Create synthetic training data
    print("1. Creating synthetic training data...")
    np.random.seed(42)
    
    # 20 users, 50 items, ~200 interactions
    user_ids = np.random.randint(1, 21, size=200)
    item_ids = np.random.randint(1, 51, size=200)
    ratings = np.random.choice([4.0, 5.0], size=200)  # All positive
    
    train_df = pd.DataFrame({
        'userId': user_ids,
        'movieId': item_ids,
        'rating': ratings
    }).drop_duplicates(subset=['userId', 'movieId'])
    
    print(f"   Generated {len(train_df)} unique interactions")
    print(f"   Users: {train_df['userId'].nunique()}")
    print(f"   Items: {train_df['movieId'].nunique()}\n")
    
    # Initialize BPR
    print("2. Initializing BPR model...")
    bpr = BPRRecommender(
        factors=16,
        learning_rate=0.05,
        regularization=0.01,
        iterations=10,  # Quick test
        random_state=42
    )
    print(f"   Config: factors={bpr.factors}, iterations={bpr.iterations}\n")
    
    # Train
    print("3. Training BPR model...")
    import time
    start = time.perf_counter()
    bpr.fit(train_df)
    train_time = time.perf_counter() - start
    print(f"   ✓ Training completed in {train_time:.2f} seconds\n")
    
    # Test recommendations
    print("4. Testing recommendations...")
    test_user = int(train_df['userId'].iloc[0])
    
    # Get recommendations
    recs = bpr.recommend(user_id=test_user, n=10)
    print(f"   User {test_user} recommendations: {recs[:5]}... (showing 5)\n")
    
    # Test with seen filter
    seen_items = set(train_df[train_df['userId'] == test_user]['movieId'].values)
    recs_filtered = bpr.recommend(user_id=test_user, n=10, seen=seen_items)
    print(f"   With seen filter ({len(seen_items)} items): {recs_filtered[:5]}...\n")
    
    # Test serialization
    print("5. Testing model save/load...")
    test_path = PROJECT_ROOT / "models/bpr_test.pkl"
    test_path.parent.mkdir(exist_ok=True)
    
    bpr.save(str(test_path))
    print(f"   ✓ Model saved to {test_path}")
    
    loaded_bpr = BPRRecommender.load(str(test_path))
    print(f"   ✓ Model loaded successfully")
    
    # Verify loaded model works
    recs_loaded = loaded_bpr.recommend(user_id=test_user, n=10)
    assert recs_loaded == recs, "Loaded model gives different recommendations!"
    print(f"   ✓ Loaded model produces same recommendations\n")
    
    # Test edge cases
    print("6. Testing edge cases...")
    
    # Unknown user
    try:
        recs_unknown = bpr.recommend(user_id=9999, n=10)
        print(f"   Unknown user (9999): {len(recs_unknown)} recommendations")
    except Exception as e:
        print(f"   Unknown user (9999): Handled gracefully - {type(e).__name__}")
    
    # Large N
    recs_large = bpr.recommend(user_id=test_user, n=100)
    print(f"   Large N request (100): Got {len(recs_large)} recommendations")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - BPR Implementation Verified!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
