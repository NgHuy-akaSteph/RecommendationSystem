# BPR Quick Reference Card

## 🚀 Quick Start

```python
from src.recommenders import BPRRecommender

# Initialize
bpr = BPRRecommender(
    factors=64,           # Latent dimensions
    learning_rate=0.05,   # SGD learning rate
    regularization=0.01,  # L2 regularization
    iterations=100        # Training iterations
)

# Train
bpr.fit(train_df)  # DataFrame: userId, movieId, rating

# Recommend
recommendations = bpr.recommend(user_id=123, n=10)

# Save/Load
bpr.save("models/bpr_model.pkl")
bpr = BPRRecommender.load("models/bpr_model.pkl")
```

## 📊 Key Differences: BPR vs ALS

| Feature | BPR | ALS |
|---------|-----|-----|
| **Best for** | Top-N ranking | General recommendations |
| **Training** | SGD (slower) | Alternating LS (faster) |
| **Iterations** | 100+ | 15-30 |
| **Matrix** | Binary (0/1) | Confidence-weighted |
| **Metrics** | Higher NDCG@K | Balanced |

## 🎯 When to Use BPR

✅ **Use BPR when:**
- Top-N ranking is primary goal
- Have implicit feedback (clicks, views, purchases)
- Care more about ranking quality than speed
- Want to optimize for NDCG/HitRate

❌ **Use ALS when:**
- Need faster training
- Want balanced performance
- Working with large-scale systems
- Speed > slight accuracy gain

## 📈 Hyperparameter Guide

### Default (balanced)
```python
BPRRecommender(factors=64, learning_rate=0.05, iterations=100)
```

### Fast (quick experiments)
```python
BPRRecommender(factors=32, learning_rate=0.1, iterations=50)
```

### Accurate (production)
```python
BPRRecommender(factors=128, learning_rate=0.01, iterations=200)
```

## 🔍 Evaluation Metrics

```python
# HitRate@10: Did we find the item in top-10?
hit_rate = sum(hits) / len(test_users)

# NDCG@10: How high did we rank it?
# Higher position = better score (1/log2(rank+2))
ndcg = mean([1/log2(rank+2) if rank < 10 else 0])
```

**Target metrics (MovieLens):**
- HitRate@10: > 0.65
- NDCG@10: > 0.45

## 🛠️ Common Operations

### Filter seen items
```python
seen_items = set(user_history['movieId'])
recs = bpr.recommend(user_id=123, n=10, seen=seen_items)
```

### Batch recommendations
```python
all_recs = {}
for user_id in user_ids:
    all_recs[user_id] = bpr.recommend(user_id, n=10)
```

### Handle cold-start
```python
recs = bpr.recommend(user_id=new_user, n=10)
if not recs:
    # Fallback to popularity
    recs = popularity_recommender.recommend(n=10)
```

## 🐛 Troubleshooting

### Model trains but gives poor recommendations
- Increase iterations (100 → 200)
- Lower learning rate (0.05 → 0.01)
- Check data quality (enough interactions?)

### Training too slow
- Decrease iterations (100 → 50)
- Reduce factors (64 → 32)
- Sample data for experiments

### Memory issues
- Use smaller factors
- Sample users/items
- Process in batches

### Unknown users return empty list
- This is expected (cold-start)
- Use fallback strategy
- Consider content-based features

## 📝 Files

```
src/recommenders/bpr_recommender.py          # Core implementation
notebooks/03-bpr-experiments.ipynb           # Full pipeline
docs/bpr-implementation-complete-report.md   # Detailed docs
test_bpr_quick.py                            # Integration test
```

## 🔗 Resources

- Paper: Rendle et al. (2009) - BPR UAI
- Library: https://github.com/benfred/implicit
- Docs: [`docs/bpr-implementation-complete-report.md`](./bpr-implementation-complete-report.md)

---

**Last updated:** 2026-09-15  
**Status:** ✅ Production-ready
