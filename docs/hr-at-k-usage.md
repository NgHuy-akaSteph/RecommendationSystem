# HR@K (Hit Rate@K) - Usage Guide

## Tổng Quan

HR@K (Hit Rate@K) đã được bổ sung vào `src/evaluation.py` trong class `RecommenderEvaluator`.

## Định Nghĩa

**Hit Rate@K** là binary metric đánh giá xem có ít nhất 1 relevant item trong top-K recommendations hay không:

```
HR@K = 1.0  nếu có ít nhất 1 relevant item trong top-K
HR@K = 0.0  nếu không có relevant item nào trong top-K
```

Khi aggregate cho nhiều users:
```
HR@K = (Số users có ít nhất 1 hit) / (Tổng số users)
```

## Cách Sử Dụng

### 1. Import

```python
from src.evaluation import RecommenderEvaluator
```

### 2. Single User Evaluation

```python
evaluator = RecommenderEvaluator()

# Recommended items (ordered by score)
recommended = [101, 205, 303, 405, 501]

# Ground truth relevant items
relevant = {303, 607, 708}

# Compute HR@5
hr_5 = evaluator.hit_rate_at_k(recommended, relevant, k=5)
# Output: 1.0 (vì item 303 có trong top-5)

# Compute HR@3
hr_3 = evaluator.hit_rate_at_k(recommended, relevant, k=3)
# Output: 1.0 (vì item 303 có trong top-3)

# Compute HR@2
hr_2 = evaluator.hit_rate_at_k(recommended, relevant, k=2)
# Output: 0.0 (không có item nào trong top-2)
```

### 3. Multiple Users Evaluation

```python
import numpy as np

evaluator = RecommenderEvaluator(k_values=[5, 10, 20])

# Dictionary: {user_id: recommended_items}
recommendations = {
    1: [10, 20, 30, 40, 50],
    2: [15, 25, 35, 45, 55],
    3: [11, 21, 31, 41, 51]
}

# Dictionary: {user_id: relevant_items}
ground_truth = {
    1: {30, 60, 70},  # Hit: item 30 trong recommendations
    2: {16, 26, 36},  # Miss: không có item nào
    3: {51, 61}       # Hit: item 51 trong recommendations
}

# Compute HR@K cho từng user
hr_scores = []
for user_id, recommended in recommendations.items():
    relevant = ground_truth[user_id]
    hr = evaluator.hit_rate_at_k(recommended, relevant, k=5)
    hr_scores.append(hr)

# Aggregate: Average HR@5 across all users
avg_hr_5 = np.mean(hr_scores)
print(f"HR@5: {avg_hr_5:.3f}")
# Output: HR@5: 0.667 (2 out of 3 users có hit)
```

## Khác Biệt với Precision@K

| Metric | Giá Trị | Ý Nghĩa |
|--------|---------|---------|
| **HR@K** | {0.0, 1.0} | Binary: Có hay không có hit? |
| **Precision@K** | [0.0, 1.0] | Tỷ lệ: Bao nhiêu % items là relevant? |

### Ví Dụ So Sánh

```python
recommended = [1, 2, 3, 4, 5]
relevant = {3, 6, 7}
k = 5

hr = evaluator.hit_rate_at_k(recommended, relevant, k)
prec = evaluator.precision_at_k(recommended, relevant, k)

print(f"HR@{k}: {hr}")      # 1.0 (có hit)
print(f"Prec@{k}: {prec}")  # 0.2 (1/5 = 20%)
```

Cả hai đều đo lường quality nhưng:
- **HR@K**: "User có nhận được ít nhất 1 good recommendation không?"
- **Precision@K**: "Bao nhiêu % recommendations là good?"

## Use Cases

### 1. BPR/ALS Evaluation (Implicit Feedback)

Theo `notes/bpr-ml25m-report.md`, HR@K là metric chính cho ranking evaluation:

```python
# Sampled ranking protocol: 1 positive + 99 negatives
for user_id in test_users:
    # Rank 100 candidates
    ranked_items = model.rank_candidates(user_id, candidates)
    
    # Compute HR@K
    hr_5 = evaluator.hit_rate_at_k(ranked_items, {positive_item}, k=5)
    hr_10 = evaluator.hit_rate_at_k(ranked_items, {positive_item}, k=10)
```

### 2. Content-Based Filtering

Theo `notes/person2-content-based-guide.md`:

```python
# Đánh giá % users có ít nhất 1 good recommendation
hr_scores = []
for user_id, recommended in recommendations.items():
    relevant = test_relevant[user_id]
    hr = evaluator.hit_rate_at_k(recommended, relevant, k=10)
    hr_scores.append(hr)

avg_hr = np.mean(hr_scores)
print(f"Coverage: {avg_hr*100:.1f}% users got ≥1 good recommendation")
```

### 3. Baseline Comparison

```python
models = ['Popularity', 'Bayesian', 'Wilson', 'ALS', 'BPR']
results = {}

for model_name in models:
    # ... train model ...
    hr_scores = []
    for user_id in test_users:
        recommended = model.recommend(user_id, n=10)
        relevant = test_relevant[user_id]
        hr = evaluator.hit_rate_at_k(recommended, relevant, k=10)
        hr_scores.append(hr)
    
    results[model_name] = np.mean(hr_scores)

# Report
for model_name, hr in results.items():
    print(f"{model_name}: HR@10 = {hr:.3f}")
```

## Edge Cases

Method xử lý tất cả edge cases:

```python
evaluator = RecommenderEvaluator()

# k <= 0
assert evaluator.hit_rate_at_k([1,2,3], {3}, k=0) == 0.0
assert evaluator.hit_rate_at_k([1,2,3], {3}, k=-1) == 0.0

# Empty lists
assert evaluator.hit_rate_at_k([], {3}, k=5) == 0.0
assert evaluator.hit_rate_at_k([1,2,3], set(), k=5) == 0.0

# k > len(recommended)
assert evaluator.hit_rate_at_k([1,2,3], {3}, k=100) == 1.0
```

## Testing

Chạy test suite:

```bash
python3 test_hr_simple.py
```

Expected output:
```
✓ ALL TESTS PASSED!
```

## Tham Khảo

- `src/evaluation.py`: Implementation
- `notes/bpr-ml25m-report.md`: BPR evaluation protocol với HR@K
- `notes/person2-content-based-guide.md`: Content-based evaluation với HR@K
- `test_hr_simple.py`: Comprehensive test cases
