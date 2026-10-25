# BPR Implementation Summary

## Overview

Đã triển khai thành công **Bayesian Personalized Ranking (BPR)** cho Collaborative Filtering và tích hợp vào hệ thống recommendation hiện có.

## Implementation Details

### 1. BPRRecommender Class

**File:** `src/recommenders/bpr_recommender.py`

**API Design:**
```python
class BPRRecommender:
    def __init__(self, factors=64, learning_rate=0.05, 
                 regularization=0.01, iterations=100, 
                 verify_negative_samples=True, random_state=42)
    def fit(self, train_implicit_df: pd.DataFrame) -> "BPRRecommender"
    def recommend(self, user_id: int, n: int = 10, 
                  seen: Optional[Set[int]] = None) -> List[int]
    def save(self, path: str)
    @classmethod
    def load(cls, path: str) -> "BPRRecommender"
```

**Key Features:**
- Sử dụng `implicit.bpr.BayesianPersonalizedRanking` từ thư viện implicit
- Binary matrix (0/1) - không cần confidence weighting như ALS
- Tối ưu pairwise ranking thay vì reconstruction error
- API nhất quán với `ALSRecommender` để dễ swap models
- Pickle serialization cho model persistence

**Differences từ ALS:**
| Aspect | BPR | ALS |
|--------|-----|-----|
| Matrix | Binary (0/1) | Weighted confidence (1 + α × rating) |
| Objective | Pairwise ranking | Reconstruction error |
| Training | SGD on triplets (u,i,j) | Alternating least squares |
| Iterations | 100+ (default) | 15-30 (default) |
| Best for | Top-N ranking | General implicit feedback |

### 2. Notebook Implementation

**File:** `notebooks/03-bpr-experiments.ipynb`

**Structure (32 cells):**

1. **Setup & Import** - Dependencies và configuration
2. **Helper Functions**
   - `temporal_leave_one_positive_out()` - Split dữ liệu
   - `build_ranking_candidates()` - Tạo 1 positive + 99 negatives
   - `evaluate_ranking_at_k()` - HitRate@K và NDCG@K
3. **Load Data** - MovieLens 25M với sampling option
4. **Data Split** - Temporal leave-one-positive-out protocol
5. **Train BPR** - Fit model với binary matrix
6. **Train ALS** - Baseline để so sánh
7. **Build Candidates** - Sampled ranking protocol
8. **Evaluation** - Evaluate cả 2 models trên cùng protocol
9. **Results Summary** - In kết quả so sánh
10. **Visualization** - Bar charts cho HitRate@K và NDCG@K
11. **Export** - Save models và results JSON
12. **Conclusion** - Key findings và next steps

### 3. Evaluation Protocol

**Sampled Ranking Protocol:**
- Mỗi user: 1 positive item (held-out) + 99 negative items (sampled)
- Model rank tất cả 100 candidates
- Tìm vị trí của positive item trong rankings

**Metrics:**
- **HitRate@K**: Tỷ lệ positive item xuất hiện trong top-K
- **NDCG@K**: Discounted cumulative gain với position discount (1/log₂(rank+2))

**K values:** 5, 10, 20

### 4. Integration

**Updated Files:**
- `src/recommenders/__init__.py` - Export `BPRRecommender`
- `src/recommenders/bpr_recommender.py` - New implementation
- `notebooks/03-bpr-experiments.ipynb` - Complete experiment notebook
- `notebooks/bpr_experiment_script.py` - Standalone script version

**Export Locations:**
- Models: `models/bpr_best.pkl`, `models/als_bpr_comparison.pkl`
- Results: `results/bpr_vs_als_comparison.csv`, `results/bpr_ml25m_results.json`
- Figures: `reports/figures/bpr_vs_als_comparison.png`

## Usage

### Import và sử dụng BPRRecommender

```python
from src.recommenders import BPRRecommender

# Initialize
bpr = BPRRecommender(
    factors=64,
    learning_rate=0.05,
    regularization=0.01,
    iterations=100,
    random_state=42
)

# Train
bpr.fit(train_implicit_df)

# Recommend
recommendations = bpr.recommend(user_id=123, n=10)

# Save/Load
bpr.save("models/bpr_best.pkl")
loaded_bpr = BPRRecommender.load("models/bpr_best.pkl")
```

### Chạy notebook

```bash
# Option 1: Jupyter
jupyter notebook notebooks/03-bpr-experiments.ipynb

# Option 2: Standalone script (for testing)
python notebooks/bpr_experiment_script.py
```

## Technical Considerations

### Data Requirements

**Input:** DataFrame với columns `[userId, movieId, rating]`
- Chỉ chứa positive interactions (rating >= threshold, default 4.0)
- No need for confidence weighting

**Matrix:** Binary CSR matrix
- 1.0 nếu user-item interaction tồn tại
- 0.0 nếu không (unobserved)

### Performance

**Training:**
- BPR thường chậm hơn ALS (nhiều iterations)
- Sample 10% của ML-25M để test nhanh (~15-30 phút)
- Full dataset cần nhiều giờ

**Memory:**
- Binary matrix nhẹ hơn confidence-weighted matrix
- Sparse CSR format hiệu quả với implicit feedback

### Hyperparameters

**BPR defaults:**
- `factors=64` - latent dimensions
- `learning_rate=0.05` - SGD learning rate
- `regularization=0.01` - L2 regularization
- `iterations=100` - training iterations

**Tuning tips:**
- Tăng `iterations` nếu chưa converge
- Giảm `learning_rate` nếu training unstable
- Tăng `regularization` nếu overfitting

## Comparison with ALS

### Expected Results

**BPR strengths:**
- ✅ Tối ưu trực tiếp cho ranking
- ✅ Thường tốt hơn trên NDCG@K
- ✅ Phù hợp với Top-N recommendations

**ALS strengths:**
- ✅ Train nhanh hơn (fewer iterations)
- ✅ Stable và dễ scale
- ✅ Confidence weighting linh hoạt

**Trade-offs:**
- BPR: Tốt hơn cho ranking nhưng chậm hơn
- ALS: Cân bằng performance/speed

## Next Steps

### Immediate

- [x] Implement BPRRecommender class
- [x] Create notebook với complete pipeline
- [x] Integrate với existing codebase
- [ ] Chạy notebook end-to-end và verify results
- [ ] Document actual performance metrics

### Future Enhancements

1. **Hyperparameter Tuning**
   - Grid search cho learning_rate, iterations
   - Validation set để select best config

2. **Full Dataset Experiment**
   - Chạy trên 100% MovieLens 25M
   - So sánh với published benchmarks

3. **Advanced Metrics**
   - MAP (Mean Average Precision)
   - MRR (Mean Reciprocal Rank)
   - Coverage và diversity metrics

4. **Hybrid Approaches**
   - Ensemble BPR + ALS
   - BPR + Content-based features
   - Neural extensions (NCF, NGCF)

## References

1. **Paper:** Rendle, S. et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI.
2. **Library:** implicit - Python library for implicit feedback datasets
3. **Dataset:** MovieLens 25M (GroupLens Research)
4. **Guide:** `notes/advanced-cf-bpr-svd-svdpp-ml1m-guide.md`

## Files Created/Modified

### New Files
- `src/recommenders/bpr_recommender.py` - BPR implementation
- `notebooks/03-bpr-experiments.ipynb` - Complete experiment notebook
- `notebooks/bpr_experiment_script.py` - Standalone script
- `docs/bpr-implementation-summary.md` - This file

### Modified Files
- `src/recommenders/__init__.py` - Added BPRRecommender export

### Generated (after running)
- `models/bpr_best.pkl` - Trained BPR model
- `models/als_bpr_comparison.pkl` - ALS model for comparison
- `results/bpr_vs_als_comparison.csv` - Metrics comparison
- `results/bpr_ml25m_results.json` - Detailed results
- `reports/figures/bpr_vs_als_comparison.png` - Visualization

## Troubleshooting

### Common Issues

**1. Import Error: No module named 'implicit'**
```bash
pip install implicit==0.7.3
```

**2. Memory Error**
- Reduce `sample_frac` in data loading
- Use smaller `factors` value
- Close other applications

**3. Training Too Slow**
- Reduce `iterations` for testing (e.g., 30)
- Use smaller dataset sample
- Check CPU usage and parallelization

**4. Cold-start Users**
- BPR returns empty list for unknown users
- Use popularity fallback for cold-start
- Implement hybrid with content-based

## Conclusion

BPR implementation hoàn tất và sẵn sàng để thực nghiệm. Class API nhất quán với ALSRecommender, notebook có đầy đủ pipeline từ data loading đến visualization và model export. 

**Status:** ✅ Implementation Complete | ⏳ Awaiting Full Experiment Run
