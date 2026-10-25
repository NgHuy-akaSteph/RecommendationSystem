# Implementation Summary - MovieLens 25M Dataset

## ✅ Đã Hoàn Thành

### 1. Data Loading Module (`src/data_loader.py`)

**Class `ML25MLoader`:**
- Load ratings.csv với memory optimization (dtype optimization)
- Support sampling để test nhanh (sample_frac parameter)
- Load movies.csv và tags.csv

**Class `DataPreprocessor`:**
- `temporal_split()`: Chia train/test theo timestamp (70/30)
- `filter_cold_start()`: Loại bỏ cold-start users/items khỏi test set
- `to_implicit()`: Convert explicit ratings → implicit (threshold ≥ 4.0)
- `build_user_item_matrix()`: Tạo sparse matrix cho CF

### 2. Data Preprocessing Pipeline

**Notebook: `notebooks/01-ml25m-preprocessing.ipynb`**
- Load full ML-25M dataset (~25M ratings)
- Basic statistics: users, movies, sparsity, time range
- Temporal split (70% train, 30% test)
- Filter cold-start interactions
- Convert to implicit feedback (rating ≥ 4.0)
- Build user-item sparse matrix
- Create sampled train set (30% users) cho fast iteration
- Save processed data + mappings (pickle)

### 3. EDA Visualizations (6 figures)

1. **Rating Distribution**: Phân bố ratings trên ML-25M
2. **User Activity**: Distribution của số ratings/user (linear + log scale)
3. **Movie Popularity**: Distribution của số ratings/movie (linear + log scale)
4. **Temporal Patterns**: Rating activity over time
5. **Top-20 Movies**: Most rated movies với counts
6. **ML-1M vs ML-25M Comparison**: Side-by-side comparison (users, movies, ratings, sparsity)

### 4. Collaborative Filtering Module (`src/collaborative_filtering.py`)

**Class `ALSRecommender`:**
- Wrapper cho implicit.als.AlternatingLeastSquares
- `fit()`: Train ALS model với user-item matrix
- `recommend()`: Generate top-N recommendations cho user
- `recommend_batch()`: Batch recommendations cho multiple users
- `get_similar_items()`: Find similar items dựa vào embeddings
- `save_model()` / `load_model()`: Persist trained model

**Features:**
- Configurable hyperparameters (factors, regularization, iterations)
- Handle cold-start users gracefully
- ID mappings (user_id ↔ matrix index, movie_id ↔ matrix index)

### 5. Training Pipeline

**Notebook: `notebooks/02-als-training.ipynb`:**
- Load processed data
- Build implicit matrix với confidence weighting
- Train ALS model (factors=50, reg=0.01, iter=20)
- Test recommendations cho random users
- Test item similarity
- Save trained model

## 📊 Expected Outputs

### Files Generated:

```
data/
├── processed/
│   ├── train_ml25m.csv          # ~17.5M implicit interactions
│   ├── test_ml25m.csv            # ~7.5M implicit interactions
│   ├── sampled/
│   │   └── train_ml25m.csv       # 30% users subsample
│   ├── user_id_map.pkl           # userId → matrix index
│   └── movie_id_map.pkl          # movieId → matrix index

figures/
├── ml25m_rating_distribution.png
├── ml25m_user_activity.png
├── ml25m_movie_popularity.png
├── ml25m_temporal_patterns.png
├── ml25m_top20_movies.png
└── ml1m_vs_ml25m_comparison.png

models/
└── als_f50_r001_i20.pkl         # Trained ALS model
```

### Key Statistics (Expected):

**ML-25M Dataset:**
- Total ratings: 25,000,095
- Unique users: ~162,541
- Unique movies: ~59,047
- Sparsity: ~99.74%
- Date range: 1995-01-09 to 2019-11-21

**After Preprocessing:**
- Train implicit (≥4.0): ~12.2M interactions (70%)
- Test implicit (≥4.0): ~5.2M interactions (30%)
- Cold-start removed: ~2-5%
- Sampled train (30% users): ~3.7M interactions

## 🎯 Next Steps

### Week 2: Baseline & CF Implementation (Continue from here)

1. **Hyperparameter Tuning** (2 giờ):
   - Grid search trên sampled data (30% users)
   - Test factors: [20, 50, 100, 150]
   - Test regularization: [0.001, 0.01, 0.1]
   - Test iterations: [10, 20, 30]
   - Select best config dựa vào validation metrics

2. **Evaluation Module** (1.5 giờ):
   - Implement Precision@K, Recall@K
   - Implement NDCG@K
   - Implement Hit Rate@K
   - Implement Coverage metrics
   - Compare Popularity Baseline vs ALS

3. **Full Training** (0.5 giờ):
   - Train ALS trên full training set với best hyperparams
   - Generate final evaluation metrics

### Week 3: Analysis & Visualization

4. **Results Analysis**:
   - Performance comparison table
   - Learning curves (iterations vs metrics)
   - User-level analysis (active vs casual users)
   - Case studies (specific user examples)

5. **Report Writing**:
   - Chapter 3: Dataset (ML-25M statistics + EDA)
   - Chapter 4.3: CF with ALS (theory + implementation)
   - Chapter 5.3: CF Results (metrics + analysis)

## 💡 Implementation Notes

### Memory Optimization:
- Sử dụng `dtype` optimization (int32, float16) khi load CSV
- Sparse matrix format (CSR) để tiết kiệm memory
- Sampled training set (30% users) cho fast iteration

### Temporal Split Rationale:
- Simulate real-world: predict future từ past
- No data leakage: train timestamps < test timestamps
- Realistic evaluation: users/items đều xuất hiện trong train

### Implicit Feedback Choice:
- Threshold ≥ 4.0 = positive signal
- ~49% của ratings trở thành implicit
- Phù hợp với ALS implicit feedback model

### Code Structure:
- Modular design: data_loader, baselines, collaborative_filtering
- Reusable functions: load_data(), build_matrix(), etc.
- Type hints + docstrings đầy đủ
- Follow Python best practices

## 🔗 References

**Papers:**
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets.

**Libraries:**
- implicit: https://github.com/benfred/implicit
- scipy.sparse: https://docs.scipy.org/doc/scipy/reference/sparse.html
- pandas: https://pandas.pydata.org/

**Dataset:**
- MovieLens 25M: https://grouplens.org/datasets/movielens/25m/
