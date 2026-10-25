# Khung Báo Cáo Recommendation System

**Tổng số trang:** 35-40 trang  
**Dataset:** MovieLens 1M  
**Methods:** Popularity Baseline + Collaborative Filtering (ALS) + Content-based (TF-IDF)  
**Timeline:** 4 tuần

---

## Abstract (1 trang)

**Người chịu trách nhiệm:** Người Phụ 3

**Nội dung:**
- Tóm tắt 150-200 từ
- Problem statement
- Methods overview
- Dataset info
- Key results và findings

---

## Chapter 1: Introduction (3 trang)

**Người chịu trách nhiệm:** Người Phụ A

### 1.1 Background and Motivation
- Vai trò recommendation systems trong ecommerce
- Tầm quan trọng của personalization
- Động lực nghiên cứu

### 1.2 Problem Statement
- Bài toán cụ thể cần giải quyết
- Challenges trong recommendation
- Gap trong literature

### 1.3 Research Objectives
- Mục tiêu chính (implement CF và Content-based)
- So sánh performance hai methods
- Đánh giá trên implicit feedback dataset

### 1.4 Scope and Limitations
- Phạm vi nghiên cứu (MovieLens 1M, 2 methods)
- Limitations (không cover deep learning, hybrid advanced)

### 1.5 Report Organization
- Overview structure của báo cáo

---

## Chapter 2: Literature Review & Theoretical Background (8 trang)

### 2.1 Recommendation Systems Overview (1 trang)

**Người chịu trách nhiệm:** Người Phụ A

- Definition của recommendation systems
- Applications (ecommerce, streaming, social media)
- Classification of approaches:
  - Content-based
  - Collaborative Filtering
  - Hybrid
  - Knowledge-based
  - Deep learning-based (brief mention)

### 2.2 Collaborative Filtering (4 trang)

**Người chịu trách nhiệm:** Người 1

#### 2.2.1 User-based Collaborative Filtering (1 trang)
- Intuition: "Users like you also liked..."
- Similarity metrics (Cosine, Pearson)
- Ưu điểm và nhược điểm
- Example minh họa

#### 2.2.2 Item-based Collaborative Filtering (1 trang)
- Intuition: "Customers who bought X also bought Y..."
- Why more scalable than user-based
- Item similarity computation
- Ưu điểm và nhược điểm

#### 2.2.3 Matrix Factorization (1 trang)
- Concept: decompose user-item matrix thành latent factors
- User embedding × Item embedding
- Intuition về latent factors (không đi sâu công thức)
- Example: Movie preferences = (Action, Romance, Comedy) scores

#### 2.2.4 ALS for Implicit Feedback (1 trang)
- Paper reference: Hu, Koren, Volinsky (2008)
- Explicit vs Implicit feedback
- Confidence weighting concept
- Why ALS suitable cho ecommerce data
- Brief algorithm overview (không heavy math)

**Key References:**
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. ICDM.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. IEEE Computer.

### 2.3 Content-based Filtering (2 trang)

**Người chịu trách nhiệm:** Người 2

#### 2.3.1 Feature Extraction
- Item features (text, category, metadata)
- Feature representation methods

#### 2.3.2 TF-IDF and Similarity Measures
- TF-IDF intuition và formula
- Cosine similarity
- Similarity-based recommendation

#### 2.3.3 Advantages for Cold-start
- New item problem solution
- No need interaction history
- Trade-offs vs CF

### 2.4 Evaluation Metrics for Ranking (1 trang)

**Người chịu trách nhiệm:** Người 2

- Why not RMSE/MAE for ranking tasks
- **Precision@K**: accuracy trong top-K
- **Recall@K**: coverage của relevant items
- **NDCG@K**: reward higher ranks
- **MAP@K**: mean average precision
- Trade-offs between metrics

---

## Chapter 3: Dataset & Preprocessing (5 trang)

**Người chịu trách nhiệm:** Người 1

### 3.1 MovieLens 1M Overview (2 trang)

#### Data Structure
- **ratings.dat**: 1,000,209 ratings
  - Fields: user_id, movie_id, rating (1-5), timestamp
- **movies.dat**: 3,883 movies
  - Fields: movie_id, title, genres
- **users.dat**: 6,040 users
  - Fields: user_id, gender, age, occupation, zip

#### Statistics
- Table: Basic statistics summary
  - Total ratings, users, movies
  - Rating distribution (1-5)
  - Average ratings per user/movie
  - Sparsity: ~95.8% sparse
- Timespan của dataset

#### Data Characteristics
- Explicit ratings (1-5 stars)
- Temporal information available
- Multiple genres per movie
- User demographic info (optional use)

### 3.2 Exploratory Data Analysis (2 trang)

**Visualizations required:**

1. **Rating Distribution** (histogram)
   - Frequency của mỗi rating (1-5)
   - Mean rating: ~3.58

2. **User Interaction Distribution** (histogram + log scale)
   - Long-tail: few power users, many casual users
   - Percentiles (P50, P75, P90, P95)

3. **Item Popularity Distribution** (histogram + log scale)
   - Long-tail: few blockbusters, many niche movies
   - Percentiles

4. **Ratings Over Time** (line chart)
   - Temporal patterns
   - Activity trends

5. **Sparsity Heatmap** (100x100 sample)
   - Visualize sparse nature

6. **Genre Distribution** (bar chart)
   - Most common genres
   - Genre co-occurrence

7. **Top 20 Popular Movies** (bar chart with titles)

**Key Findings:**
- Long-tail distribution (Pareto principle)
- High sparsity challenges
- Temporal patterns (if any)

### 3.3 Data Preprocessing (1 trang)

#### Filtering Strategy
- **Min interactions threshold:** 5
  - Remove users with <5 ratings
  - Remove movies with <5 ratings
- **Rationale:** reduce noise, improve model quality
- **Impact:** số lượng users/items sau filter

#### Train/Test Split
- **Method:** Time-based split (80/20)
  - Sort by timestamp
  - First 80% → train
  - Last 20% → test
- **Why time-based:**
  - Avoid data leakage
  - Realistic evaluation (predict future from past)
  - Industry standard practice
- **Statistics:**
  - Train set: ~800K ratings
  - Test set: ~200K ratings

#### Data Transformation
- Convert to sparse matrix (scipy.sparse)
- User/item ID mapping (continuous indices)

---

## Chapter 4: Methodology (7 trang)

### 4.1 Experimental Setup (1 trang)

**Người chịu trách nhiệm:** Người 1

#### Hardware & Software Environment
- CPU/GPU specs (if applicable)
- RAM
- OS

#### Implementation Libraries
- Python 3.x
- pandas, numpy, scipy
- scikit-learn
- **implicit** library (for ALS)
- matplotlib, seaborn (visualization)

#### Code Repository
- GitHub/GitLab link (if available)

### 4.2 Baseline Model: Popularity-based (1 trang)

**Người chịu trách nhiệm:** Người 1

#### Algorithm
- Rank items by total interaction count
- Recommend top-N popular items to all users
- No personalization

#### Implementation
```python
class PopularityRecommender:
    def fit(self, train_df):
        self.popular_items = train_df['movie_id'].value_counts().index
    
    def recommend(self, user_id, n=10):
        return self.popular_items[:n]
```

#### Why Baseline
- Minimum performance threshold
- Surprisingly effective for new/cold users
- Benchmark for personalized methods

### 4.3 Collaborative Filtering with ALS (3 trang)

**Người chịu trách nhiệm:** Người 1

#### Model Architecture
- Matrix Factorization approach
- User matrix U (6040 × K)
- Item matrix V (3883 × K)
- K = latent factors (hyperparameter)

#### Alternating Least Squares (ALS)
- Optimization algorithm
- Fix U, optimize V
- Fix V, optimize U
- Iterate until convergence

#### Confidence Weighting
- Explicit ratings → implicit confidence
- Higher rating = higher confidence
- Confidence formula (if applicable)

#### Hyperparameter Configuration

**Grid Search:**
| Parameter | Values Tested |
|-----------|---------------|
| Factors (K) | [20, 50, 100] |
| Regularization | [0.01, 0.1] |
| Iterations | [15, 20, 30] |

**Total combinations:** 18

#### Training Procedure
1. Load train matrix (user-item sparse matrix)
2. Initialize model with hyperparameters
3. Fit model using ALS
4. Generate recommendations (top-N per user)

#### Computational Complexity
- Time complexity: O(iterations × K × (users + items) × nnz)
- Training time per configuration: ~X seconds

### 4.4 Content-based Filtering (2 trang)

**Người chịu trách nhiệm:** Người 2

#### Feature Engineering
- Extract genres from movies.dat
- TF-IDF vectorization on genres
- Movie feature vectors (3883 × vocab_size)

#### Similarity Computation
- Cosine similarity between all movie pairs
- Similarity matrix (3883 × 3883)

#### Recommendation Generation
1. Get user's watched movies from train
2. For each watched movie, find K most similar
3. Aggregate similarity scores
4. Rank and return top-N

---

## Chapter 5: Experimental Results (7 trang)

### 5.1 Evaluation Protocol (1 trang)

**Người chịu trách nhiệm:** Người 2

#### Metrics Used
- Precision@K
- Recall@K
- NDCG@K
- K values: [5, 10, 20]

#### Evaluation Process
- For each user in test set:
  - Get true relevant items (rated ≥4)
  - Generate top-K recommendations
  - Compute metrics
- Average across all users

#### Implementation
```python
def evaluate_model(model, test_df, k=10):
    # Pseudo-code
    precision = compute_precision_at_k()
    recall = compute_recall_at_k()
    ndcg = compute_ndcg_at_k()
    return {precision, recall, ndcg}
```

### 5.2 Baseline Results (1 trang)

**Người chịu trách nhiệm:** Người 1

#### Popularity Model Performance

**Results Table:**
| Metric | K=5 | K=10 | K=20 |
|--------|-----|------|------|
| Precision@K | 0.XXX | 0.XXX | 0.XXX |
| Recall@K | 0.XXX | 0.XXX | 0.XXX |
| NDCG@K | 0.XXX | 0.XXX | 0.XXX |

#### Observations
- Decent performance despite no personalization
- Good for cold-start users
- Ceiling for comparison

### 5.3 Collaborative Filtering Results (3 trang)

**Người chịu trách nhiệm:** Người 1

#### Hyperparameter Tuning Results

**Grid Search Table (top 5 configs):**
| Factors | Reg | Iterations | P@10 | R@10 | NDCG@10 |
|---------|-----|------------|------|------|---------|
| 100 | 0.01 | 30 | 0.XXX | 0.XXX | 0.XXX |
| 50 | 0.01 | 30 | 0.XXX | 0.XXX | 0.XXX |
| ... | ... | ... | ... | ... | ... |

**Visualization: Bar chart of top 5 configs**

#### Best Configuration
- **Factors:** 100 (hoặc best value)
- **Regularization:** 0.01
- **Iterations:** 30
- **Rationale:** highest NDCG@10 with reasonable training time

#### Performance Comparison with Baseline

**Table:**
| Model | P@10 | R@10 | NDCG@10 |
|-------|------|------|---------|
| Popularity | 0.XXX | 0.XXX | 0.XXX |
| ALS (best) | 0.XXX | 0.XXX | 0.XXX |
| **Improvement** | +X% | +X% | +X% |

**Visualization: Side-by-side bar chart**

#### Metrics vs K Analysis

**Line Chart:** Precision@K, Recall@K, NDCG@K for K=[5,10,20]

**Observations:**
- Recall increases with K (expected)
- Precision decreases with K (expected)
- NDCG shows diminishing returns after K=X

### 5.4 Content-based Results (2 trang)

**Người chịu trách nhiệm:** Người 2

#### Performance Metrics

**Results Table:**
| Metric | K=5 | K=10 | K=20 |
|--------|-----|------|------|
| Precision@K | 0.XXX | 0.XXX | 0.XXX |
| Recall@K | 0.XXX | 0.XXX | 0.XXX |
| NDCG@K | 0.XXX | 0.XXX | 0.XXX |

#### Feature Importance Analysis
- Most discriminative genres
- Genre co-occurrence patterns

---

## Chapter 6: Comparison & Analysis (4 trang)

**Người chịu trách nhiệm:** Người 2 (lead), Người 1 (collaborate)

### 6.1 Quantitative Comparison (2 trang)

#### Overall Performance Table
| Model | P@5 | P@10 | P@20 | R@5 | R@10 | R@20 | NDCG@10 |
|-------|-----|------|------|-----|------|------|---------|
| Popularity | X | X | X | X | X | X | X |
| ALS (CF) | X | X | X | X | X | X | X |
| Content-based | X | X | X | X | X | X | X |

**Visualization: Grouped bar chart for all metrics**

#### Statistical Significance Tests
- Paired t-test between ALS vs Baseline
- Paired t-test between ALS vs Content-based
- P-values và significance levels

#### Analysis by User Activity Level
- Split users: Low activity (<20 ratings), Medium (20-50), High (>50)
- Performance comparison per segment

**Table:**
| Model | Low Activity | Medium | High |
|-------|--------------|--------|------|
| Popularity | X | X | X |
| ALS | X | X | X |
| Content | X | X | X |

**Finding:** CF performs better on high-activity users, Content better on low-activity

### 6.2 Qualitative Analysis (2 trang)

#### Case Studies (5 users)

**User 1:** High activity, diverse tastes
- History: [Movie titles]
- CF recommendations: [Titles with explanations]
- Content recommendations: [Titles with explanations]
- **Analysis:** CF captures collaborative patterns, Content focuses on genre similarity

**User 2:** Low activity, specific genre preference
- Similar structure

**Users 3-5:** Diverse profiles

#### Beyond-Accuracy Metrics

**Coverage:**
- % of items recommended at least once
- CF coverage vs Content coverage

**Diversity:**
- Intra-list diversity (avg pairwise distance)
- CF tends to recommend popular items (filter bubble)
- Content provides more diverse genres

**Novelty:**
- Avg popularity rank of recommendations
- CF: more popular items
- Content: more niche items

**Table:**
| Model | Coverage (%) | Diversity | Avg Novelty Rank |
|-------|--------------|-----------|-------------------|
| Popularity | X | X | X |
| ALS | X | X | X |
| Content | X | X | X |

#### When Each Method Performs Better

**CF (ALS) excels when:**
- User has rich interaction history
- Collaborative patterns exist
- Items have sufficient interaction data

**Content-based excels when:**
- New items (cold-start)
- New users with few ratings
- Strong genre preferences

---

## Chapter 7: Discussion (3 trang)

**Người chịu trách nhiệm:** Người 1 + Người 2 collaborate

### 7.1 Key Findings

- ALS outperforms baseline by X% on NDCG@10
- CF better for experienced users, Content better for cold-start
- Sparsity remains challenge for CF
- Content-based provides better diversity

### 7.2 CF vs Content-based Trade-offs

#### Collaborative Filtering Strengths:
- Discovers latent patterns
- No need item features
- Effective for popular items

#### CF Weaknesses:
- Cold-start problem (new users/items)
- Sparsity challenges
- Popularity bias

#### Content-based Strengths:
- No cold-start for new items
- Explainable recommendations
- Better diversity

#### Content Weaknesses:
- Limited by feature quality
- Over-specialization (filter bubble)
- No serendipity

### 7.3 Cold-start Problem Analysis

#### New User Scenario
- CF: cannot recommend (no history)
- Content: can use demographics (not implemented)
- Baseline: use popularity

#### New Item Scenario
- CF: cannot recommend (no interactions)
- Content: can recommend immediately (based on features)

### 7.4 Limitations

- Dataset: MovieLens may not fully represent ecommerce
- Methods: didn't implement hybrid or deep learning
- Evaluation: only offline metrics (no A/B testing)
- Features: only genres used (no text, images)

### 7.5 Practical Implications

- Use CF for main users
- Use Content for cold-start
- Combine both in production (hybrid)
- Monitor diversity và fairness

---

## Chapter 8: Conclusion & Future Work (2 trang)

**Người chịu trách nhiệm:** Người Phụ B

### 8.1 Summary of Contributions

- Implemented và compared CF (ALS) vs Content-based on MovieLens 1M
- Comprehensive evaluation with ranking metrics
- Analysis of trade-offs and use cases
- Insights on cold-start problem

### 8.2 Future Directions

#### Hybrid Approaches
- Weighted combination of CF + Content
- Switching hybrid based on data availability
- LightFM (hybrid matrix factorization)

#### Deep Learning Methods
- Neural Collaborative Filtering (NCF)
- Autoencoders for recommendation
- Transformer-based models (BERT4Rec)

#### Scalability Improvements
- Approximate Nearest Neighbors (ANN)
- FAISS for large-scale retrieval
- Distributed training (Spark MLlib)

#### Additional Features
- Text descriptions (BERT embeddings)
- Images (CNN features)
- Temporal dynamics (session-based)

#### Production Considerations
- Real-time recommendation serving
- A/B testing framework
- Monitoring và feedback loops

---

## References (1-2 trang)

**Người chịu trách nhiệm:** Người Phụ A

**Format:** IEEE or ACM style

**Key Papers:**

1. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *2008 Eighth IEEE International Conference on Data Mining*, 263-272.

2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.

3. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.

4. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1-19.

**Online Resources:**

5. Implicit library documentation: https://github.com/benfred/implicit

6. Scikit-learn documentation

---

## Appendix (1 trang)

**Người chịu trách nhiệm:** Người Phụ C

### A. Hyperparameter Grid Search Results (Full Table)

Complete table với tất cả 18 configurations

### B. Additional Visualizations

- Full sparsity heatmap
- User/item distribution plots
- Genre co-occurrence matrix

### C. Code Repository

- Link to GitHub/GitLab
- File structure overview
- Usage instructions

---

## Phân Công Chi Tiết

| Người | Chapters | Trang | Nội dung chính |
|-------|----------|-------|----------------|
| **Người 1** | 2.2, 3, 4.1-4.3, 5.2-5.3, phần của 6-7 | ~20 | CF theory, Dataset, EDA, ALS implementation, CF results |
| **Người 2** | 2.3-2.4, 4.4, 5.1, 5.4, 6 (lead), phần của 7 | ~18 | Content-based theory, implementation, evaluation metrics, comparison |
| **Người Phụ A** | 1, 2.1, References | ~5 | Introduction, RecSys overview, References |
| **Người Phụ B** | 8 | ~2 | Conclusion & Future work |
| **Người Phụ C** | Abstract, Appendix | ~2 | Abstract, Appendix |

**Tổng:** 35-40 trang

---

## Timeline

| Tuần | Người 1 | Người 2 | Deliverable |
|------|---------|---------|-------------|
| 1 | Data + EDA + preprocessing | Research content-based methods | Train/test split, EDA notebook |
| 2 | Baseline + ALS implementation | Content-based implementation | Models working |
| 3 | CF evaluation + tuning | Implement metrics + evaluate both | Results tables |
| 4 | Write chapters 2.2, 3, 4.1-4.3, 5.2-5.3 | Write chapters 2.3-2.4, 4.4, 5.1, 5.4, 6 | Draft report |

**Post Tuần 4:**
- Người Phụ A, B, C: write assigned chapters
- Tất cả: review và finalize
