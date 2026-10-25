# Brainstorm Summary: Hoàn Thiện Yêu Cầu Tuần 1 - Recommendation System

**Ngày:** 2 tháng 9, 2026  
**Phạm vi:** Tuần 1 - Nền tảng lý thuyết + Dataset Analysis + Đề cương báo cáo  
**Mục tiêu:** Chuẩn bị foundation vững chắc trước khi implement code (Tuần 2+)

---

## Problem Statement

Hoàn thiện các deliverables của Tuần 1 trong roadmap 8 tuần xây dựng Recommendation System cho ecommerce:
1. Tài liệu lý thuyết đầy đủ về RecSys (beginner-friendly)
2. Phân tích chi tiết dataset Retailrocket (EDA với visualizations)
3. Đề cương báo cáo học thuật hoàn chỉnh (chuẩn IEEE/ACM)

---

## Dataset Context

**Retailrocket Ecommerce Dataset** (`/data/raw/`):
- `events.csv`: 2,756,102 interactions
  - 1,407,580 unique visitors
  - 235,061 unique items
  - Event types: view (96.7%), addtocart (2.5%), transaction (0.8%)
  - Timespan: June 2015 (cần verify exact range)
- `item_properties_part1/2.csv`: 20M+ property rows
  - Properties: categoryid, availability, và ~900 property codes khác
- `category_tree.csv`: 1,670 categories với hierarchical structure

**Key Characteristics:**
- Implicit feedback (không có rating)
- Extreme sparsity: ~0.0008% density
- Temporal data: cho phép time-based split
- Multi-modal signals: 3 event types với semantic khác nhau

---

## Requirements Analysis

### User Constraints
- **Skill level:** Beginner (cần giải thích concepts từ cơ bản)
- **Language:** Tiếng Việt (docs & comments), có thể bilingual cho báo cáo
- **Report format:** Academic (IEEE/ACM style)
- **Data processing:** Sample 10% cho EDA (tăng tốc độ)
- **Visualization:** matplotlib + seaborn + plotly
- **Timeline:** Flexible, ưu tiên chất lượng (~8-10 giờ)
- **Stopping point:** Dừng sau Tuần 1, review trước khi sang Tuần 2

### Success Criteria
✅ Lý thuyết: Beginner-friendly, có ví dụ thực tế, tránh công thức phức tạp  
✅ Dataset: EDA notebook với 5+ visualizations, statistics table, data quality report  
✅ Đề cương: Abstract + Intro đầy đủ, Section 2-3 filled, Section 4-7 templates  
✅ Documentation: README cập nhật, code comments rõ ràng

---

## Proposed Solution

### Architecture: 3-Track Parallel Approach

```
Track 1: Theory Document          Track 2: Dataset Analysis         Track 3: Report Outline
  (2-3 giờ)                          (3-4 giờ)                        (2 giờ)
      │                                   │                                │
      ├─ RecSys Overview                  ├─ Load & Sample Data            ├─ Abstract
      ├─ CF vs Content-based              ├─ Basic Statistics              ├─ Introduction
      ├─ Implicit Feedback                ├─ Event Analysis                ├─ Related Work
      ├─ ALS Intuition                    ├─ Item Properties               ├─ Dataset (from Track 2)
      ├─ Evaluation Metrics               ├─ Category Tree                 ├─ Methodology (template)
      └─ Cold-start Problem               ├─ Conversion Funnel             ├─ Experiments (template)
                                          ├─ Temporal Patterns             ├─ Discussion (template)
                                          ├─ Sparsity Analysis             └─ Conclusion (template)
                                          └─ Train/Test Strategy
                                          
                        Integration Phase (1 giờ)
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
            Cross-reference   Consistency   Summary Report
                              Check
```

### Track 1: Tài Liệu Lý Thuyết

**File:** `docs/01-ly-thuyet-recommendation-systems.md`

**Sections:**
1. **Giới Thiệu RecSys**
   - Định nghĩa, use cases (Amazon, Netflix, Shopee)
   - Vai trò trong ecommerce
   - Personalization vs Generalization

2. **Phân Loại Phương Pháp**
   - Content-based: dựa vào features sản phẩm
   - Collaborative Filtering:
     * User-based: "Users like you also liked..."
     * Item-based: "Customers who bought X also bought Y..."
     * Matrix Factorization (ALS)
   - Hybrid approaches
   - Deep Learning (overview only: NCF, Neural CF, Transformers)

3. **Explicit vs Implicit Feedback**
   - Explicit: rating 1-5 ⭐ → clear preference signal
   - Implicit: view/click/purchase → infer preference
   - Why Retailrocket = implicit
   - Challenges: no negative feedback, noisy signals

4. **Matrix Factorization & ALS**
   - Intuition: User preferences = hidden factors
   - Example: Movie preferences = (Action, Romance, Comedy) scores
   - ALS optimization: alternating between user & item factors
   - Confidence weighting cho implicit (paper Hu et al. 2008)
   - KHÔNG đi sâu công thức, focus visualization

5. **Cold-Start Problem**
   - New user: không có history → dùng popularity/demographics
   - New item: không có interactions → dùng content-based
   - Hybrid solutions

6. **Evaluation Metrics**
   - Tại sao KHÔNG dùng RMSE/MAE cho implicit
   - **Precision@K**: accuracy trong top-K
   - **Recall@K**: coverage của relevant items
   - **NDCG@K**: reward higher ranks
   - **MAP@K**: mean average precision
   - Ví dụ minh họa từng metric

7. **Best Practices**
   - Time-based train/test split (tránh data leakage)
   - Filter users/items với ít interactions
   - Handle sparsity
   - Scalability considerations (ANN, FAISS cho large-scale)

**References to collect:**
- Hu, Koren, Volinsky (2008): Collaborative Filtering for Implicit Feedback
- Google ML Crash Course - RecSys
- Microsoft RecSys overview
- Implicit library docs
- LightFM docs

---

### Track 2: Dataset Analysis Notebook

**File:** `notebooks/week1-dataset-analysis.ipynb`

**Cell Structure:**

```python
# === CELL 1: Setup ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

%matplotlib inline
sns.set_style("whitegrid")

# === CELL 2: Load Data với Sampling ===
# Sample 10% của events.csv để tăng tốc
events_sample = pd.read_csv('data/raw/events.csv', 
                            skiprows=lambda i: i>0 and np.random.rand()>0.1)
item_props = pd.read_csv('data/raw/item_properties_part1.csv', nrows=1000000)
category_tree = pd.read_csv('data/raw/category_tree.csv')

print(f"Events sample: {len(events_sample):,} rows (~10% of 2.7M)")
print(f"Item properties: {len(item_props):,} rows")
print(f"Category tree: {len(category_tree):,} rows")

# === CELL 3: Basic Statistics ===
stats = {
    'Total Interactions (sample)': len(events_sample),
    'Unique Visitors': events_sample['visitorid'].nunique(),
    'Unique Items': events_sample['itemid'].nunique(),
    'Date Range': f"{events_sample['timestamp'].min()} to {events_sample['timestamp'].max()}",
    'Event Types': events_sample['event'].value_counts().to_dict()
}

# === CELL 4: Event Distribution ===
# Pie chart: view vs addtocart vs transaction
# Bar chart với percentages

# === CELL 5: User Interaction Distribution ===
# Histogram: interactions per user (log scale)
# Long-tail analysis
# Identify active vs casual users

# === CELL 6: Item Popularity Distribution ===
# Histogram: interactions per item (log scale)
# Top 20 most popular items
# Long-tail: % items với <5, <10, <50 interactions

# === CELL 7: Temporal Patterns ===
# Convert timestamp to datetime
# Interactions per day (line chart)
# Interactions per hour (heatmap)
# Weekly patterns

# === CELL 8: Conversion Funnel ===
# view → addtocart → transaction
# Conversion rates
# Sankey diagram (plotly)

# === CELL 9: Sparsity Analysis ===
# Matrix density calculation
# Visualization: heatmap of sampled user-item matrix

# === CELL 10: Item Properties Analysis ===
# Property types distribution
# CategoryID coverage (% items có category)
# Availability status

# === CELL 11: Category Tree ===
# Tree depth
# Top categories by item count
# Hierarchy visualization (if possible)

# === CELL 12: Data Quality Issues ===
# Missing values
# Duplicates
# Outliers (users với >1000 interactions)
# Items without properties

# === CELL 13: Proposed Implicit Weighting ===
weighting_scheme = {
    'view': 1,
    'addtocart': 2,
    'transaction': 3
}
# Rationale explanation

# === CELL 14: Train/Test Split Strategy ===
# Time-based split illustration
# Leave-one-out per user
# Data leakage warning

# === CELL 15: Summary Statistics Table ===
summary_df = pd.DataFrame({
    'Metric': [...],
    'Value': [...]
})
```

**Visualizations Required:**
1. Event type distribution (pie + bar chart)
2. User interaction histogram (log scale) + cumulative %
3. Item popularity histogram (log scale) + cumulative %
4. Temporal line chart (interactions over time)
5. Hourly heatmap (day of week vs hour)
6. Conversion funnel (Sankey diagram)
7. Sparsity heatmap (sampled matrix)
8. Property types bar chart
9. Category tree sunburst/treemap (plotly)

**Export:** `reports/week1-dataset-statistics.csv`

---

### Track 3: Đề Cương Báo Cáo

**File:** `reports/outline-bao-cao.md`

**Structure (IEEE/ACM Academic Format):**

```markdown
# Đề Cương Báo Cáo: Recommendation System cho Ecommerce 
# sử dụng Collaborative Filtering và Content-based Filtering

---

## Abstract (Tóm tắt) - 150 từ

Recommendation systems đóng vai trò then chốt trong việc cá nhân hóa trải nghiệm 
mua sắm trên các nền tảng ecommerce. Nghiên cứu này tập trung vào việc xây dựng 
và đánh giá hai phương pháp chính: Collaborative Filtering (sử dụng ALS cho 
implicit feedback) và Content-based Filtering, áp dụng trên dataset Retailrocket 
với 2.7 triệu interactions từ 1.4 triệu users trên 235 nghìn sản phẩm...

[Tiếp tục điền sau khi có results]

---

## 1. Introduction (Giới thiệu)

### 1.1 Background and Motivation

Trong thời đại thương mại điện tử phát triển mạnh mẽ, người dùng thường phải đối 
mặt với hàng triệu lựa chọn sản phẩm...

### 1.2 Research Objectives

Mục tiêu của nghiên cứu này là:
- Xây dựng recommendation system sử dụng collaborative filtering (ALS)
- So sánh với content-based filtering approach
- Đánh giá hiệu quả trên implicit feedback dataset
- Phân tích khả năng xử lý cold-start problem

### 1.3 Scope and Limitations

Phạm vi nghiên cứu tập trung vào...

### 1.4 Report Organization

Báo cáo được tổ chức thành 7 phần chính...

---

## 2. Theoretical Background and Related Work

### 2.1 Recommendation Systems Overview

#### 2.1.1 Content-based Filtering
#### 2.1.2 Collaborative Filtering
  - User-based CF
  - Item-based CF
  - Model-based CF (Matrix Factorization)
#### 2.1.3 Hybrid Approaches
#### 2.1.4 Deep Learning-based Methods (overview)

### 2.2 Matrix Factorization for Implicit Feedback

#### 2.2.1 Alternating Least Squares (ALS)
Paper reference: Hu, Koren, Volinsky (2008)...

#### 2.2.2 Confidence Weighting
Why different from explicit feedback...

### 2.3 Evaluation Strategies

#### 2.3.1 Metrics for Ranking Tasks
- Precision@K, Recall@K
- NDCG@K, MAP@K

#### 2.3.2 Train/Test Splitting
Why temporal split is critical...

### 2.4 Cold-Start Problem
Solutions and trade-offs...

---

## 3. Dataset Description

### 3.1 Retailrocket Dataset Overview

Dataset thu thập từ một nền tảng ecommerce thực tế trong khoảng thời gian...

### 3.2 Data Structure

#### 3.2.1 Events Data
- `timestamp`: Unix timestamp
- `visitorid`: Anonymous user identifier
- `event`: {view, addtocart, transaction}
- `itemid`: Product identifier
- `transactionid`: Transaction ID (if applicable)

#### 3.2.2 Item Properties
- `itemid`, `property`, `value`
- Properties include: categoryid, availability, etc.

#### 3.2.3 Category Tree
Hierarchical category structure...

### 3.3 Exploratory Data Analysis

[Điền từ notebook Track 2]

#### 3.3.1 Scale and Distribution
- 2,756,102 interactions
- 1,407,580 unique users
- 235,061 unique items
- Sparsity: ~0.0008%

#### 3.3.2 Event Type Distribution
- Views: 96.7%
- Add to cart: 2.5%
- Transactions: 0.8%

#### 3.3.3 User and Item Characteristics
Long-tail distribution analysis...

#### 3.3.4 Temporal Patterns
Seasonality, time-of-day effects...

### 3.4 Data Preprocessing

#### 3.4.1 Filtering Strategy
- Remove users với <5 interactions
- Remove items với <5 interactions
- Rationale...

#### 3.4.2 Implicit Feedback Weighting
Proposed weighting scheme:
- transaction: 3
- addtocart: 2
- view: 1

#### 3.4.3 Train/Test Split
Time-based split: 80/20 earliest vs latest...

---

## 4. Methodology

[Template - điền ở Tuần 3-5]

### 4.1 Baseline Models

#### 4.1.1 Popularity-based Recommendation
#### 4.1.2 Random Baseline

### 4.2 Collaborative Filtering with ALS

#### 4.2.1 Model Architecture
#### 4.2.2 Hyperparameter Configuration
- Latent factors: 64, 128, 256
- Regularization: 0.01, 0.1
- Iterations: 15, 30
#### 4.2.3 Training Procedure

### 4.3 Content-based Filtering

#### 4.3.1 Feature Engineering
- TF-IDF on category and properties
#### 4.3.2 Similarity Computation
- Cosine similarity
#### 4.3.3 Recommendation Generation

### 4.4 Hybrid Approach (Optional)

---

## 5. Experimental Results

[Template - điền ở Tuần 4-6]

### 5.1 Experimental Setup
### 5.2 Evaluation Metrics
### 5.3 Baseline Results
### 5.4 ALS Results
### 5.5 Content-based Results
### 5.6 Comparison and Analysis

---

## 6. Discussion

[Template - điền ở Tuần 6-7]

### 6.1 Key Findings
### 6.2 Limitations
### 6.3 Cold-Start Handling
### 6.4 Scalability Considerations

---

## 7. Conclusion and Future Work

[Template - điền ở Tuần 7-8]

### 7.1 Summary
### 7.2 Contributions
### 7.3 Future Directions
### 7.4 Practical Applications

---

## References

[To be collected - IEEE format]

1. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for 
   implicit feedback datasets. ICDM 2008.
   
2. [More references to be added]
```

---

## Implementation Plan

### Phase 1: Theory Document (2-3 giờ)
- [ ] Tạo `docs/01-ly-thuyet-recommendation-systems.md`
- [ ] Viết sections 1-7 với beginner focus
- [ ] Thêm ví dụ minh họa
- [ ] Collect references
- [ ] Review & refine

### Phase 2: Dataset Analysis (3-4 giờ)
- [ ] Setup notebook environment
- [ ] Implement sampling strategy
- [ ] Basic statistics & distributions
- [ ] Create 9 visualizations
- [ ] Data quality analysis
- [ ] Export summary CSV
- [ ] Document findings

### Phase 3: Report Outline (2 giờ)
- [ ] Create outline structure
- [ ] Write Abstract draft
- [ ] Complete Introduction
- [ ] Fill Section 2 (from theory doc)
- [ ] Fill Section 3 (from notebook)
- [ ] Create templates for Sections 4-7
- [ ] Collect initial references

### Phase 4: Integration (1 giờ)
- [ ] Cross-reference documents
- [ ] Ensure consistency
- [ ] Update README.md
- [ ] Create summary report
- [ ] Prepare for user review

---

## Technical Decisions & Rationale

### Decision 1: Sample 10% of Events Data
**Rationale:** 
- Beginner project → prioritize learning over scale
- Faster iteration cycles
- Still statistically significant (~275K interactions)
- Can scale to full data later if needed

**Trade-off:** May miss rare patterns in long tail

### Decision 2: Vietnamese Documentation
**Rationale:**
- Faster comprehension for beginner
- Faster writing speed
- Can translate to English in Week 7

**Trade-off:** Final report needs translation effort

### Decision 3: Both matplotlib + plotly
**Rationale:**
- matplotlib: publication-quality static charts
- plotly: interactive exploration
- Need to install plotly (not in requirements.txt)

**Action:** Add `plotly>=5.0.0` to requirements.txt

### Decision 4: Time-based Split Strategy
**Rationale:**
- Avoid data leakage (critical for recSys)
- Realistic evaluation (predict future from past)
- Industry standard

**Trade-off:** None - this is best practice

### Decision 5: Confidence Weighting (3:2:1)
**Rationale:**
- Transaction >> Addtocart > View in terms of preference signal
- Conservative ratios (not 10:5:1) due to class imbalance
- Can tune later based on results

**Trade-off:** Somewhat arbitrary, needs experimentation

---

## Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dataset quá lớn, xử lý chậm | Medium | High | Sample 10%, filter time window, optimize pandas operations |
| Item properties không có documentation | High | Medium | Reverse-engineer từ data, document assumptions |
| Lý thuyết quá phức tạp cho beginner | Medium | High | Dùng analogies, visual examples, tránh công thức |
| Visualizations không đẹp/rõ | Low | Low | Use seaborn defaults + plotly templates |
| Mất deadline | Low | Medium | Time-box mỗi phase, focus essentials first |
| Code lỗi do version conflicts | Low | Medium | Test imports trước, có fallback cho plotly |

---

## Expected Outputs (Deliverables)

```
RecommendationSystem/
├── docs/
│   └── 01-ly-thuyet-recommendation-systems.md  [NEW - 2-3K words]
├── notebooks/
│   └── week1-dataset-analysis.ipynb            [NEW - 15 cells, 9 charts]
├── reports/
│   ├── brainstorm-week1-summary.md             [THIS FILE]
│   ├── outline-bao-cao.md                      [NEW - Full structure]
│   └── week1-dataset-statistics.csv            [NEW - Exported stats]
├── README.md                                    [UPDATED - Progress tracker]
└── requirements.txt                             [UPDATED - Add plotly]
```

---

## Success Metrics

**Quantitative:**
- ✅ Theory doc: ≥2000 words, ≥7 sections, ≥5 references
- ✅ Notebook: ≥15 code cells, ≥9 visualizations, 0 errors
- ✅ Report outline: ≥7 sections, Abstract + Intro complete
- ✅ Statistics: ≥15 key metrics documented

**Qualitative:**
- ✅ Beginner can understand theory without ML background
- ✅ Visualizations clearly show data characteristics
- ✅ Report outline provides clear roadmap for Weeks 2-8
- ✅ Code is well-commented and reusable

---

## Alternative Approaches Considered

### Alternative 1: Start with Code Implementation
**Pros:** Faster tangible progress, hands-on learning
**Cons:** No theoretical foundation, harder to debug, trial-and-error approach
**Decision:** ❌ Rejected - foundation first approach is more systematic

### Alternative 2: Use Full Dataset from Start
**Pros:** No information loss, more accurate statistics
**Cons:** Slower iteration, may frustrate beginner, not necessary for learning
**Decision:** ❌ Rejected - 10% sample is sufficient for Week 1

### Alternative 3: Skip Report Outline, Focus on Code
**Pros:** More coding practice
**Cons:** No clear roadmap, harder to write report later
**Decision:** ❌ Rejected - outline ensures structured approach

### Alternative 4: English Documentation
**Pros:** No translation needed later
**Cons:** Slower writing, harder comprehension for beginner
**Decision:** ❌ Rejected - Vietnamese for efficiency, translate in Week 7

---

## Post-Week 1 Roadmap Preview

After completing Week 1 deliverables, the next steps are:

**Week 2:** Data Preprocessing Pipeline
- Implement filtering logic (min interactions threshold)
- Build user-item sparse matrix
- Implement confidence weighting
- Create train/test split function
- Save processed data for modeling

**Week 3:** Baseline + ALS Implementation
- Popularity baseline
- Random baseline
- ALS model with `implicit` library
- Hyperparameter tuning
- Generate recommendations

**Week 4:** Evaluation & Metrics
- Implement Precision@K, Recall@K, NDCG@K
- Evaluate all models
- Create comparison tables
- Analyze results

...continuing through Week 8.

---

## Approval & Next Actions

**This brainstorm is complete.** 

**Approved solution:**
- 3-track parallel approach
- 10% data sampling
- Vietnamese documentation
- matplotlib + plotly visualizations
- Stop after Week 1 for review

**Next steps:**
1. ✅ Brainstorm complete (this document)
2. ⏸️ **STOP - Waiting for user approval**
3. ⏭️ If approved: Begin implementation (Theory → Dataset → Outline)
4. ⏭️ After implementation: User review before Week 2

---

## Questions for Future Consideration

**Dataset-related:**
- Có nên lọc bots/outliers (users với >1000 interactions)?
- Property codes (888, 790, etc.) có documentation ở đâu không?
- Cần merge item_properties_part1 và part2 không?

**Modeling-related:**
- ALS factors: bắt đầu với 64 hay 128?
- Regularization: strong (0.1) hay weak (0.01)?
- Có nên thử weighted ALS với 3:2:1 ratios ngay không?

**Evaluation-related:**
- K values cho @K metrics: 5, 10, 20?
- Leave-one-out hay time-based 80/20?
- Có cần cross-validation không?

→ Những câu hỏi này sẽ được giải quyết trong Weeks 2-4.

---

## References for Week 1

### Papers
1. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. ICDM 2008.
2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. IEEE Computer.

### Online Resources
3. Google ML Crash Course - Recommendation Systems
4. Microsoft RecSys Tutorial
5. Implicit library documentation: https://github.com/benfred/implicit

### Documentation
6. Pandas: https://pandas.pydata.org/docs/
7. Matplotlib: https://matplotlib.org/stable/contents.html
8. Plotly: https://plotly.com/python/
9. Seaborn: https://seaborn.pydata.org/

---

## Changelog

- **2026-09-02:** Initial brainstorm session
  - Analyzed dataset structure (2.7M events, 1.4M users, 235K items)
  - Confirmed user requirements (beginner, Vietnamese, academic format)
  - Proposed 3-track approach
  - User approved: sample 10%, both viz libraries
  - Created comprehensive brainstorm summary

---

**Status:** ✅ Brainstorm Complete - Awaiting Implementation Approval  
**Estimated Implementation Time:** 8-10 hours  
**Next Document:** `/docs/01-ly-thuyet-recommendation-systems.md`