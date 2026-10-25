# Plan: BPR/SVD/SVD++ Vietnamese Guide on MovieLens 1M

## 1. Task Summary

Produce a **concise implementation plan** for a new Vietnamese Markdown guide that explains and teaches:
- **BPR** (Bayesian Personalized Ranking)
- **SVD** (Singular Value Decomposition)
- **SVD++**

Using the **MovieLens 1M** dataset. **Documentation only** — no code modifications to the existing project.

---

## 2. Recommended Libraries & Dependency Commands

### 2.1 BPR → `implicit`

```bash
pip install implicit==0.7.2
```

- Handles BPR natively via `implicit.lms.LMS()` (logistic matrix factorization)
- Also offers `implicit.bpr.BayesianPersonalizedRanking()`
- Accepts CSR sparse user-item matrix (implicit, binary or count-weighted)
- GPU-accelerated via CuPy (optional)
- Compatible with Python 3.8–3.11

### 2.2 Explicit SVD/SVD++ → `surprise`

```bash
pip install scikit-surprise==1.1.3
```

- `surprise.SVD` — classic SVD with bias terms (Funk SVD)
- `surprise.SVDpp` — SVD++ (incorporates implicit feedback signal from all items rated by user)
- Both operate on explicit `(user, item, rating)` tuples via `surprise.Dataset`
- Built-in cross-validation, grid search, RMSE/MAE metrics

### 2.3 Shared Dependencies

```bash
pip install pandas==2.0.3 numpy==1.24.3 scipy==1.11.1 matplotlib==3.7.2 seaborn==0.12.2
```

---

## 3. Key Theoretical Differences

| Dimension | BPR (`implicit`) | SVD (`surprise.SVD`) | SVD++ (`surprise.SVDpp`) |
|---|---|---|---|
| **Feedback type** | Implicit (binary) | Explicit (ratings 1–5) | Explicit (ratings 1–5) |
| **Objective** | Maximize pairwise ranking probability | Minimize RMSE on observed ratings | Minimize RMSE on observed ratings |
| **Input matrix** | User-item interaction (0/1 or counts) | User-item-rating triplets | User-item-rating triplets + implicit set |
| **Output** | Ranked item list per user | Predicted rating per user-item | Predicted rating per user-item |
| **Training time** | Faster on sparse implicit data | Fast (~minutes on ML-1M) | Slower (incorporates all items per user) |
| **Typical runtime** | 5–20 min on ML-1M | 2–10 min on ML-1M | 20–60 min on ML-1M |

### 3.1 Evaluation Caveats

| Concern | Detail |
|---|---|
| **Rating-based metrics for BPR** | BPR optimizes ranking, not rating prediction. Applying RMSE/MAE to BPR outputs is **misleading** — it evaluates a different objective. Use ranking metrics (Precision@K, Recall@K, NDCG@K, AUC) for BPR. |
| **BPR on explicit data** | MovieLens 1M has explicit ratings. To use BPR, convert to binary implicit (e.g., rating ≥ 4 → 1, else 0). This discards rating magnitude. |
| **SVD/SVD++ on implicit data** | Applying SVD/SVD++ to binary implicit data produces uncalibrated outputs — RMSE is meaningless across 0/1 targets. |
| **Cold-start test items** | If test items have zero training interactions, all three models will fail. Filter warm items only. |
| **Fair comparison** | To compare fairly, run all three on the same user/item sets and same train/test split. Use **ranking metrics** for all, or **rating metrics** for SVD/SVD++ only. Do not mix metric families. |

---

## 4. Critical Correction: BPR vs ALS

### Popular Misconception

> *"ALS (Alternating Least Squares) is for implicit data, BPR is for explicit data — use ALS when you have ratings."*

This is **incorrect** and widely propagated. The actual distinction:

| | ALS | BPR |
|---|---|---|
| **What it optimizes** | Weighted squared error on observed interactions | Pairwise ranking loss on triplets |
| **Feedback type it accepts** | Implicit OR explicit (via confidence weighting) | Implicit (binary) OR explicit converted to binary |
| **Library** | `implicit.als.AlternatingLeastSquares` | `implicit.bpr.BayesianPersonalizedRanking` |
| **Output** | User/item latent factors | User/item latent factors |

**Correct statement:** ALS and BPR are **different optimization objectives** for learning latent factors. Neither is intrinsically "for implicit" or "for explicit" — both can handle either, but with different assumptions. The guide must explicitly debunk this misconception.

---

## 5. Document Outline

### File name suggestion
```
notes/bpr-svd-svdpp-guide.md
```

### Proposed Structure (targeting novice Vietnamese students)

```
# Hướng Dẫn Triển Khai BPR, SVD, SVD++ trên MovieLens 1M

## Mục Lục

## 1. Giới Thiệu & Kiến Thức Nền Tảng (Lý Thuyết)
  1.1 Giới thiệu bài toán Recommendation
  1.2 Ma trận User-Item: Explicit vs Implicit Feedback
  1.3 Matrix Factorization là gì? Intuition bằng ví dụ đơn giản
  1.4 BPR là gì? Maximum A Posteriori + Pairwise Ranking
  1.5 SVD là gì? Funk SVD với Bias Terms
  1.6 SVD++ là gì? Thêm implicit signal từ all items của user
  1.7 Bảng so sánh BPR vs ALS (GIẢI THÍCH SAI LẦM PHỔ BIẾN)
  1.8 Đánh giá: Rating metrics (RMSE, MAE) vs Ranking metrics (NDCG, Precision@K)
  1.9 Các lưu ý quan trọng khi so sánh BPR và ALS

## 2. Chuẩn Bị Môi Trường & Dữ Liệu
  2.1 Download MovieLens 1M
  2.2 Cài đặt thư viện (implicit, surprise, pandas, v.v.)
  2.3 Khám phá dữ liệu: Ratings, Users, Movies
  2.4 Tách Train/Test (time-based split, 80/20)
  2.5 Chuyển đổi Explicit → Implicit cho BPR
  2.6 Xây dựng User-Item Sparse Matrix

## 3. Triển Khai Mô Hình (Code Chi Tiết)
  3.1 BPR với implicit library
    - Chuẩn bị data
    - Huấn luyện LMS/BPR
    - Sinh top-K recommendations
    - Lưu model
  3.2 SVD với Surprise
    - Load data vào Surprise format
    - Huấn luyện SVD
    - Dự đoán rating
    - Grid search hyperparameters
  3.3 SVD++ với Surprise
    - Khác biệt với SVD (implicit factors)
    - Huấn luyện và timing
    - So sánh training time với SVD

## 4. Chạy Thực Tế & Timing
  4.1 Thiết lập experiment pipeline
  4.2 Training BPR — log thời gian
  4.3 Training SVD — log thời gian
  4.4 Training SVD++ — log thời gian
  4.5 Bảng so sánh runtime
  4.6 Xử lý khi training bị OOM hoặc quá chậm

## 5. Đánh Giá & So Sánh
  5.1 Metrics cho từng model
    - BPR: NDCG@K, Recall@K, AUC (KHÔNG dùng RMSE)
    - SVD/SVD++: RMSE, MAE
  5.2 Fair evaluation setup (same train/test split, same user/item set)
  5.3 Precision@K, Recall@K cho cả 3 model (ranking view)
  5.4 Visualize kết quả (bar charts, line plots)
  5.5 Phân tích: Khi nào dùng model nào?

## 6. Cheat Sheet Cho Bảo Vệ Oral Defense
  6.1 Câu hỏi lý thuyết thường gặp
    - "Tại sao BPR dùng pairwise loss thay vì pointwise?"
    - "SVD++ cải thiện gì so với SVD?"
    - "Tại sao không dùng RMSE để đánh giá BPR?"
    - "BPR và ALS khác nhau chỗ nào?"
  6.2 Mẫu trả lời ngắn gọn (3-5 câu)
  6.3 Công thức toán cần nhớ
  6.4 So sánh nhanh BPR vs SVD vs SVD++ (bảng)
  6.5 Demo chạy nhanh (terminal commands)

## 7. Troubleshooting
  7.1 ImportError: No module named 'implicit'
  7.2 Surprise không đọc được file .dat
  7.3 MemoryError khi xây dựng sparse matrix
  7.4 SVD++ training quá chậm ( > 1 giờ)
  7.5 Kết quả BPR RMSE rất cao → giải thích sai mục đích đánh giá
  7.6 Cold-start items in test set → lọc warm items
  7.7 Inconsistent results giữa các lần chạy → set random seed
```

---

## 6. Pinned Dependency Versions

```text
# File: notes/bpr-svd-svdpp-guide-requirements.txt
pandas==2.0.3
numpy==1.24.3
scipy==1.11.1
scikit-learn==1.3.0
implicit==0.7.2
scikit-surprise==1.1.3
matplotlib==3.7.2
seaborn==0.12.2
```

---

## 7. Key Decisions

| Decision | Rationale |
|---|---|
| MovieLens 1M (not ML-25M) | Smaller, faster training cycles for novices; manageable on any laptop |
| `implicit` for BPR | Industry-standard, optimized Cython code, well-documented |
| `surprise` for SVD/SVD++ | Clean API, built-in cross-validation, educational value |
| Time-based 80/20 split | Respects temporal ordering; realistic for production simulation |
| Separate evaluation tracks | Ranking metrics for BPR; rating metrics for SVD/SVD++; cross-evaluated fairly |
| Oral defense section | Directly addresses student needs for viva preparation |

---

## 8. Out of Scope

- ALS implementation (handled by existing `person1-implementation-guide.md`)
- Neural models (NCF, AutoRec, VAE)
- Content-based hybrid
- Deployment/inference serving
- Multi-core GPU setup (optional notes only)

---

## 9. Unresolved Questions

1. Should the guide include a **dedicated section comparing ALS vs BPR** in more depth, or keep it as a single correction subsection?
2. Should ** Surprise's BPR** (via `surprise.BPR`) be added as a third BPR option alongside `implicit.bpr`, or is one sufficient?
3. Is there an existing **LaTeX template** in the project for the oral defense cheat sheet, or should the Markdown be self-contained?
4. Should the guide include **Jupyter notebook** step-by-step, or pure Python scripts, or both?

---

**Status:** ✅ DONE — Plan delivered. Implementation deferred to next agent/phase.
