# Số liệu chuẩn — đã verify từ `results/final/`

> Trích từ [`../plan.md`](../plan.md) §0.4–0.6. **Mọi số ở đây đã đối chiếu với CSV
> thật** — agent thực hiện không cần tự tính lại, nhưng vẫn nên kiểm chứng khi sửa.
>
> Quy ước: số dùng dấu phẩy thập phân kiểu Việt (`0,1375`).
> Biên độ sweep tính theo công thức `(max−min)/max`.

---

## 1. Biên độ sweep (protocol `dev`, 20.000 user)

### Theo model — mỗi model là một lưới tham số riêng

| Model | n | min → max | Biên độ |
|---|---:|---|---:|
| `cf-als` | 15 | 0,1135 → 0,1383 | **17,94%** |
| `cf-itemknn` | 3 | 0,1263 → 0,1297 | **2,57%** |
| `rs-bpr` | 21 | 0,0364 → 0,1106 | **67,07%** |
| `rs-itemcf` | 12 | 0,1056 → 0,1297 | **18,58%** |
| `cb1-tfidf` | 15 | 0,0036 → 0,0914 | **96,06%** |
| `cb2-bm25` | 21 | 0,0821 → 0,0986 | **16,82%** |
| `cb3-genome` | 19 | 0,0067 → 0,0703 | **90,45%** |
| `cb3-ridge` | 3 | 0,0030 → 0,0125 | **75,98%** |
| `cb4-hybrid` | 32 | 0,0726 → 0,1067 | **32,00%** |

### Theo nhóm — toàn bảng ablation của nhóm

| Nhóm | File | n | min → max | Biên độ |
|---|---|---:|---|---:|
| CF | `ablation-cf.csv` | 51 | 0,0364 → 0,1383 | **73,65%** |
| CB-1 | `ablation-cb1.csv` | 15 | 0,0036 → 0,0914 | **96,06%** |
| CB-2 | `ablation-cb2.csv` | 21 | 0,0821 → 0,0986 | **16,82%** |
| CB-3 | `ablation-cb3.csv` | 22 | 0,0030 → 0,0703 | **95,72%** |
| CB-4 | `ablation-cb4.csv` | 35 | 0,0726 → 0,1067 | **32,00%** |

⚠️ `ablation-cb4.csv` chứa **3 dòng `model=cb1-tfidf`** (`cb4-03-timedecay-hl*`) —
họ time-decay đặt trên CB-1, **không phải** hybrid. Biên độ nhóm CB-4 (32,00%) vẫn
đúng vì 32/35 dòng là `cb4-hybrid`; nhưng khi tách theo model phải nói rõ.

⚠️ **Hai quy ước biên độ đang cùng tồn tại trong tài liệu:**
- `(max−min)/max` → ALS = 17,94%
- `(max−min)/trung bình` → ALS = 18,65% (con số test `test_docs_state_full_als_spread`
  đang đòi, và `README.md` + `docs/project-changelog.md` đang ghi)

→ **Phải chọn một và sửa cả hai nơi cùng lúc.** Không để hai tài liệu nói khác nhau.

---

## 2. Đối chiếu `dev` vs `full`

| Chỉ số | `dev` (20k) | `full` (158.774) |
|---|---|---|
| Tương tác train/user (median) | 58 | 58 |
| Item relevant trong test/user (median) | 8 | 8 |
| Coverage thời gian (1996–2019) | 11,5% – 13,7% mỗi năm | — |

`dev` ⊂ `full` (100% user dev có trong full), cùng `train.parquet` / `test.parquet`.

| Mô hình | `dev` | `full` | Lệch |
|---|---:|---:|---:|
| ALS | 0,1383 | 0,1375 | −0,6% |
| Item-CF | 0,1297 | 0,1290 | −0,5% |
| BPR | 0,1106 | 0,1099 | −0,6% |
| MostPop | 0,0723 | 0,0739 | +2,2% |
| CB-4 | 0,1043 | 0,1054 | +1,1% |
| CB-2 | 0,0986 | 0,0998 | +1,1% |
| CB-1 | 0,0921 | 0,0937 | +1,7% |
| CB-3 | 0,0703 | 0,0713 | +1,3% |

→ Thứ hạng **không đảo**. Dùng biên độ sweep (`dev`) trong báo cáo là hợp lệ,
**nhưng phải ghi rõ mức user của từng bảng**.

---

## 3. Bảng CF — `benchmark-cf.csv` (protocol `full`, 158.774 user)

| Thuật toán | `exp_name` | NDCG@10 | HR@10 |
|---|---|---:|---:|
| ALS | `cf-als-f064-a005-reg0p1` | 0,137470 | 0,536599 |
| Item-CF (RS) | `rs-itemcf-k200-binary-pos` | 0,129014 | 0,487044 |
| BPR (RS) | `rs-bpr-f64-i10-reg0p00001-lr0p01` | 0,109922 | 0,459307 |
| MostPop | `baseline-mostpop` | 0,073903 | 0,324606 |
| Random | `baseline-random` | 0,001171 | 0,010329 |

## 4. Bảng CB — `benchmark-cb.csv` (protocol `full`)

| Thuật toán | `exp_name` | NDCG@10 | HR@10 |
|---|---|---:|---:|
| CB-4 Hybrid | `cb4-04-late-bm25-w075` | 0,105435 | 0,432949 |
| CB-2 BM25 | `cb2-01-k1-05-b000` | 0,099761 | 0,420529 |
| CB-1 TF-IDF | `cb4-03-timedecay-hl0365` | 0,093718 | 0,402037 |
| CB-3 Tag Genome | `cb3-04d-item-knn-tau020-k100` | 0,071292 | 0,338084 |

**Chênh lệch ALS − CB-4:**
- mẫu số ALS: (0,137470 − 0,105435) / 0,137470 = **23,3%** ← khung đang dùng
- mẫu số CB-4: (0,137470 − 0,105435) / 0,105435 = **30,4%**

## 5. Bảng cold-start — `benchmark-coldstart.csv` (10.992 user)

| Thuật toán | NDCG@10 | HR@10 |
|---|---:|---:|
| CB-4 Hybrid | 0,034721 | 0,110808 |
| CB-1 TF-IDF | 0,032795 | 0,101255 |
| CB-2 BM25 | 0,030579 | 0,098799 |
| CB-3 Tag Genome | 0,030501 | 0,096434 |
| Random (sanity) | 0,000362 | 0,001729 |
| ALS / Item-CF / BPR / MostPop | **0,000000** | **0,000000** |

## 6. Bootstrap CI — `bootstrap-ci.csv` (protocol `full`)

| Mô hình | NDCG@10 | CI 95% |
|---|---:|---|
| ALS | 0,137470 | [0,136620 ; 0,138343] |
| Item-CF | 0,129014 | [0,128124 ; 0,129905] |
| BPR | 0,109922 | [0,109152 ; 0,110716] |
| CB-4 (pop_prior 0,4) | 0,108071 | [0,107297 ; 0,108856] |
| CB-4 (content thuần) | 0,105435 | [0,104652 ; 0,106205] |
| CB-2 | 0,099761 | [0,099028 ; 0,100473] |
| CB-1 | 0,093718 | [0,092953 ; 0,094403] |
| MostPop | 0,073903 | [0,073233 ; 0,074637] |
| CB-3 | 0,071292 | [0,070662 ; 0,071955] |
| CB-3 ridge | 0,012852 | [0,012605 ; 0,013104] |
| Chỉ genres | 0,004039 | [0,003915 ; 0,004180] |
| Random | 0,001171 | [0,001105 ; 0,001239] |

## 7. Wilcoxon cold-start — `wilcoxon-coldstart.csv` (mốc = CB-4 Hybrid)

| So với | Chênh lệch TB | p | Có ý nghĩa |
|---|---:|---:|---|
| ALS | 0,034721 | 8,77e-201 | ✅ |
| Item-CF (RS) | 0,034721 | 8,77e-201 | ✅ |
| BPR (RS) | 0,034721 | 8,77e-201 | ✅ |
| MostPop | 0,034721 | 8,77e-201 | ✅ |
| Random | 0,034359 | 1,07e-195 | ✅ |
| CB-3 Tag Genome | 0,004220 | 1,04e-06 | ✅ |
| CB-2 BM25 | 0,004141 | 7,07e-12 | ✅ |
| CB-1 TF-IDF | 0,001926 | 3,92e-03 | ✅ |

## 8. Chống rò rỉ — `tag-leakage-strata.csv` (protocol `full`, 158.774 user)

| Mô hình | < 2006 | 2006–2012 | ≥ 2012 |
|---|---:|---:|---:|
| CB-2 BM25 | **0,082373** | **0,126327** | **0,109397** |
| CB-1 TF-IDF (tag) | 0,070698 | 0,125643 | 0,108262 |
| CB-3 Tag Genome | 0,051559 | 0,083608 | 0,092036 |
| ALS | 0,128278 | 0,150673 | 0,143026 |
| Item-CF (RS) | 0,129919 | 0,137910 | 0,122860 |
| BPR (RS) | 0,103763 | 0,118621 | 0,113727 |
| MostPop | 0,064021 | 0,104622 | 0,070788 |
| Chỉ genres (không rò rỉ) | 0,002575 | 0,004117 | 0,006038 |

Số user mỗi tầng: **75.194 / 29.657 / 53.923**.

## 9. `tag-leakage-summary.csv`

| Chỉ số | Giá trị |
|---|---:|
| Ngày gán tag sớm nhất (toàn bộ `tags.csv`) | 2005-12-24 |
| Median ngày kết thúc huấn luyện (user tập phát triển) | 2006-09-25 |
| Pearson(số tag của phim, độ phổ biến) | **0,768** |
| Spearman(số tag của phim, độ phổ biến) | **0,700** |
| Số phim không có tag | 183 |

## 10. Chế độ trọng số hồ sơ — `weight-mode-paired-tests.csv` (mốc `binary_pos`)

| Chế độ | Nhóm | n | Chênh lệch | p |
|---|---|---:|---:|---:|
| `centered` | 5–30 | 5.278 | −0,022823 | 6,00e-45 |
| `centered` | 31–100 | 8.291 | −0,019513 | 1,93e-63 |
| `centered` | >100 | 6.431 | −0,030924 | 5,12e-76 |
| `rating` | 5–30 | 5.278 | **+0,002570** | 4,04e-02 |
| `rating` | 31–100 | 8.291 | −0,002646 | 3,53e-06 |
| `rating` | >100 | 6.431 | −0,003289 | 1,04e-06 |

`weight-mode-buckets.csv` (NDCG@10):

| `exp_name` | 5–30 | 31–100 | >100 |
|---|---:|---:|---:|
| `cb1-01-add-tags-tfidf` | 0,071314 | 0,073632 | 0,130906 |
| `cb1-04-centered-weights` | 0,048491 | 0,054119 | 0,099982 |
| `cb1-04b-rating-weights` | 0,073884 | 0,070986 | 0,127617 |

## 11. Metric ngoài độ chính xác — `beyond-accuracy-full.csv` (158.654 user, 120 bị loại)

| Mô hình | coverage@10 | gini@10 | novelty@10 | ILD@10 |
|---|---:|---:|---:|---:|
| ALS | 0,149392 | 0,973176 | 3,614974 | 0,733300 |
| Item-CF (RS) | 0,299218 | 0,983192 | 3,150403 | 0,663985 |
| BPR (RS) | 0,197452 | 0,955893 | 4,315226 | 0,709279 |
| CB-1 TF-IDF | 0,469456 | 0,983268 | 3,528960 | 0,628399 |
| CB-2 BM25 | 0,494789 | 0,985549 | 3,225206 | 0,631363 |
| CB-3 Tag Genome | 0,672047 | 0,940684 | 5,525165 | 0,520810 |
| CB-4 (content thuần) | 0,488854 | 0,980172 | 3,506354 | 0,584217 |
| CB-4 (pop_prior 0,4) | 0,330631 | 0,988449 | 3,001832 | 0,583842 |
| MostPop | 0,020556 | 0,998504 | 1,954977 | 0,578365 |

## 12. `beyond-accuracy-coldstart.csv` — `frac_unseen@10`

| Mô hình | frac_unseen@10 |
|---|---:|
| CB-4 (content thuần) | **0,222333** |
| CB-1 TF-IDF | 0,214006 |
| CB-3 Tag Genome | 0,211558 |
| CB-2 BM25 | 0,207290 |
| Random | 0,102184 |
| CB-4 (pop_prior 0,4) | **0,003613** |
| ALS / Item-CF / BPR / MostPop | 0,000000 |

## 13. Nhóm user — `coldstart-user-buckets.csv` (protocol `full`, NDCG@10)

| Mô hình | 5–30 | 31–100 | >100 |
|---|---:|---:|---:|
| ALS | 0,124104 | 0,113888 | 0,178642 |
| Item-CF (RS) | 0,118224 | 0,107086 | 0,165979 |
| BPR (RS) | 0,094537 | 0,093048 | 0,144075 |
| CB-2 BM25 | 0,079305 | 0,079938 | 0,141802 |
| CB-4 (content thuần) | 0,077653 | 0,083022 | 0,156717 |
| CB-1 TF-IDF | 0,071410 | 0,075783 | 0,134818 |
| CB-3 Tag Genome | 0,048167 | 0,056316 | 0,109235 |
| MostPop | 0,061178 | 0,056561 | 0,106513 |
| Chỉ genres | 0,002659 | 0,003362 | 0,006025 |

Số user: **41.343 / 66.131 / 51.300**.

## 14. Bẫy `pop_prior` ở cold-start

`results/coldstart_cb4-06-late-bm25-pop-lam0040_metrics.csv`:
NDCG@10 = **0,000952**, HR@10 = 0,002456.
So với CB-4 content thuần: 0,034721 / 0,110808 → sụp ~97%.

## 15. Dataset — `dataset-summary.csv`

| Chỉ số | Giá trị |
|---|---:|
| Số user | 162.540 |
| Số item | 13.816 |
| Số rating | 24.674.113 |
| Mật độ ma trận | 1,0988% |
| Rating ≥ 4 | 50,1% |
| Rating trung bình | 3,54 |
| Rating/user (trung vị / p90) | 71 / 351 |
| Rating thuộc 20% phim phổ biến nhất | 84,4% |
| Item có ≥ 1 tag | 13.633 |
| Khoảng thời gian | 1995–2019 |
| Số thể loại | 19 |
| Số tag genome | 1.128 |
| Relevance genome trung bình | 0,116 |
| Cosine TB giữa hai phim (genome thô) | 0,61 |
| Lượt gán tag (sau làm sạch) | 937.428 |
| Từ tag khác nhau (sau làm sạch) | 30.067 |
| Tag/phim (trung vị) | 21 |
| Rating train | 19.801.784 |
| Rating test (user hợp lệ) | 4.831.397 |
| Rating test liên quan (≥ 4) | 2.288.027 |
| User hợp lệ để đánh giá | 158.774 |
| Phim liên quan mỗi user trong test (trung vị) | 8 |
| User tập phát triển | 20.000 |

---

## 16. `exp_name` trùng trong `configs/` (12 nhóm, 13 file dư)

| `exp_name` | File |
|---|---|
| `baseline-mostpop` | `baseline-mostpop.yaml`, `best-mostpop.yaml` |
| `baseline-random` | `baseline-random.yaml`, `best-random.yaml` |
| `cb2-01-k1-05-b000` | `best-cb2-bm25.yaml`, `cb2-01-k1-05-b000.yaml` |
| `cb3-04d-item-knn-tau020-k100` | `best-cb3-genome.yaml`, `cb3-04d-item-knn-tau020-k100.yaml` |
| `cb3-05-ridge-lambda100` | `best-cb3-ridge.yaml`, `cb3-05-ridge-lambda100.yaml` |
| `cb4-03-timedecay-hl0365` | `best-cb1-tfidf.yaml`, `cb4-03-timedecay-hl0365.yaml` |
| `cb4-04-late-bm25-w075` | `best-cb4-hybrid-content.yaml`, `cb4-04-late-bm25-w075.yaml` |
| `cb4-06-late-bm25-pop-lam0040` | `best-cb4-hybrid.yaml`, `cb4-06-late-bm25-pop-lam0040.yaml` |
| `cf-als-f064-a005-reg0p1` | `best-cf-als.yaml`, `cf-als-f064-a005-reg0p1.yaml` |
| `cf-itemknn-k200` | `best-cf-itemknn.yaml`, `cf-itemknn-k200.yaml` |
| `rs-bpr-f64-i10-reg0p00001-lr0p01` | `best-rs-bpr.yaml`, `rs-bpr-f64-i10-reg0p00001-lr0p01.yaml` |
| `rs-itemcf-k200-binary-pos` | `best-rs-itemcf.yaml`, `best-rs-itemcf-binary-pos.yaml`, `rs-itemcf-k200-binary-pos.yaml` |

→ **Khoá ghép đúng là cặp (`exp_name`, `config`)**, không phải `exp_name` đơn thuần.

## 17. So sánh hai file `dev-summary`

| | `results/dev-summary.csv` | `results/final/dev-summary.csv` |
|---|---|---|
| số dòng | 11 | 9 |
| `split_id` / `catalog_id` / `seed` | **không có** | **có** |
| `cb4-04-late-bm25-w075` | có | **không** |

→ Không file nào dùng đơn lẻ được. Nguồn thay thế đã kiểm đủ 12 `exp_name`:
`results/{exp_name}_metrics.csv`.
