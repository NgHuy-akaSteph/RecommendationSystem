# Báo cáo đồ án — Recommendation System (MovieLens 25M)

> **Trạng thái:** Đủ 9 chương + phụ lục (Markdown). Mục lục bên dưới là điểm vào.
>
> **Nguồn sự thật duy nhất:** mọi số liệu trong báo cáo lấy từ `results/final/*.csv`
> và `results/figures/*.png` đã commit trong repo. Không chép tay số — script
> `scripts/check_report_numbers.py` tự đối chiếu và báo lệch.

---

## 1. Mục lục

| Chương | File | Trạng thái | Nguồn dữ liệu chính |
|---|---|---|---|
| Abstract | [`00-abstract.md`](00-abstract.md) | ✅ Đã viết | tổng hợp toàn báo cáo |
| 1. Introduction | [`01-introduction.md`](01-introduction.md) | ✅ Đã viết | — |
| 2. Literature Review | [`02-literature-review.md`](02-literature-review.md) | ✅ Đã viết | — |
| 3. Dataset & Preprocessing | [`03-dataset-and-preprocessing.md`](03-dataset-and-preprocessing.md) | ✅ Đã viết | `dataset-summary.csv` |
| 4. Methodology | [`04-methodology.md`](04-methodology.md) | ✅ Đã viết | — |
| **5. Experimental Results** | [`05-experimental-results.md`](05-experimental-results.md) | ✅ Đã viết | `benchmark-*.csv`, `ablation-*.csv`, `bootstrap-ci.csv`, `wilcoxon-*.csv` |
| **6. Phân tích chống rò rỉ & tranh luận** | [`06-leakage-analysis-and-discussion.md`](06-leakage-analysis-and-discussion.md) | ✅ Đã viết | `tag-leakage-*.csv`, `weight-mode-*.csv`, `beyond-accuracy-*.csv` |
| 7. Discussion | [`07-discussion.md`](07-discussion.md) | ✅ Đã viết | tổng hợp §5–6 |
| 8. Conclusion & Future Work | [`08-conclusion.md`](08-conclusion.md) | ✅ Đã viết | tổng hợp toàn báo cáo |
| Phụ lục | [`09-appendix.md`](09-appendix.md) | ✅ Đã viết (sinh tự động) | `ablation-*.csv` (144 dòng) |

> Phụ lục được **sinh tự động** bằng `venv/bin/python scripts/build_report_appendix.py`
> từ `results/final/*.csv` — không chép tay số.

## 2. Quy ước bất di bất dịch

1. **Không chép tay số** — mọi con số lấy từ `results/final/*.csv`; bảng trích phải
   kèm `split_id` + `catalog_id`.
2. **Thứ hạng CB: `CB-2 ≥ CB-1 > CB-3`** — tuyệt đối không viết "CB-1 > CB-2".
3. **Mọi biên độ sweep** ghi kèm protocol (`dev`, 20.000 user), công thức
   `(max−min)/max`, và nói rõ là **theo model** hay **theo nhóm**.
4. **Số kiểu Việt:** `0,1375` (dấu phẩy thập phân).
5. Báo cáo đặt ở `docs/report/` — **không** ở `notes/` (đang gitignore).

## 3. Nguồn sự thật — `results/final/` (27 bảng)

| Nhóm | File | Dùng cho chương |
|---|---|---|
| Bảng chính | `benchmark-cf.csv`, `benchmark-cb.csv` | §5.1 |
| Bảng `dev` | `benchmark-cf-dev.csv`, `benchmark-cb-dev.csv` | §5.2 |
| Sweep | `ablation-cf.csv` (51), `ablation-cb1.csv` (15), `ablation-cb2.csv` (21), `ablation-cb3.csv` (22), `ablation-cb4.csv` (35) | §5.2 |
| Cold-start | `benchmark-coldstart.csv`, `coldstart-item.csv`, `beyond-accuracy-coldstart.csv` | §5.3 |
| Thống kê | `bootstrap-ci.csv`, `wilcoxon-cf.csv`, `wilcoxon-cb.csv`, `wilcoxon-coldstart.csv` (+ bản `-dev`) | §5.4 |
| Nhóm user | `coldstart-user-buckets.csv` | §5.5 |
| Chống rò rỉ | `tag-leakage-strata.csv`, `tag-leakage-summary.csv` | §6.1–6.2 |
| Trọng số hồ sơ | `weight-mode-buckets.csv`, `weight-mode-paired-tests.csv` | §6.3 |
| Metric ngoài độ chính xác | `beyond-accuracy-full.csv`, `beyond-accuracy-coldstart.csv` | §6.4 |
| Dataset | `dataset-summary.csv` | Chương 3 |

**Hình:** `results/figures/` — `eda-*.png` (Chương 3), `results-accuracy.png`,
`results-metric-vs-k.png`, `results-sweep-range.png`, `results-user-buckets.png`
(Chương 5), `results-beyond-accuracy.png`, `results-accuracy-vs-novelty.png`
(Chương 6).

## 4. Ba giao thức đánh giá

| Giao thức | User | `split_id` | Dùng để |
|---|---:|---|---|
| `dev` | 20.000 | `temporal80_20` | **sweep tham số** (144 config) |
| `full` | 158.774 | `temporal80_20` | **bảng chính** |
| `coldstart` | 10.992 | `temporal80_20+coldstart_item` | cold-start item |

`dev` ⊂ `full`, **cùng** `train.parquet`/`test.parquet` — khác biệt duy nhất là ai
được chấm điểm. Median train/user = 58 ở cả hai; NDCG@10 lệch ≤ 1,7% và **thứ hạng
không đảo** → dùng biên độ sweep (`dev`) trong báo cáo là hợp lệ, **nhưng phải ghi
rõ mức user của từng bảng**.

## 5. Kiểm chứng

```bash
# Đối chiếu số liệu báo cáo ↔ results/final/ (phải báo 0 lệch)
venv/bin/python scripts/check_report_numbers.py

# Test hồi quy tài liệu + biên độ sweep
venv/bin/python -m pytest tests/ -q
```
