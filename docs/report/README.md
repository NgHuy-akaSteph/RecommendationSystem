# Báo cáo đồ án — Recommendation System (MovieLens 25M)

> **Trạng thái:** Đủ 5 chương + abstract + phụ lục (Markdown). Mục lục bên dưới là
> điểm vào.
>
> **Nguồn sự thật duy nhất:** mọi số liệu trong báo cáo lấy từ `results/final/*.csv`
> và `results/figures/*.png` đã commit trong repo. Không chép tay số — script
> `scripts/check_report_numbers.py` tự đối chiếu và báo lệch.

---

## 1. Mục lục

| Chương | File | Trạng thái | Nguồn dữ liệu chính |
|---|---|---|---|
| Abstract | [`00-abstract.md`](00-abstract.md) | ✅ Đã viết | tổng hợp toàn báo cáo |
| **1. Giới thiệu** | [`01-introduction.md`](01-introduction.md) | ✅ Đã viết | — |
| **2. Tổng quan và cơ sở lý thuyết** | [`02-literature-review.md`](02-literature-review.md) | ✅ Đã viết | — |
| **3. Dữ liệu, tiền xử lý và phương pháp** | [`03-dataset-and-methodology.md`](03-dataset-and-methodology.md) | ✅ Đã viết | `dataset-summary.csv` |
| **4. Kết quả thực nghiệm và phân tích** | [`04-results-and-analysis.md`](04-results-and-analysis.md) | ✅ Đã viết | `benchmark-*.csv`, `ablation-*.csv`, `bootstrap-ci.csv`, `wilcoxon-*.csv`, `tag-leakage-*.csv` |
| **5. Thảo luận và kết luận** | [`05-discussion-and-conclusion.md`](05-discussion-and-conclusion.md) | ✅ Đã viết | tổng hợp toàn báo cáo |
| Phụ lục | [`09-appendix.md`](09-appendix.md) | ✅ Đã viết (sinh tự động) | `ablation-*.csv` (144 dòng) |

> Phụ lục được **sinh tự động** bằng `venv/bin/python scripts/build_report_appendix.py`
> từ `results/final/*.csv` — không chép tay số.

### Cấu trúc chi tiết

| Chương | Mục |
|---|---|
| **1** | §1.1 Bối cảnh · §1.2 Câu hỏi nghiên cứu · §1.3 Đóng góp · §1.4 Phạm vi và giới hạn · §1.5 Cấu trúc báo cáo |
| **2** | §2.1 Tổng quan · §2.2 Collaborative Filtering · §2.3 Content-based Filtering · §2.4 Metric đánh giá · §2.5 Tóm tắt |
| **3** | §3.1–§3.4 Dữ liệu & tiền xử lý · §3.5–§3.10 Phương pháp & giao thức đánh giá |
| **4** | §4.1–§4.5 Kết quả chính · §4.6–§4.11 Phân tích bổ sung & trình diễn |
| **5** | §5.1–§5.6 Thảo luận · §5.7–§5.9 Đóng góp, hướng phát triển, lời kết |

## 2. Quy ước bất di bất dịch

1. **Không chép tay số** — mọi con số lấy từ `results/final/*.csv`; bảng trích phải
   kèm `split_id` + `catalog_id`.
2. **Thứ hạng CB: `CB-2 ≥ CB-1 > CB-3`** — tuyệt đối không viết "CB-1 > CB-2".
3. **Mọi biên độ sweep** ghi kèm protocol (`dev`, 20.000 user), công thức
   `(max−min)/max`, và nói rõ là **theo model** hay **theo nhóm**.
4. **Số kiểu Việt:** `0,1375` (dấu phẩy thập phân), `158.774` (dấu chấm phân cách
   nghìn). Cả hai dạng đều được `check_report_numbers.py` kiểm tra.
5. Báo cáo đặt ở `docs/report/` — **không** ở `notes/` (đang gitignore).

## 3. Nguồn sự thật — `results/final/`

| Nhóm | File | Dùng cho mục |
|---|---|---|
| Bảng chính | `benchmark-cf.csv`, `benchmark-cb.csv` | §4.1 |
| Bảng `dev` | `benchmark-cf-dev.csv`, `benchmark-cb-dev.csv` | §4.2 |
| Sweep | `ablation-cf.csv` (51), `ablation-cb1.csv` (15), `ablation-cb2.csv` (21), `ablation-cb3.csv` (22), `ablation-cb4.csv` (35) | §4.2 |
| Cold-start | `benchmark-coldstart.csv`, `coldstart-item.csv`, `beyond-accuracy-coldstart.csv` | §4.3 |
| Thống kê | `bootstrap-ci.csv`, `wilcoxon-cf.csv`, `wilcoxon-cb.csv`, `wilcoxon-coldstart.csv`, `wilcoxon-cross-cf-cb.csv` (+ bản `-dev`) | §4.4 |
| Nhóm user | `coldstart-user-buckets.csv` | §4.5 |
| Chống rò rỉ | `tag-leakage-strata.csv`, `tag-leakage-summary.csv` | §4.6–§4.7 |
| Trọng số hồ sơ | `weight-mode-buckets.csv`, `weight-mode-paired-tests.csv` | §4.8 |
| Metric ngoài độ chính xác | `beyond-accuracy-full.csv`, `beyond-accuracy-coldstart.csv` | §4.9 |
| Giao thức | `sampled-vs-full.csv` | §3.9.3, Phụ lục E |
| Chia tập | `split-comparison.csv` | §3.3.3 |
| Demo | `demo-coldstart-comparison.csv` | §4.11.4 |
| Dataset | `dataset-summary.csv` | Chương 3 |

**Hình:** `results/figures/` — `eda-*.png` (Chương 3), `results-accuracy.png`,
`results-metric-vs-k.png`, `results-sweep-range.png`, `results-user-buckets.png`
(Chương 4), `results-beyond-accuracy.png`, `results-accuracy-vs-novelty.png`,
`demo/tab*.png` (Chương 4).

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

Lựa chọn **full ranking** thay vì sampled LOO có bằng chứng trong repo: Kendall
τ = 0,908 (`p = 4,81e-05`) giữa thứ hạng hai giao thức trên 12 mô hình tốt nhất
(`results/final/sampled-vs-full.csv`, §3.9.3).

## 5. Kiểm chứng

```bash
# Đối chiếu số liệu báo cáo ↔ results/final/ (phải báo 0 lệch)
venv/bin/python scripts/check_report_numbers.py

# Kiểm tra tham chiếu §x.y có trỏ tới mục tồn tại (phải báo 0 hỏng)
venv/bin/python scripts/check_report_crossrefs.py

# Sinh lại phụ lục từ CSV
venv/bin/python scripts/build_report_appendix.py

# Test hồi quy tài liệu + biên độ sweep
venv/bin/python -m pytest tests/ -q
```
