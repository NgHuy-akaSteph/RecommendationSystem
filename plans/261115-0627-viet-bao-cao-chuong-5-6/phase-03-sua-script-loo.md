# Phase 03 — Sửa script so sánh giao thức LOO

**Trạng thái:** ⬜ Chưa bắt đầu
**Ưu tiên:** Trung bình (chuẩn bị bằng chứng Kendall τ cho Chương 7.4 — **chưa chạy**)
**Plan gốc:** [`../plan.md`](plan.md) §4

## Context Links

- Plan: [`../plan.md`](plan.md) §0.5 + §4
- Script: [`scripts/run_sampled_loo_comparison.py`](../../scripts/run_sampled_loo_comparison.py)
- Evaluator: [`src/eval/sampled_loo_evaluator.py`](../../src/eval/sampled_loo_evaluator.py)
- Test: [`tests/test_phase4_protocols.py`](../../tests/test_phase4_protocols.py)

## Overview

Script hiện tính Kendall τ **sai** vì hai lỗi độc lập. Phase này **sửa code + thêm
test**, **không chạy thật** (người dùng gác).

## Key Insights

**Lỗi 1 — đọc sai file.** Dòng 38 đọc `results/dev-summary.csv` (artifact trung gian,
gitignore):

| | `results/dev-summary.csv` | `results/final/dev-summary.csv` |
|---|---|---|
| số dòng | 11 | 9 |
| `split_id`/`catalog_id` | **không có** | **có** |
| `cb4-04-late-bm25-w075` | có | **không** |

→ Không file nào dùng đơn lẻ được.

**Lỗi 2 — trùng `exp_name`.** `best-rs-itemcf.yaml` và
`best-rs-itemcf-binary-pos.yaml` **cùng** `exp_name` → merge sinh dòng trùng →
`rank()` sai → Kendall τ sai. (12 nhóm trùng trong `configs/`, xem plan §0.5.)

**Hướng sửa đề xuất:** đọc metric gốc `results/{exp_name}_metrics.csv` — đã kiểm tra
**đủ 12** `exp_name` của `best-*.yaml` (kể cả `cb4-04-late-bm25-w075`), mỗi file
một dòng → không trùng, không mất, không phụ thuộc bảng trung gian.

## Requirements

**Chức năng**
- Script không còn đọc `results/dev-summary.csv`.
- Dedup `best-*.yaml` theo `exp_name` (giữ file đầu tiên theo `sorted()`), log file bị bỏ.
- Bảng kết quả có cột `config` → khoá ghép là cặp (`exp_name`, `config`).
- Assert: sau merge **không trùng dòng**, **không mất `exp_name`**.

**Phi chức năng**
- Test ở mức hàm, không cần chạy mô hình thật.
- Giữ nguyên ngữ nghĩa Kendall τ (hạng theo NDCG@10 của mỗi giao thức).

## Architecture

```mermaid
flowchart LR
    BestCfg[configs/best-*.yaml] --> Dedup[dedup theo exp_name]
    Dedup --> Run[run_config / sampled LOO]
    Metrics[results/{exp}_metrics.csv] --> Merge[merge theo exp_name + config]
    Run --> Merge
    Merge --> Tau[Kendall tau]
    Merge --> Check{assert}
    Check -->|trùng/mất| Fail[SystemExit]
```

## Related Code Files

**Sửa**
- `scripts/run_sampled_loo_comparison.py`
- `tests/test_phase4_protocols.py` — thêm test

## Implementation Steps

1. Đọc lại `scripts/run_sampled_loo_comparison.py` toàn bộ.
2. Viết hàm `_unique_best_configs()`: duyệt `sorted(paths.CONFIG_DIR.glob("best-*.yaml"))`,
   giữ `exp_name` đầu tiên, log các file bị bỏ (vd `best-rs-itemcf.yaml` trùng
   `best-rs-itemcf-binary-pos.yaml`).
3. Đổi nguồn metric full-ranking: đọc `results/{exp_name}_metrics.csv` cho từng
   config đã dedup → DataFrame `exp_name`, `config`, `NDCG@10`.
4. `compare_protocols()` merge theo `exp_name` (đã dedup nên 1-1) + giữ cột `config`.
5. Thêm assert rõ ràng: `len(merged) == len(unique_exp_names)` và không có
   `exp_name` trùng.
6. Thêm test:
   - `test_loo_comparison_has_no_duplicate_or_missing_rows()` — dựng bảng giả có
     `exp_name` trùng cố ý, khẳng định hàm loại trùng + giữ đủ.
   - `test_loo_script_does_not_read_stale_dev_summary()` — assert script không chứa
     `paths.RESULTS_DIR / "dev-summary.csv"`.
7. Chạy `venv/bin/python -m pytest tests/test_phase4_protocols.py -q`.

## Todo List

- [ ] Dedup `best-*.yaml`
- [ ] Đổi nguồn metric, merge theo khoá
- [ ] Assert chống trùng/mất
- [ ] Test hồi quy
- [ ] `pytest` xanh

## Success Criteria

- Script không còn đọc file dev-summary trung gian.
- Merge 12 config `best-*` (12 `exp_name` khác nhau) → **12** dòng, 0 trùng.
- Test xanh.

## Risk Assessment

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Chạy script thật tốn ~20 phút | — | **Không chạy** trong phase này |
| Đổi nguồn metric làm lệch số | Thấp | Test khẳng định tập `exp_name` khớp 12 |

## Security Considerations

Không có.

## Next Steps

- Sau khi có báo cáo: chạy `venv/bin/python scripts/run_sampled_loo_comparison.py`
  để lấy Kendall τ → điền vào Chương 7.4.
