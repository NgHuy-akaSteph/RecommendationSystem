# Viết báo cáo Chương 5–6 + vá 3 sai lệch đã phát hiện

> **Ngày lập:** 2026-11-15 · **Trạng thái:** sẵn sàng thực hiện
> **Work context:** `/home/huyquang/Workspace/Personal/PyCharmProjects/RecommendationSystem`
> **Branch:** `v1` (ahead `origin/v1` 1 commit — `080561c`)
> **Báo cáo đích:** `docs/report/` (Markdown, mỗi chương 1 file)

## Mục lục

| File | Nội dung |
|---|---|
| [`research/verified-numbers.md`](research/verified-numbers.md) | **Số liệu chuẩn — đã verify từ `results/final/`** (đọc file này trước khi viết) |
| [`phase-01-nen-tang-bao-cao.md`](phase-01-nen-tang-bao-cao.md) | Tạo `docs/report/`, đính chính §5.2, thêm test |
| [`phase-02-hinh-sweep-range.md`](phase-02-hinh-sweep-range.md) | Sinh `results-sweep-range.png` |
| [`phase-03-sua-script-loo.md`](phase-03-sua-script-loo.md) | Sửa lỗi Kendall τ (chưa chạy) |
| [`phase-04-viet-chuong-5.md`](phase-04-viet-chuong-5.md) | Viết Chương 5 |
| [`phase-05-viet-chuong-6.md`](phase-05-viet-chuong-6.md) | Viết Chương 6 |
| [`phase-06-dong-bo-kiem-chung.md`](phase-06-dong-bo-kiem-chung.md) | Đồng bộ docs, kiểm chứng, commit |

## Bối cảnh

`RecommendationSystem` đã hoàn tất pipeline đánh giá — 27 bảng trong `results/final/`,
9 hình trong `results/figures/`, khung báo cáo đã viết lại ỉ `notes/report-structure.md`.
Phần **còn thiếu là prose**: chưa có một dòng báo cáo hoàn chỉnh nào.

### Ba sai lệch đã phát hiện (đều đã verify bằng dữ liệu thật)

| # | Vị trí | Vấn đề |
|---|---|---|
| 1 | `notes/report-structure.md` §5.2 | Quy ước biên độ sweep **trộn hai loại**: "CF 17,94%" là biên độ riêng lưới ALS; biên độ toàn nhóm CF là **73,65%** |
| 2 | `scripts/run_sampled_loo_comparison.py` | Merge theo `exp_name` → **mất dòng `cb4-04`** + **dòng trùng** (2 config `rs-itemcf` trùng `exp_name`) → Kendall τ sai |
| 3 | `results/figures/` | §5.2 yêu cầu hình "thanh ngang so sánh toàn sweep" nhưng **không tồn tại** |

Chi tiết đầy đủ + số liệu chuẩn: [`research/verified-numbers.md`](research/verified-numbers.md).

### Trạng thái chạy thực nghiệm

| Tầng | Protocol | User | Config | Trạng thái |
|---|---|---:|---|---|
| Sweep | `dev` | 20.000 | CF 51 + CB 93 = **144/144** | ✅ đủ, mọi dòng có `split_id`+`catalog_id` |
| Bảng chính | `full` | 158.774 | 13 | ✅ |
| Cold-start | `coldstart` | 10.992 | 12 | ✅ |

`dev` ⊂ `full`, **cùng** `train.parquet`/`test.parquet` — khác biệt duy nhất là ai được
chấm điểm. Median train/user = 58 ở cả hai; NDCG@10 lệch ≤ 1,7% và **thứ hạng không
đảo** → dùng biên độ sweep (`dev`) trong báo cáo là hợp lệ, **nhưng phải ghi rõ mức
user của từng bảng**.

## Quy ước bất di bất dịch

1. **Không chép tay số** — mọi con số lấy từ `results/final/*.csv`; bảng trích phải kèm
   `split_id` + `catalog_id`.
2. **Thứ hạng CB: `CB-2 ≥ CB-1 > CB-3`** — tuyệt đối không viết "CB-1 > CB-2".
3. **Mọi biên độ sweep** ghi kèm protocol (`dev`, 20.000 user), công thức
   `(max−min)/max`, và nói rõ là **theo model** hay **theo nhóm**.
4. **Số kiểu Việt**: `0,1375`.
5. Báo cáo đặt ở `docs/report/` — **không** ở `notes/` (đang gitignore).

## Việc cố ý gác sau (theo yêu cầu người dùng)

| Việc | Lý do |
|---|---|
| Chạy sweep trên protocol `full` (144 config) | Người dùng muốn tự chạy sau khi có báo cáo để học cách chạy |
| Chạy thật `run_sampled_loo_comparison.py` để lấy Kendall τ | Phase 03 chỉ sửa + test |
| Đẩy commit lên `origin/v1` | Chưa được yêu cầu |
| Viết Chương 1–4, 7, 8 | Ngoài phạm vi hiện tại |

## Nghiệm thu

- [ ] `docs/report/README.md` — mục lục, nguồn sự thật, quy ước, mapping chương ↔ bảng
- [ ] `notes/report-structure.md` §5.2 — biên độ tách model / nhóm, kèm protocol
- [ ] Test biên độ sweep chạy động từ `results/final/ablation-*.csv`
- [ ] `results/figures/results-sweep-range.png` tồn tại, 8 hình cũ **không đổi byte**
- [ ] Script LOO: không đọc `dev-summary.csv` trung gian, 12 dòng, 0 trùng
- [ ] `docs/report/05-experimental-results.md` — §5.1–5.5
- [ ] `docs/report/06-leakage-analysis-and-discussion.md` — §6.1–6.5, có mục giới hạn
- [ ] `scripts/check_report_numbers.py` báo **0 lệch**
- [ ] `pytest tests/` xanh toàn bộ
- [ ] Roadmap + changelog đã đồng bộ; commit đã tạo, **chưa push**

## Câu hỏi còn mở

1. **Con số "18,65%" của ALS** — test `test_docs_state_full_als_spread` đòi chuỗi
   `"18,65%"` (mẫu số **trung bình**), còn bảng số liệu dùng 17,94% (mẫu số **max**).
   Hai quy ước khác nhau → phải chọn một và sửa `README.md` +
   `docs/project-changelog.md` + test **cùng lúc**.
2. **`ablation-cb4.csv` chứa 3 dòng `model=cb1-tfidf`** (họ `cb4-03-*` time-decay).
   Đề xuất: vẽ theo model, ghi chú trong caption.
3. **Nguồn metric cho script LOO** — đề xuất đọc `results/{exp_name}_metrics.csv`
   (đủ 12 `exp_name`, đã kiểm chứng). Xác nhận hướng này.
