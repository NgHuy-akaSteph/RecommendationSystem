# Kế hoạch thống nhất RecommendationSystem và recommendation-cb

> **Đã bị thay thế** (2026-11-13) bởi
> [`plans/261113-2106-unify-recsys-and-recommendation-cb/plan.md`](261113-2106-unify-recsys-and-recommendation-cb/plan.md).
>
> **Phase 00–03 đã hoàn tất** (2026-11-14). Bảng so sánh chung đã chạy: 8 mô hình × 3
> giao thức, kết quả ở `results/final/benchmark-{cf,cb,coldstart}.csv`. Phần còn lại
> (dọn code legacy + viết `docs/`) là Phase 04.
>
> Bản này giữ lại làm **bản rà soát phân tích** (vì sao hai bảng kết quả hiện tại không so
> trực tiếp được, và 4 lỗi P0 cụ thể của `RecommendationSystem`). Phần triển khai đã rút
> gọn: 8 mô hình thay vì 14, 3 phase thay vì 4, bỏ phần deploy/demo, hybrid để dành.

## Kết luận rà soát

`RecommendationSystem` là mẫu hợp lý cho **API gợi ý công khai**: nhận `userId`, trả
`movieId`, có `recommend(user_id, n, seen)`. Chữ ký `fit` của từng thuật toán không cần
giống nhau: Item-CF nhận ma trận và map, Popularity nhận DataFrame, ALS nhận implicit
DataFrame. Dùng một runner hoặc adapter để tạo đúng đầu vào cho từng mô hình.

`RecommendationSystem` **chưa phải chuẩn đánh giá cuối cùng**. Split hiện tại là 70/30
theo thời gian toàn cục, test bị lọc về warm user/item, còn `recommendation-cb` dùng
80/20 theo từng user trên catalog 13.816 phim. Hai bảng kết quả hiện tại không so
trực tiếp được. Evaluator của `RecommendationSystem` cũng cần sửa Precision@K, xử lý
model lỗi và kiểm tra user khi làm kiểm định ghép cặp.

**Quyết định:** lấy `RecommendationSystem` làm nơi tích hợp và API công khai; lấy giao
thức full-ranking của `recommendation-cb` làm giao thức nghiên cứu chính, rồi chạy
lại *mọi* baseline từ `RecommendationSystem` trên chính split/catalog đó. Kết quả cũ
chỉ là tham khảo. Nếu chọn giao thức khác, vẫn phải chạy lại toàn bộ mô hình trên một
giao thức duy nhất.

## Hợp đồng chung cần chốt trước khi sửa mô hình

| Thành phần | Hợp đồng |
|---|---|
| Catalog chính | 13.816 `movieId` có Tag Genome; lưu thứ tự `movieId -> item_idx` cố định. Catalog 62.423 phim là thí nghiệm riêng. |
| User | `userId -> user_idx` cố định, xây từ train/test đã chốt; không tự remap trong từng mô hình khi so sánh. |
| Split | Temporal per-user 80/20, seed 42, tie-break rõ ràng; lưu manifest/hash và train/test. |
| Relevance | Rating test >= 4; các mô hình chỉ học từ train. |
| Ứng viên | Cùng catalog, bỏ **mọi** item đã rating trong train, kể cả rating thấp; full ranking. |
| User đánh giá | Cùng danh sách user có >= 1 relevant test và đủ lịch sử train; không âm thầm thay đổi theo mô hình. |
| API ngoài | `recommend(user_id: int, n: int = 10, seen: set[int] | None = None) -> list[int]`, trả `movieId` duy nhất, đúng thứ tự, không chứa `seen`. |
| API trong | Giữ `fit(train, features)` / `score_users(user_idx)` theo lô trong `recommendation-cb` để đánh giá nhanh. |
| Metrics | P/R/NDCG/HR@{5,10,20}, thêm Coverage@10; P@K = hits/K kể cả khi trả ít hơn K. |
| Kết quả | Mỗi hàng có model/config, split ID, catalog ID, seed, số user, candidate policy, metric và thời gian chạy. |

## P0 — sửa điều kiện đánh giá của RecommendationSystem

1. `src/evaluation/recommender_evaluator.py`: đổi Precision@K từ
   `hits / len(top_k)` thành `hits / k`. Thêm test cho trường hợp model chỉ trả 2
   item khi K=10.
2. `src/evaluation/evaluation_pipeline.py`: khi model ném lỗi, lưu lỗi và dừng run
   hoặc đánh dấu run không hợp lệ; không chuyển lỗi thành score 0. Khi không có gợi ý
   nhưng không lỗi, ghi đủ P/R/NDCG/HR bằng 0. Kiểm tra item gợi ý thuộc catalog.
3. `src/evaluation/statistical_analysis.py`: trước paired test, xác nhận thứ tự và
   tập `userId` giống nhau; ghép theo `userId` thay vì chỉ so độ dài mảng.
4. `src/recommenders/item_cf.py`: `fit` dùng đúng ma trận và map của runner; trong
   `recommend`, tự mask các item đã có trong `user_item[user_idx]` và hợp thêm `seen`.
   Tránh `toarray()` cho toàn bộ hàng nếu có thể; kiểm tra không trả item train khi
   `seen=None`.
5. `src/recommenders/als_recommender.py` và BPR: runner phải dùng map catalog chung
   (có cột item cold-start); không dùng riêng map chỉ từ các rating dương. Khi lọc
   đã xem, lọc toàn bộ train history, không chỉ các rating dương. Không nuốt lỗi
   `recommend` thành danh sách rỗng khi chạy benchmark.

**Gate P0:** kiểm thử toy dataset có thể tính metric bằng tay; mọi baseline trả cùng
kiểu `movieId`, không có seen item, và lỗi mô hình làm run thất bại rõ ràng.

## P1 — sửa và bổ sung recommendation-cb để dùng chung

1. Giữ `src/models/*` và batch evaluator hiện có. Thêm lớp adapter, ví dụ
   `src/integration/recommendation_system_adapter.py`, nhận fitted CB model,
   `user_map`, `movie_map`, train CSR. `recommend()` đổi `userId -> user_idx`, gọi
   `score_users(np.array([user_idx]))`, mask train + `seen`, lấy top-N có thứ tự và
   đổi `item_idx -> movieId`. User chưa có lịch sử trả `[]` hoặc fallback Popularity
   theo cấu hình; báo rõ trong báo cáo. `n<=0` trả `[]`.
2. Thêm `get_similar_items(movie_id, n)` và `get_movie_features(movie_id)` nếu muốn
   đáp ứng phần gợi ý tương tự và giải thích trong guide Person 2. Với item không có
   content, trả trạng thái/fallback rõ ràng.
3. Tách cấu hình **content thuần** và **CB + popularity prior**. Không gọi CB-4 có
   popularity prior là content thuần. Đánh giá hai cấu hình riêng ở warm và cold item.
4. Tag document hiện gom toàn bộ `tags.csv`, kể cả tag phát sinh sau thời điểm test.
   Bổ sung thí nghiệm chỉ dùng genres (đối chứng không có tag), và một giao thức
   tag theo cutoff thời gian nếu muốn kết luận triển khai theo thời gian. Nếu giữ
   toàn bộ tag, ghi rõ đây là đánh giá hồi cứu với metadata đầy đủ.
5. Duy trì split cold item riêng: xóa toàn bộ rating train của held-out items,
   giữ nguyên cột item và feature. Gọi đây là **cold-start mô phỏng**. Xuất tỷ lệ
   phim held-out được gợi ý cùng NDCG; không dùng kết quả đó như bằng chứng về
   phim mới thật ngoài tập dữ liệu.
6. Tạo export/import manifest gồm `movieId`, `userId`, item/user index, split ID,
   feature version và config. Adapter phải từ chối map khác thứ tự ma trận hoặc
   model artifact khác catalog.

**Gate P1:** toy test đối chiếu kết quả `adapter.recommend()` với top-N của
`score_users`; thử user lạ, item lạ, n=0, train seen, explicit `seen`, cold item,
score bằng nhau và model chưa fit. Không thay đổi thứ hạng batch evaluator khi thêm
adapter.

## P2 — chạy bảng so sánh chung

1. Tạo runner thống nhất ở `RecommendationSystem` đọc đúng manifest/split của P1.
   Runner fit Random, MostPop, Bayesian/Wilson, Item-CF, ALS, BPR và CB-1/2/3/4
   trên cùng train; chỉ các mô hình dùng nội dung mới nhận item features.
2. Dùng một evaluator và một danh sách user/candidate cho mọi mô hình. Đánh giá
   dev để chọn tham số, khóa cấu hình, rồi chạy full/test đúng một lần. Không chọn
   tham số theo test. Xuất per-user metrics để kiểm định ghép cặp.
3. Xuất bảng warm và bảng cold item riêng. Có P/R/NDCG/HR@5/10/20, Coverage@10,
   tỷ lệ user không nhận đủ K item, số user đánh giá và thời gian fit/score.
4. Đối chiếu evaluator mới với `recommendation-cb` trên toy data và cùng một tập
   top-K đã lưu; sai khác metric phải bằng 0 trước khi chạy 25M ratings.

**Gate P2:** cùng split/catalog/user set; chạy lại bằng seed cố định cho kết quả
trong sai số số học; bảng kết quả có nguồn config và không dùng số cũ từ hai repo.

## P3 — bản hoàn chỉnh để trình bày

1. Viết README một đường chạy: chuẩn bị dữ liệu -> train -> evaluate -> xuất bảng.
   Pin phiên bản Python và package; ghi nguồn dữ liệu MovieLens và không commit raw data.
2. Hoàn thiện báo cáo `recommendation-cb` từ các bảng P2: cách chia dữ liệu, metric,
   ablation, warm/cold, coverage, giới hạn tag leakage và mô phỏng cold-start.
3. Thêm demo tối thiểu: chọn user hoặc phim, hiển thị top-10, tên phim, genres/tags
   giải thích và fallback khi user mới. Demo tải artifact đã lưu, không fit lại 25M.
4. Chạy kiểm tra tái lập từ môi trường sạch; lưu lệnh, thời gian, cấu hình và hash
   manifest cùng kết quả cuối.

## Các điểm không được hiểu nhầm

- Chữ ký `ItemCFRecommender.fit(user_item_csr, user_map, movie_map)` là hợp đồng
  riêng của Item-CF, không phải mẫu `fit` bắt buộc cho mọi recommender.
- `recommendation-cb` hiện có evaluator batch tốt hơn cho full ranking quy mô lớn;
  không cần ép mọi mô hình chạy vòng lặp Python từng user của pipeline cũ.
- Các metric cũ của hai repo không cùng giao thức; không dùng chúng để kết luận
  mô hình nào tốt hơn sau khi tích hợp.
