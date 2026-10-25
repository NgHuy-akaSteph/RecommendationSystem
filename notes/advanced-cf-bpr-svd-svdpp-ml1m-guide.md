# Hướng Dẫn CF Nâng Cao: BPR, SVD và SVD++ với MovieLens 1M

## Mục tiêu và phạm vi

Tài liệu này bổ sung ba phương pháp Collaborative Filtering (CF) để học, trình bày và có thể chạy thử trên **MovieLens 1M**:

- **SVD**: Matrix Factorization cho dự đoán rating explicit.
- **SVD++**: SVD mở rộng, dùng thêm tập phim user đã tương tác.
- **BPR** (*Bayesian Personalized Ranking*): Matrix Factorization tối ưu thứ hạng Top-N cho implicit feedback.

Mục tiêu không phải thay thế pipeline ALS dùng MovieLens 25M của dự án. Đây là một thực nghiệm độc lập, nhỏ hơn và có thể tái lập, giúp nhóm trả lời đầy đủ khi được hỏi về các hướng CF nâng cao. Không chép số liệu mẫu vào báo cáo: phải chạy script, lưu kết quả và chỉ báo cáo kết quả thật.

## Chọn phương pháp nào?

| Phương pháp | Dữ liệu học | Mục tiêu tối ưu | Output tự nhiên | Khi phù hợp |
|---|---|---|---|---|
| SVD | Rating 1–5 | Sai số dự đoán rating | Điểm rating ước lượng | Muốn dự đoán user sẽ chấm mấy sao |
| SVD++ | Rating 1–5 + phim đã rate | Sai số rating + implicit neighborhood | Điểm rating ước lượng | Muốn tăng chất lượng SVD, chấp nhận train chậm hơn |
| BPR | Positive interactions | Item dương xếp trên item chưa tương tác | Thứ hạng Top-N | Muốn gợi ý danh sách phim |
| Implicit ALS | Positive interactions + confidence | Weighted least squares | Thứ hạng Top-N | Cần mô hình ổn định, dễ scale sparse data |

**Điểm quan trọng khi bảo vệ:** SVD/SVD++ và BPR không cùng bài toán gốc. SVD/SVD++ tối ưu **rating regression**; BPR tối ưu **ranking**. Vì vậy, hãy so sánh RMSE/MAE giữa SVD và SVD++, còn BPR nên đánh giá bằng HitRate@K/NDCG@K. Nếu đặt cả ba trong một bảng Top-N, phải ghi rõ đó là đánh giá xếp hạng trên cùng candidate set, không phải chứng minh chúng có cùng objective.

## Lý thuyết ngắn gọn để trả lời vấn đáp

### 1. SVD / FunkSVD

Mỗi user \(u\) có vector ẩn \(p_u\), mỗi phim \(i\) có vector ẩn \(q_i\). Rating dự đoán:

\[
\hat r_{ui}=\mu+b_u+b_i+p_u^Tq_i
\]

Trong đó \(\mu\) là rating trung bình toàn cục; \(b_u\), \(b_i\) là bias của user/phim. Model học các vector sao cho \(\hat r_{ui}\) gần rating thực tế, thường bằng SGD và regularization. Tên `SVD` trong thư viện Surprise là cách triển khai Matrix Factorization kiểu FunkSVD; nó không phải gọi trực tiếp phép SVD đầy đủ trên ma trận sparse.

### 2. SVD++

SVD++ thêm tín hiệu từ tập phim user từng tương tác \(N(u)\):

\[
\hat r_{ui}=\mu+b_u+b_i+q_i^T\left(p_u+\frac{1}{\sqrt{|N(u)|}}\sum_{j\in N(u)}y_j\right)
\]

Nói đơn giản: ngoài "gu ẩn" của user, model còn xem các phim user đã rate. Vì phải cập nhật thêm vector \(y_j\), SVD++ thường chậm và tốn bộ nhớ hơn SVD. Nó không tự giải quyết cold-start: user/phim chưa có lịch sử vẫn thiếu tín hiệu.

### 3. BPR

BPR chỉ cần biết user có tương tác tích cực với phim hay không. Với user \(u\), phim đã thích \(i\) và phim chưa tương tác \(j\), BPR muốn:

\[
\hat x_{ui}>\hat x_{uj}
\]

Nó tối ưu xác suất theo cặp \((u,i,j)\), thường lấy mẫu \(j\) làm negative. Vì tối ưu **thứ tự**, BPR hợp với Top-N; score của BPR không phải rating 1–5 và không nên diễn giải là "user sẽ chấm 4.7 sao".

### 4. ALS khác BPR thế nào?

Cả BPR và implicit ALS đều dùng feedback implicit. ALS tối ưu weighted reconstruction trên toàn bộ ma trận (có confidence), còn BPR lấy mẫu cặp positive/negative và tối ưu thứ tự giữa chúng. ALS thường dễ song song hoá và ổn định; BPR phù hợp khi metric chính là ranking. Không có quy luật BPR luôn tốt hơn ALS: phải đánh giá trên cùng split và protocol.

## Điều kiện trước khi chạy

Dùng môi trường riêng để không bị xung đột với `requirements.txt` hiện có (file đó phục vụ pipeline ML-25M). Khuyến nghị Python 3.11 và các phiên bản dưới đây; `scikit-surprise` có C extension nên NumPy 1.x an toàn hơn với nó.

```bash
cd /home/huyquang/Workspace/Personal/PyCharmProjects/RecommendationSystem
python3.11 -m venv .venv-advanced-cf
source .venv-advanced-cf/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 \
  scikit-surprise==1.1.4 implicit==0.7.2
```

Tải MovieLens 1M từ GroupLens, rồi giải nén sao cho có file `data/raw/ml-1m/ratings.dat`:

```bash
mkdir -p data/raw
curl -L https://files.grouplens.org/datasets/movielens/ml-1m.zip -o /tmp/ml-1m.zip
unzip -q /tmp/ml-1m.zip -d data/raw
ls data/raw/ml-1m/ratings.dat
```

MovieLens 1M chứa các dòng `UserID::MovieID::Rating::Timestamp`. Guide này tự đặt tên cột là `user_id`/`movie_id` khi load `ratings.dat`; tên này khác `userId`/`movieId` trong CSV MovieLens 25M của guide chính. Dataset này là **explicit rating**, nên SVD/SVD++ giữ toàn bộ rating 1–5. Với BPR, chỉ rating từ 4 trở lên được xem là positive interaction. Đây là một quy ước thực nghiệm, không khẳng định rating 1–3 là negative interaction chắc chắn.

## Bước 1 — Load dữ liệu và temporal leave-one-out split

Tạo notebook `notebooks/03-advanced-cf-ml1m.ipynb` hoặc chạy từng khối dưới đây. Split giữ lại rating positive cuối cùng của mỗi user làm test; dữ liệu sau thời điểm test bị loại để không leak tương lai vào train. User cần tối thiểu một positive trước đó mới được đánh giá.

```python
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
POSITIVE_THRESHOLD = 4.0
DATA_PATH = Path("data/raw/ml-1m/ratings.dat")

ratings = pd.read_csv(
    DATA_PATH,
    sep="::",
    names=["user_id", "movie_id", "rating", "timestamp"],
    engine="python",
    dtype={"user_id": "int32", "movie_id": "int32", "rating": "float32", "timestamp": "int64"},
)


def temporal_leave_one_positive_out(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each eligible user's final positive rating without future leakage."""
    train_parts: list[pd.DataFrame] = []
    test_rows: list[pd.Series] = []

    for _, user_ratings in frame.sort_values(["user_id", "timestamp"]).groupby("user_id", sort=False):
        user_ratings = user_ratings.reset_index(drop=True)
        positive_positions = np.flatnonzero(user_ratings["rating"].to_numpy() >= threshold)
        if len(positive_positions) < 2:
            continue

        test_position = int(positive_positions[-1])
        history = user_ratings.iloc[:test_position].copy()
        if (history["rating"] >= threshold).sum() == 0:
            continue

        train_parts.append(history)
        test_rows.append(user_ratings.iloc[test_position])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.DataFrame(test_rows).reset_index(drop=True)
    return train, test


train_explicit, test_positive = temporal_leave_one_positive_out(ratings, POSITIVE_THRESHOLD)
train_implicit = train_explicit.loc[train_explicit["rating"] >= POSITIVE_THRESHOLD].copy()

print(f"Ratings gốc: {len(ratings):,}")
print(f"Train explicit: {len(train_explicit):,}")
print(f"Train positive cho BPR: {len(train_implicit):,}")
print(f"Test positive: {len(test_positive):,} users")
```

## Bước 2 — Chuẩn bị sparse matrix cho BPR

BPR chỉ học từ các tương tác positive trong `train_implicit`. Không thêm rating thấp vào ma trận như negative: thuật toán tự lấy mẫu item chưa tương tác làm negative trong lúc train. Test phải chỉ chứa user và phim đã có mặt ở train; các trường hợp còn lại là cold-start, không công bằng để đánh giá CF thuần.

```python
from scipy.sparse import csr_matrix

warm_users = set(train_implicit["user_id"])
warm_items = set(train_implicit["movie_id"])
test_positive = test_positive.loc[
    test_positive["user_id"].isin(warm_users)
    & test_positive["movie_id"].isin(warm_items)
].copy()

user_ids = np.sort(train_implicit["user_id"].unique())
item_ids = np.sort(train_implicit["movie_id"].unique())
user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
item_to_index = {movie_id: index for index, movie_id in enumerate(item_ids)}
index_to_movie = dict(enumerate(item_ids))

user_item = csr_matrix(
    (
        np.ones(len(train_implicit), dtype=np.float32),
        (
            train_implicit["user_id"].map(user_to_index),
            train_implicit["movie_id"].map(item_to_index),
        ),
    ),
    shape=(len(user_ids), len(item_ids)),
)
print(f"BPR matrix: {user_item.shape}; nnz={user_item.nnz:,}")
# `implicit` BPR nhận CSR có hàng là user và cột là item.
```

**Phân biệt hai loại ID:** `movie_id` là ID gốc của MovieLens; `item_to_index` đổi nó thành chỉ số cột `0..n_items-1` của CSR. API `implicit` chỉ làm việc với **chỉ số cột nội bộ** này: `.recommend()` trả `item_index`, và đối số `items=` cũng phải nhận `item_index`. Vì vậy cần map `movie_id → item_index` trước khi đánh giá, rồi map ngược bằng `index_to_movie` khi hiển thị kết quả.

## Bước 3 — Train và recommend bằng BPR

`implicit` dùng Cython/OpenMP, do đó phù hợp với BPR trên sparse matrix. Bắt đầu ít factor/iteration để xác nhận pipeline chạy; chỉ tuning sau khi evaluator hoạt động.

```python
from implicit.bpr import BayesianPersonalizedRanking

bpr = BayesianPersonalizedRanking(
    factors=64,
    learning_rate=0.05,
    regularization=0.01,
    iterations=100,
    verify_negative_samples=True,
    random_state=RANDOM_SEED,
)
bpr.fit(user_item, show_progress=True)


def recommend_bpr(user_id: int, n: int = 10) -> list[tuple[int, float]]:
    """Return unseen MovieLens movie IDs and BPR ranking scores."""
    if user_id not in user_to_index:
        return []  # cold-start fallback belongs to popularity/content-based model

    item_indices, scores = bpr.recommend(
        userid=user_to_index[user_id],
        user_items=user_item[user_to_index[user_id]],
        N=n,
        filter_already_liked_items=True,
    )
    return [(int(index_to_movie[item_index]), float(score))
            for item_index, score in zip(item_indices, scores)]

example_user = int(user_ids[0])
print(recommend_bpr(example_user, n=10))
```

**Đọc kết quả:** score BPR chỉ dùng để xếp hạng *trong cùng model*; không phải xác suất và không phải rating. Nếu train quá chậm, thử `factors=32, iterations=30` để smoke test, sau đó tăng dần. Không giảm `iterations` rồi dùng kết quả đó làm kết luận cuối cùng.

## Bước 4 — Split riêng cho SVD/SVD++ và train explicit-rating models

SVD/SVD++ cần test rating **không chỉ gồm các rating cao**. Vì vậy tạo một protocol temporal riêng: mỗi user giữ lại rating cuối cùng (dù là 1–5) làm test. Đây là thí nghiệm rating-prediction độc lập với BPR, còn BPR vẫn dùng positive leave-one-out ở Bước 1.

```python
from surprise import Dataset, Reader, SVD, SVDpp, accuracy


def temporal_last_rating_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's final rating for explicit-rating evaluation."""
    ordered = frame.sort_values(["user_id", "timestamp"])
    last_rows = ordered.groupby("user_id", sort=False).tail(1).index
    return ordered.drop(index=last_rows).copy(), ordered.loc[last_rows].copy()


svd_train, svd_test = temporal_last_rating_split(ratings)
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(
    svd_train[["user_id", "movie_id", "rating"]], reader
)
svd_trainset = surprise_data.build_full_trainset()
svd_testset = list(svd_test[["user_id", "movie_id", "rating"]].itertuples(index=False, name=None))
```

Chạy SVD trước để xác nhận dữ liệu/môi trường. `n_epochs` là số vòng SGD; `n_factors` là kích thước embedding. Seed cố định giúp tái lập, nhưng thời gian train vẫn phụ thuộc máy.

```python
svd = SVD(
    n_factors=64,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.02,
    random_state=RANDOM_SEED,
)
svd.fit(svd_trainset)
svd_predictions = svd.test(svd_testset)
svd_rmse = accuracy.rmse(svd_predictions, verbose=True)
svd_mae = accuracy.mae(svd_predictions, verbose=True)
print({"model": "SVD", "rmse": svd_rmse, "mae": svd_mae})
```

Sau khi SVD chạy đúng, đổi **chỉ model** sang `SVDpp`. Đừng chạy SVD++ trước: nó thường chậm đáng kể trên ML-1M vì cập nhật thêm implicit-item factors.

```python
svdpp = SVDpp(
    n_factors=32,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=RANDOM_SEED,
)
svdpp.fit(svd_trainset)
svdpp_predictions = svdpp.test(svd_testset)
svdpp_rmse = accuracy.rmse(svdpp_predictions, verbose=True)
svdpp_mae = accuracy.mae(svdpp_predictions, verbose=True)
print({"model": "SVD++", "rmse": svdpp_rmse, "mae": svdpp_mae})
```

**Cách kết luận đúng:** RMSE/MAE thấp hơn nghĩa là dự đoán rating chính xác hơn trong split explicit này. Nó **không tự chứng minh** SVD++ sẽ cho Top-N tốt hơn BPR/ALS.

## Bước 5 — Tạo Top-N recommendation từ SVD hoặc SVD++

SVD/SVD++ trả về estimated rating cho một cặp user–phim. Để biến thành Top-N, chấm các phim user chưa rate trong candidate set rồi sắp xếp giảm dần. Không chấm toàn bộ catalogue khi chỉ cần demo hoặc evaluation mẫu vì sẽ rất chậm.

```python
def recommend_explicit(
    model: SVD | SVDpp,
    user_id: int,
    seen_movie_ids: set[int],
    candidate_movie_ids: np.ndarray,
    n: int = 10,
) -> list[tuple[int, float]]:
    """Rank unseen candidate movies by the model's estimated explicit rating."""
    candidates = [int(movie_id) for movie_id in candidate_movie_ids
                  if int(movie_id) not in seen_movie_ids]
    predictions = [
        (movie_id, float(model.predict(user_id, movie_id).est))
        for movie_id in candidates
    ]
    return sorted(predictions, key=lambda pair: pair[1], reverse=True)[:n]

seen_by_user = (
    svd_train.groupby("user_id")["movie_id"].agg(set).to_dict()
)
example_svd_user = next(iter(seen_by_user))
print(recommend_explicit(
    svdpp,
    example_svd_user,
    seen_by_user[example_svd_user],
    svd_train["movie_id"].unique(),
    n=10,
))
```

Nếu demo cho người xem, nối `movie_id` với `movies.dat` để in title/genres. Tuyệt đối ghi nhãn output SVD/SVD++ là **estimated rating**, còn output BPR là **ranking score**.

## Bước 6 — Đánh giá BPR bằng sampled ranking protocol

Với mỗi user, test gồm một phim positive bị giữ lại và 99 phim chưa tương tác được lấy mẫu làm negative candidate. Sau đó xem test item có ở top-K, tính HitRate@K và NDCG@K. Đây là **sampled ranking**, kết quả phụ thuộc random seed và 99 negatives; luôn ghi protocol trong báo cáo.

```python
def build_ranking_candidates(
    train_positive: pd.DataFrame,
    test_positive: pd.DataFrame,
    all_movie_ids: np.ndarray,
    negatives_per_user: int = 99,
    random_seed: int = RANDOM_SEED,
) -> dict[int, np.ndarray]:
    """Create one positive plus sampled unseen movies per evaluable user."""
    rng = np.random.default_rng(random_seed)
    seen = train_positive.groupby("user_id")["movie_id"].agg(set).to_dict()
    catalog = set(map(int, all_movie_ids))
    candidates: dict[int, np.ndarray] = {}

    for row in test_positive.itertuples(index=False):
        user_id, positive_item = int(row.user_id), int(row.movie_id)
        excluded = seen[user_id] | {positive_item}
        pool = np.fromiter(catalog - excluded, dtype=np.int32)
        sample_size = min(negatives_per_user, len(pool))
        negatives = rng.choice(pool, size=sample_size, replace=False)
        candidates[user_id] = np.concatenate(([positive_item], negatives))
    return candidates


def evaluate_bpr_at_k(
    model: BayesianPersonalizedRanking,
    candidate_map: dict[int, np.ndarray],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate one held-out positive item per user with HitRate and NDCG."""
    hits: list[float] = []
    ndcgs: list[float] = []

    for user_id, candidates in candidate_map.items():
        candidate_indices = np.array(
            [item_to_index[movie_id] for movie_id in candidates], dtype=np.int32
        )
        ranked_indices, _ = model.recommend(
            userid=user_to_index[user_id],
            user_items=user_item[user_to_index[user_id]],
            N=len(candidate_indices),
            filter_already_liked_items=False,
            items=candidate_indices,
        )
        ranked = np.array([index_to_movie[item_index] for item_index in ranked_indices])
        rank = int(np.flatnonzero(ranked == candidates[0])[0])
        if rank < k:
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)

    return {"hit_rate": float(np.mean(hits)), "ndcg": float(np.mean(ndcgs)), "users": len(hits)}

ranking_candidates = build_ranking_candidates(
    train_implicit, test_positive, item_ids, negatives_per_user=99
)
print(evaluate_bpr_at_k(bpr, ranking_candidates, k=10))
```

> Đối số `items=candidate_indices` khiến BPR chỉ score candidate set, nhanh hơn và kiểm soát protocol tốt hơn. Phải đặt `N=len(candidate_indices)` để model trả về toàn bộ candidate; nếu không, phim positive có thể bị cắt khỏi danh sách trước khi tính hạng. Để so sánh công bằng với ALS, dùng **cùng** `train_implicit`, `test_positive`, `ranking_candidates`, `k` và seed; không dùng RMSE để so sánh BPR với ALS.

## Bước 7 — Bảng kết quả và cách viết báo cáo

Sau khi chạy, tự điền kết quả thật; cột thời gian tính từ `time.perf_counter()` ngay trước/sau `.fit()`. Không sao chép các số dưới đây vì chúng chỉ là cấu trúc bảng.

| Bài toán | Model | Split / candidate protocol | Metric chính | Kết quả | Thời gian train |
|---|---|---|---|---:|---:|
| Rating prediction | SVD | temporal last-rating | RMSE, MAE | điền sau khi chạy | điền sau khi chạy |
| Rating prediction | SVD++ | temporal last-rating | RMSE, MAE | điền sau khi chạy | điền sau khi chạy |
| Top-N ranking | BPR | positive leave-one-out + 99 negatives | HitRate@10, NDCG@10 | điền sau khi chạy | điền sau khi chạy |
| Top-N ranking | ALS (nếu chạy) | cùng protocol với BPR | HitRate@10, NDCG@10 | điền sau khi chạy | điền sau khi chạy |

Mẫu nhận xét an toàn: “Trên split explicit temporal, [SVD/SVD++] đạt RMSE thấp hơn. Trên sampled Top-N ranking, [BPR/ALS] có NDCG@10 cao hơn. Hai kết quả này trả lời các mục tiêu khác nhau nên không diễn giải như một bảng xếp hạng tuyệt đối giữa mọi thuật toán.”

## Tuning tối thiểu, runtime và lỗi thường gặp

1. Chạy smoke test trước: BPR `factors=32, iterations=30`; SVD `n_factors=32, n_epochs=10`; SVD++ `n_factors=32, n_epochs=5`.
2. Khi pipeline đúng, chỉ tuning một biến mỗi lượt: factors, regularization và iterations/epochs. Lưu seed, split, version package và thời gian train.
3. SVD++ quá chậm: giảm `n_factors`/`n_epochs` lúc thử, nhưng báo cáo rõ cấu hình cuối. Không dùng random split chỉ để SVD++ có score đẹp hơn.
4. Lỗi build `scikit-surprise`: kiểm tra đang ở `.venv-advanced-cf`, `python --version`, và `numpy==1.26.4`; tạo lại môi trường nếu pip đã lấy NumPy 2.x.
5. BPR trả item đã xem: bắt buộc truyền `user_items=user_item` khi gọi `.recommend(..., filter_already_liked_items=True)`.
6. `KeyError` khi evaluate: test item/user không nằm trong train mapping là cold-start; lọc warm test như Bước 2, không âm thầm gán score 0.
7. Không đủ 99 negative: hàm trên tự lấy số còn lại. Cần ghi số negatives thực tế hoặc bỏ những user không đủ candidate nếu muốn protocol cố định tuyệt đối.

## Câu hỏi bảo vệ và câu trả lời ngắn

**Tại sao chọn ALS để triển khai chính, còn BPR/SVD/SVD++ là mở rộng?** ALS implicit cân bằng khả năng scale sparse data, độ ổn định và triển khai Top-N. BPR phù hợp hơn nếu tối ưu ranking là ưu tiên; SVD/SVD++ phù hợp khi yêu cầu chính là dự đoán rating explicit.

**BPR dùng những phim chưa tương tác làm negative có chắc user không thích không?** Không. BPR xem chúng là negative *tương đối* để học thứ tự; đây là lý do cần sampled negatives và không được nói “user ghét phim đó”.

**SVD++ có phải deep learning không?** Không. Nó là Matrix Factorization với thêm implicit feedback từ tập item user đã tương tác, vẫn là latent-factor CF cổ điển.

**Vì sao SVD++ có thể tốt hơn SVD nhưng vẫn không chọn mặc định?** Nó khai thác thêm history nên có thể giảm RMSE, đổi lại train chậm hơn và vẫn là rating-prediction objective; lợi ích phải được kiểm chứng bằng thực nghiệm của nhóm.

**BPR tốt hơn ALS không?** Không có câu trả lời cố định. BPR tối ưu cặp thứ hạng, ALS tối ưu weighted reconstruction. Chọn theo dữ liệu, metric, runtime và kết quả trên cùng protocol.

**Các model xử lý cold-start thế nào?** CF thuần không giải quyết tốt user/item mới vì chưa có interaction. Dùng popularity fallback, content-based filtering hoặc hybrid với metadata genres/tags.

## Checklist trước khi nộp

- [ ] Dataset ML-1M được tải theo giấy phép/điều khoản GroupLens và không commit raw data lớn.
- [ ] Dùng temporal split, ghi rõ ngưỡng positive = 4.0, số negative candidates, seed và package versions.
- [ ] Báo cáo RMSE/MAE riêng cho SVD/SVD++; HitRate/NDCG riêng cho BPR/ALS.
- [ ] Không ghi score BPR như rating hoặc probability.
- [ ] Lưu `results/advanced-cf-results.json`, log runtime và cấu hình tốt nhất thực tế.
- [ ] Có demo ít nhất một user với title phim, đồng thời lọc phim đã tương tác.

## Tài liệu gốc nên trích dẫn

1. Koren, Y. (2008). *Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model* — nền tảng SVD++.
2. Rendle, S. et al. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*.
3. Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets* — implicit ALS.
4. [Surprise documentation](https://surpriselib.com/) — API `SVD`, `SVDpp`.
5. [implicit documentation](https://benfred.github.io/implicit/) — API BPR/ALS.
6. [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — dataset và điều khoản sử dụng.

## Phạm vi chưa triển khai

Tài liệu này không triển khai neural CF, graph CF hay hybrid production-grade. Đây là các hướng mở rộng phù hợp để nêu trong phần future work, không nên tuyên bố đã được đánh giá chỉ dựa trên tài liệu này.
```