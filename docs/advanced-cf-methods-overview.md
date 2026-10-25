# Các Hướng Collaborative Filtering Nâng Cao: Tổng Quan Lý Thuyết

> **Bổ sung cho:** [`theoretical-foundations-cf-methods.md`](theoretical-foundations-cf-methods.md)
>
> Document này mở rộng tầm nhìn về CF bằng các hướng **chưa được triển khai chi tiết** trong dự án nhưng phổ biến trong cộng đồng nghiên cứu và công nghiệp. Mỗi phương pháp được trình bày theo cấu trúc: **ví dụ đời thường → ý tưởng chính → công thức cốt lõi → đặc điểm mạnh/yếu → một câu tóm tắt**.

---

## Mục lục

1. [Giới thiệu và bối cảnh](#1-giới-thiệu-và-bối-cảnh)
2. [User-based Collaborative Filtering](#2-user-based-collaborative-filtering)
3. [Các biến thể của Matrix Factorization](#3-các-biến-thể-của-matrix-factorization)
4. [Neural Collaborative Filtering (NCF)](#4-neural-collaborative-filtering-ncf)
5. [Graph-based Collaborative Filtering](#5-graph-based-collaborative-filtering)
6. [Bảng so sánh tổng hợp](#6-bảng-so-sánh-tổng-hợp)
7. [Khi nào nên xem xét các phương pháp này?](#7-khi-nào-nên-xem-xét-các-phương-pháp-này)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. Giới thiệu và bối cảnh

### Document này bổ sung gì?

Trong [`theoretical-foundations-cf-methods.md`](theoretical-foundations-cf-methods.md), dự án đã trình bày ba hướng CF cốt lõi đã được triển khai:

- **Item-based CF** (tính similarity giữa các item)
- **ALS** (Matrix Factorization cho implicit feedback)
- **BPR** (tối ưu ranking từ implicit feedback)

Document này mở rộng phạm vi đó với **bốn nhóm phương pháp bổ sung** thường xuất hiện trong bài báo và hệ thống production:


| Nhóm                | Phương pháp                           | Mục đích chính                              |
| -------------------- | ---------------------------------------- | ----------------------------------------------- |
| Neighborhood-based   | **User-based CF**                        | So sánh với Item-CF đã có                  |
| Matrix Factorization | **FunkSVD, SVD++, NMF, Temporal MF**     | Dự đoán rating có điều chỉnh             |
| Neural               | **NCF (Neural Collaborative Filtering)** | Học phi tuyến thay cho tích vô hướng      |
| Graph                | **LightGCN và các biến thể**         | Lan truyền thông tin qua đồ thị user–item |

### Tại sao cần biết thêm?

Mỗi phương pháp có **ưu/nhược điểm riêng**, phù hợp với **loại dữ liệu và bài toán khác nhau**. Khi trình bày báo cáo hoặc bảo vệ đồ án, việc hiểu rõ "ngoài kia còn có gì" giúp:

- Lý giải vì sao nhóm chọn (hoặc không chọn) một hướng nào đó.
- Đề xuất hướng phát triển tiếp theo (future work).
- Trả lời câu hỏi phản biện kiểu *"tại sao không dùng Neural CF?"*.

> **Ghi chú đọc:** Document này **không kèm code triển khai**. Mục tiêu là hiểu *ý tưởng* và *công thức* đủ sâu để đọc paper gốc. Khi cần code, hãy tham khảo [`advanced-cf-bpr-svd-svdpp-ml1m-guide.md`](../notes/advanced-cf-bpr-svd-svdpp-ml1m-guide.md) (đã có SVD, SVD++, BPR chạy thực nghiệm trên ML-1M).

---

## 2. User-based Collaborative Filtering

### Ví dụ đời thường

Bạn đi mua sách ở hiệu sách nhỏ. Người bán — sau khi quan sát vài lần — nhận ra *"Bạn A chọn sách gần giống bạn B"*. Lần sau bạn A tới, anh ấy gợi ý cuốn mà bạn B đã mua mà bạn A chưa mua. Đó chính là User-based CF.

### Ý tưởng chính

- Tính **độ tương đồng giữa các user** dựa trên lịch sử tương tác của họ.
- Khi cần gợi ý cho user \(u\), tìm **top-K user giống \(u\) nhất** (láng giềng).
- Gợi ý những item mà các user láng giềng đã thích mà \(u\) chưa tương tác.

### Công thức cốt lõi

**Bước 1 — Tính similarity giữa hai user \(u\) và \(v\)** (cosine trên vector rating của họ):

$$
\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui} \cdot r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2} \cdot \sqrt{\sum_{i \in I_{uv}} r_{vi}^2}}
$$

Trong đó \(I_{uv}\) là tập item **cả hai đều đã rating** (intersection).

**Bước 2 — Dự đoán rating** của \(u\) cho item \(i\) (trung bình có trọng số theo similarity):

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_K(u, i)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_K(u, i)} |\text{sim}(u, v)|}
$$

Trong đó \(N_K(u, i)\) là K user láng giềng của \(u\) đã rating item \(i\).

**Bước 3 — Ranking Top-N:** Loại item đã tương tác, sắp xếp item chưa tương tác theo \(\hat{r}_{ui}\) giảm dần, lấy top-N.

### Đặc điểm

**Mạnh:**

- Trực quan, dễ giải thích: *"vì bạn giống user X nên gợi ý như user X"*.
- Phù hợp khi **danh mục item rất lớn** so với số user (ít phải cập nhật).

**Yếu:**

- **Ma trận user–user thường rất lớn** và tốn bộ nhớ khi hệ thống có nhiều user.
- User mới chưa có tương tác → **cold-start nghiêm trọng** (vì không tính được similarity).
- Nhạy với **sparsity**: hai user ít khi cùng rating chung một item thì similarity không đáng tin.

### So sánh nhanh với Item-CF


| Tiêu chí                 | User-based                   | Item-based                     |
| -------------------------- | ---------------------------- | ------------------------------ |
| Tính similarity giữa     | Các user                    | Các item                      |
| Phù hợp khi              | Số user nhỏ, số item lớn | Số item nhỏ, số user lớn   |
| Ổn định theo thời gian | Thấp (user mới liên tục) | Cao (item ít thay đổi hơn) |
| Trực quan                 | "Bạn giống ai"             | "Sản phẩm giống cái này"  |

### Một câu tóm tắt

> **User-based CF gợi ý dựa trên "người giống bạn thích gì", phù hợp khi hệ thống có ít user và nhiều item ổn định, nhưng gặp khó với cold-start và sparsity.**

---

## 3. Các biến thể của Matrix Factorization

ALS (đã có trong document gốc) là một dạng MF cho implicit feedback. Phần này trình bày các **biến thể khác** của MF — chủ yếu dành cho **explicit rating** hoặc có **ràng buộc/thông tin phụ**.

### 3.1 FunkSVD / SVD (Explicit Rating)

#### Ví dụ đời thường

Mỗi người dùng có một "gu" ẩn (latent). Mỗi bộ phim cũng có "đặc trưng ẩn". Khi gu người dùng trùng với đặc trưng phim, rating cao. FunkSVD học chính xác điều đó từ dữ liệu rating 1–5.

#### Ý tưởng chính

- Mỗi user \(u\) có vector ẩn \(p_u \in \mathbb{R}^k\).
- Mỗi item \(i\) có vector ẩn \(q_i \in \mathbb{R}^k\).
- Rating dự đoán = tích vô hướng + bias toàn cục + bias user + bias item.

#### Công thức cốt lõi

$$
\hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i
$$

Trong đó:

- \(\mu\): rating trung bình toàn cục.
- \(b_u\): bias của user \(u\) (user này hay chấm cao/thấp hơn trung bình).
- \(b_i\): bias của item \(i\) (item này được chấm cao/thấp hơn trung bình).
- \(p_u^T q_i\): tương tác giữa "gu" user và "đặc trưng" item.

**Hàm mất mát** (regularized):

$$
\mathcal{L} = \sum_{(u,i) \in R} (r_{ui} - \hat{r}_{ui})^2 + \lambda \left(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2 \right)
$$

Học bằng **SGD** (Stochastic Gradient Descent).

> Lưu ý: tên gọi "SVD" trong thư viện Surprise thực chất là FunkSVD (không phải phép SVD đầy đủ trên ma trận sparse).

#### Đặc điểm

**Mạnh:**

- Hiệu quả trên dữ liệu **explicit rating** (1–5 sao).
- Dễ giải thích: bias tách bạch rõ user/item, phần tương tác tách bạch.
- Tốn ít bộ nhớ hơn neighborhood-based.

**Yếu:**

- Không xử lý tốt implicit feedback (chỉ dùng rating quan sát được, bỏ qua "không tương tác").
- Bias tĩnh: không phản ánh thay đổi gu theo thời gian.

### 3.2 SVD++

#### Ví dụ đời thường

FunkSVD chỉ nhìn rating 1–5. Nhưng nếu biết thêm rằng *user này đã xem 100 phim hành động* thì dự đoán sẽ tốt hơn nữa. SVD++ kết hợp cả hai nguồn tín hiệu.

#### Ý tưởng chính

- Giữ nguyên cấu trúc FunkSVD.
- Thêm **vector ẩn \(y_j\) cho mỗi item \(j\) mà user đã từng tương tác** (kể cả khi rating thấp).
- User "implicit profile" = trung bình cộng các \(y_j\).

#### Công thức cốt lõi

$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^T \left(p_u + |N(u)|^{-1/2} \sum_{j \in N(u)} y_j \right)
$$

Trong đó \(N(u)\) là tập item user \(u\) đã từng rating (bất kể giá trị). Hệ số \(|N(u)|^{-1/2}\) chuẩn hóa để user "xem nhiều" không tự động có lợi thế.

#### Đặc điểm

**Mạnh:**

- Tận dụng được **implicit signal** (lịch sử tương tác) mà không cần dữ liệu rating ngoài.
- Thường giảm RMSE so với FunkSVD trên cùng tập dữ liệu explicit.

**Yếu:**

- **Chậm hơn** FunkSVD vì phải cập nhật thêm vector \(y_j\).
- Vẫn là **rating regression** — không phù hợp tối ưu ranking trực tiếp.
- Cold-start nghiêm trọng: user mới không có \(N(u)\).

### 3.3 Non-negative Matrix Factorization (NMF)

#### Ý tưởng chính

Giống MF cổ điển nhưng **ràng buộc \(p_u, q_i \geq 0\)** (mọi phần tử không âm). Ý nghĩa:

- Mỗi latent factor được diễn giải như **thành phần cộng** thay vì vector có thể âm.
- Gần với trực giác "đặc trưng này đóng góp bao nhiêu" hơn là "trừ đi bao nhiêu".

#### Công thức cốt lõi

Tối thiểu hóa:

$$
\| R - PQ^T \|_F^2 \quad \text{với} \quad P \geq 0, Q \geq 0
$$

Trong đó \(P\) là ma trận user-factor, \(Q\) là ma trận item-factor. Tối ưu bằng **multiplicative update rules**:

$$
P \leftarrow P \odot \frac{RQ}{PQQ^T + \epsilon}, \quad Q \leftarrow Q \odot \frac{R^T P}{QP^TP + \epsilon}
$$

((\odot) là element-wise, \(\epsilon\) để tránh chia 0).

#### Đặc điểm

**Mạnh:**

- Tính **diễn giải** (interpretability) cao hơn SVD tiêu chuẩn.
- Phù hợp khi dữ liệu vốn không âm (count, duration, frequency).

**Yếu:**

- Hội tụ **chậm hơn** do ràng buộc.
- Chất lượng thường **không vượt** FunkSVD trên RMSE thuần.

### 3.4 Time-aware / Temporal Matrix Factorization

#### Ví dụ đời thường

User A năm 2020 thích phim hành động, nhưng đến 2024 chuyển sang thích phim tình cảm. MF tĩnh chỉ học *trung bình*, bỏ lỡ xu hướng. Temporal MF mô hình hóa sự thay đổi này.

#### Ý tưởng chính

- Cho phép **bias thay đổi theo thời gian**.
- User vector \(p_u(t)\) có thể được phân rã thành thành phần tĩnh + thành phần biến thiên theo thời gian.

#### Công thức cốt lõi

Một dạng phổ biến (theo Koren 2009):

$$
\hat{r}_{ui}(t) = \mu + b_u(t) + b_i(t) + p_u^T q_i
$$

Trong đó bias theo thời gian thường được mô hình bằng **binning hoặc spline**:

$$
b_u(t) = b_u + \alpha_u \cdot \text{dev}_u(t)
$$

\(\text{dev}_u(t)\) là độ lệch thời gian của user so với một mốc tham chiếu (đã được chuẩn hóa).

#### Đặc điểm

**Mạnh:**

- Phản ánh **xu hướng, concept drift** trong gu người dùng.
- RMSE tốt hơn khi dữ liệu dài hạn (nhiều năm).

**Yếu:**

- Nhiều hyperparameter hơn, dễ overfit nếu dữ liệu thưa theo thời gian.
- Cần **timestamp** chính xác — không phù hợp dataset không có thời gian.

### Một câu tóm tắt cho cả nhóm

> **Các biến thể MF mở rộng theo hai hướng: tận dụng implicit feedback (SVD++), ràng buộc diễn giải (NMF), hoặc mô hình hóa thời gian (Temporal MF) — tất cả đều tối ưu rating regression chứ không trực tiếp tối ưu ranking.**

---

## 4. Neural Collaborative Filtering (NCF)

### Ví dụ đời thường

Tích vô hướng \(p_u^T q_i\) giả định mối quan hệ user–item là *tuyến tính*. Nhưng thực tế, sở thích người dùng phi tuyến hơn nhiều: *"thích phim A nhưng không thích phim B dù cùng thể loại"*. NCF thay tích vô hướng bằng một **mạng neural** để học quan hệ phi tuyến này.

### Ý tưởng chính

Paper gốc của He et al. (2017) đề xuất **kết hợp hai thành phần**:

1. **GMF (Generalized Matrix Factorization):** tích vô hướng có trọng số (giống MF nhưng có thêm lớp chiếu).
2. **MLP (Multi-Layer Perceptron):** nối embedding user và item, đưa qua mạng fully-connected để học tương tác phi tuyến.

Hai nhánh hợp lại ở lớp cuối để dự đoán.

### Công thức cốt lõi

**Bước 1 — Embedding:**

$$
p_u = E^U [u], \quad q_i = E^I [i]
$$

Trong đó \(E^U, E^I\) là bảng embedding user và item.

**Bước 2 — GMF:**

$$
\phi^{GMF} = p_u \odot q_i
$$

(\(\odot\) là element-wise product — tương tự tích vô hướng nhưng giữ vector thay vì cộng).

**Bước 3 — MLP:**

$$
\phi^{MLP} = \text{MLP}([p_u, q_i])
$$

\([p_u, q_i]\) là phép nối (concatenation) hai vector.

**Bước 4 — Kết hợp và dự đoán:**

$$
\hat{y}_{ui} = \sigma\left( h^T \begin{bmatrix} \phi^{GMF} \\ \phi^{MLP} \end{bmatrix} \right)
$$

\(\sigma\) là sigmoid (cho bài toán implicit: xác suất user tương tác).

### Đặc điểm

**Mạnh:**

- Mô hình hóa được **tương tác phi tuyến** phức tạp giữa user và item.
- Linh hoạt: có thể thêm **side information** (metadata, text, image) dễ dàng bằng cách nối thêm vector vào MLP.

**Yếu:**

- Cần **nhiều dữ liệu hơn** MF cổ điển để tỏ ra hiệu quả.
- **Khó giải thích** hơn (black-box nhiều hơn).
- Tốn tài nguyên train hơn, cần GPU cho dataset lớn.
- Một số nghiên cứu sau này (ví dụ: He et al. 2020) cho rằng kết quả cải thiện chủ yếu đến từ **regularization và training scheme**, không phải bản thân MLP.

### Một câu tóm tắt

> **NCF thay tích vô hướng bằng mạng neural để học tương tác user–item phi tuyến, mạnh khi có đủ dữ liệu và cần tích hợp nhiều nguồn thông tin, nhưng khó giải thích và đắt hơn MF.**

---

## 5. Graph-based Collaborative Filtering

### Ví dụ đời thường

Hãy tưởng tượng user và item là **các đỉnh** trong một mạng lưới. Mỗi lượt tương tác là một **cạnh**. Khi muốn gợi ý cho user A, ta đi qua các đỉnh lân cận: A → (item A1) ← user B → (item B2) → user C → ... Mỗi "bước đi" mang theo thông tin về sở thích lan từ user này sang user khác. Đó là ý tưởng cốt lõi của Graph CF.

### Ý tưởng chính

- Xây dựng **đồ thị hai phía** (bipartite graph): một tập đỉnh user, một tập đỉnh item, cạnh là tương tác.
- **Lan truyền embedding** (message passing): mỗi đỉnh cập nhật vector của mình dựa trên các láng giềng.
- Sau K bước lan truyền, embedding đã chứa thông tin từ láng giềng K-hop.

### Công thức cốt lõi (LightGCN)

LightGCN (He et al. 2020) đơn giản hóa GCN cho đồ thị CF. Công thức cho embedding ở layer \(k+1\):

$$
e_u^{(k+1)} = \sum_{i \in N(u)} \frac{1}{\sqrt{|N(u)| \cdot |N(i)|}} e_i^{(k)}
$$

$$
e_i^{(k+1)} = \sum_{u \in N(i)} \frac{1}{\sqrt{|N(u)| \cdot |N(i)|}} e_u^{(k)}
$$

Trong đó \(N(u)\) là tập item user \(u\) từng tương tác, \(N(i)\) là tập user đã tương tác với item \(i\). Hệ số chuẩn hóa giống GCN chuẩn.

**Embedding cuối cùng** = trung bình cộng qua các layer:

$$
e_u = \sum_{k=0}^{K} \alpha_k e_u^{(k)}, \quad e_i = \sum_{k=0}^{K} \alpha_k e_i^{(k)}
$$

Thường chọn \(\alpha_k = 1/(K+1)\) (trung bình đều).

**Dự đoán** giống MF:

$$
\hat{y}_{ui} = e_u^T e_i
$$

### Đặc điểm

**Mạnh:**

- Tận dụng **cấu trúc đồ thị** thay vì chỉ vector hóa từng cặp.
- Hiệu quả trên **dữ liệu rất thưa** vì lan truyền qua láng giềng giúp "mượn" tín hiệu.
- LightGCN cực kỳ **gọn**: bỏ feature transformation và activation phi tuyến, vẫn ngang hoặc hơn GCN phức tạp.

**Yếu:**

- Tốn bộ nhớ khi đồ thị lớn (cần lưu ma trận kề chuẩn hóa).
- **Mini-batch sampling** trên đồ thị phức tạp hơn so với dense tensor.
- Vẫn là **tập trung vào tương tác trực tiếp** — thông tin phụ (metadata) cần thiết kế riêng.

### Biến thể đáng chú ý


| Tên paper                     | Ý tưởng khác biệt                                               |
| ------------------------------ | -------------------------------------------------------------------- |
| **GC-MC** (Berg et al. 2017)   | Dùng Graph Convolutional Network đầu tiên cho CF                 |
| **PinSage** (Ying et al. 2018) | Random walk + importance pooling cho đồ thị cực lớn (Pinterest) |
| **LightGCN** (He et al. 2020)  | Đơn giản hóa GCN, loại bỏ transformation phi tuyến            |
| **NGCF** (Wang et al. 2019)    | Thêm affinity embedding cho từng cặp user–item                   |

### Một câu tóm tắt

> **Graph-based CF lan truyền thông tin qua đồ thị user–item để học embedding, hiệu quả trên dữ liệu thưa và phù hợp khi muốn khai thác cấu trúc quan hệ, nhưng đắt hơn MF khi đồ thị rất lớn.**

---

## 6. Bảng so sánh tổng hợp

Bảng dưới tổng hợp **tất cả** phương pháp CF đã trình bày (cả ba phương pháp trong `theoretical-foundations-cf-methods.md` lẫn bốn nhóm trong document này):


| Phương pháp    | Loại dữ liệu         | Mục tiêu tối ưu              | Đầu ra tự nhiên | Độ phức tạp    | Cold-start       |
| ----------------- | ----------------------- | -------------------------------- | ------------------- | ------------------ | ---------------- |
| **Item-CF**       | Explicit / Implicit     | Top-N từ similarity             | Danh sách ranking  | Trung bình        | Yếu (item mới) |
| **ALS**           | Implicit                | Weighted reconstruction          | Score ranking       | Thấp–Trung bình | Yếu             |
| **BPR**           | Implicit                | Pairwise ranking                 | Danh sách ranking  | Thấp              | Yếu             |
| **User-based CF** | Explicit / Implicit     | Top-N từ user similarity        | Danh sách ranking  | Trung bình        | Yếu (user mới) |
| **FunkSVD**       | Explicit                | Rating regression                | Điểm rating 1–5  | Thấp              | Yếu             |
| **SVD++**         | Explicit + history      | Rating regression                | Điểm rating       | Trung bình–Cao   | Yếu             |
| **NMF**           | Explicit (non-negative) | Reconstruction                   | Approx rating       | Trung bình        | Yếu             |
| **Temporal MF**   | Explicit + timestamp    | Rating regression có thời gian | Điểm rating       | Trung bình        | Yếu             |
| **NCF**           | Implicit / Explicit     | Logistic / regression            | Xác suất / rating | Cao                | Yếu             |
| **LightGCN**      | Implicit                | Ranking từ graph                | Danh sách ranking  | Cao                | Yếu             |

### Đọc bảng này như thế nào?

- Cột **Mục tiêu tối ưu** cho biết *mục tiêu train* — đừng so sánh số trực tiếp giữa các mục tiêu khác nhau (ví dụ: RMSE của FunkSVD vs NDCG của BPR không cùng ý nghĩa).
- **Cold-start** đều yếu vì CF thuần không tận dụng side information. Muốn cải thiện cần hybrid với content-based.

---

## 7. Khi nào nên xem xét các phương pháp này?

### Decision tree đơn giản

```
Bạn có dữ liệu rating 1-5 (explicit) không?
├── Có
│   ├── Cần dự đoán rating cụ thể → FunkSVD / SVD++
│   └── Cần ranking Top-N → Biến predicted rating thành rank
│       (lưu ý: không tối ưu ranking, có thể không bằng BPR)
└── Không (chỉ có click/view/purchase implicit)
    ├── Dữ liệu thưa, cần ranking trực tiếp → BPR
    ├── Cần ổn định, dễ song song hóa → ALS
    ├── Có ý định dùng thêm metadata/text → NCF / Hybrid
    └── Có ý định khai thác cấu trúc quan hệ phức tạp → LightGCN
```

### Nguyên tắc chọn phương pháp

1. **Bắt đầu với baseline đơn giản** (popularity, random) rồi Item-CF/ALS — đừng nhảy thẳng vào Neural CF.
2. **Phù hợp dữ liệu là ưu tiên số một**: implicit → ALS/BPR, explicit → FunkSVD/SVD++.
3. **Metric quyết định loss**: ranking metric → BPR/LightGCN, rating metric → FunkSVD.
4. **Tài nguyên tính toán**: GPU sẵn + data lớn → NCF/Graph CF có lợi. CPU-only + data vừa → MF cổ điển ổn hơn.
5. **Cần giải thích** cho stakeholder → MF/SVD++ dễ hơn Neural/Graph.

### Khi nào KHÔNG nên dùng?

- **Dataset quá nhỏ** (< 10K tương tác): MF/Neural sẽ overfit. Neighborhood-based hoặc popularity ổn hơn.
- **Item mới liên tục** (cold-start nặng): CF thuần đều yếu. Cần hybrid với content-based.
- **Cần realtime inference cực nhanh (<10ms)** với hàng triệu user: Graph CF khó đáp ứng. Approximate nearest neighbor (ANN) trên MF embedding sẽ hợp lý hơn.

---

## 8. Tài liệu tham khảo

### Papers gốc

1. **Resnick et al. (1994)** — *GroupLens: An Open Architecture for Collaborative Filtering of Netnews*. Paper gốc về User-based CF.
2. **Sarwar et al. (2001)** — *Item-based Collaborative Filtering Recommendation Algorithms*. Nền tảng cho Item-CF.
3. **Koren, Bell, Volinsky (2009)** — *Matrix Factorization Techniques for Recommender Systems*. Tổng quan MF hiện đại.
4. **Koren (2008)** — *Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model*. Paper SVD++.
5. **Lee, Seung (1999)** — *Learning the Parts of Objects by Non-negative Matrix Factorization*. Gốc NMF.
6. **Koren (2009)** — *Collaborative Filtering with Temporal Dynamics*. Temporal MF.
7. **He et al. (2017)** — *Neural Collaborative Filtering*. Paper NCF gốc.
8. **He et al. (2020)** — *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. LightGCN.
9. **Wang et al. (2019)** — *Neural Graph Collaborative Filtering*. NGCF.
10. **Rendle et al. (2009)** — *BPR: Bayesian Personalized Ranking from Implicit Feedback*. (Đã trình bày trong `theoretical-foundations-cf-methods.md`.)

### Tài liệu bổ sung

- [`theoretical-foundations-cf-methods.md`](theoretical-foundations-cf-methods.md) — Item-CF, ALS, BPR đã được triển khai trong dự án.
- [`advanced-cf-bpr-svd-svdpp-ml1m-guide.md`](../notes/advanced-cf-bpr-svd-svdpp-ml1m-guide.md) — Hướng dẫn chạy SVD/SVD++/BPR trên MovieLens 1M.

### Thư viện tham khảo (nếu muốn triển khai)


| Phương pháp  | Thư viện phổ biến                |
| --------------- | ------------------------------------ |
| FunkSVD / SVD++ | `scikit-surprise`                    |
| ALS / BPR       | `implicit`                           |
| NCF             | `tensorflow-recommenders`, `recbole` |
| LightGCN        | `recbole`, `torch-geometric`         |

---

## Phụ lục: Ký hiệu chung


| Ký hiệu                      | Ý nghĩa                                                |
| ------------------------------ | -------------------------------------------------------- |
| \(u, v\)                       | Chỉ số user                                            |
| \(i, j\)                       | Chỉ số item                                            |
| \(R\)                          | Ma trận rating (user × item)                           |
| \(r_{ui}\)                     | Rating thực tế của user\(u\) cho item \(i\)           |
| \(\hat{r}_{ui}, \hat{y}_{ui}\) | Rating / score dự đoán                                |
| \(p_u, q_i\)                   | Vector ẩn (latent factor) của user\(u\) và item \(i\) |
| \(b_u, b_i\)                   | Bias của user / item                                    |
| \(\mu\)                        | Rating trung bình toàn cục                            |
| \(\text{sim}(u, v)\)           | Độ tương đồng giữa hai user                       |
| \(N(u)\)                       | Tập item user\(u\) đã tương tác                    |
| \(\odot\)                      | Phép nhân element-wise                                 |

---

*Tài liệu này là bản tổng quan lý thuyết, không bao gồm code triển khai. Khi cần code thực nghiệm, tham khảo các file hướng dẫn trong thư mục `notes/` và `docs/`.*
