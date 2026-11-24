# Chương 2 — Tổng quan tài liệu và cơ sở lý thuyết

> **Vai trò:** cung cấp nền tảng lý thuyết cho toàn bộ thí nghiệm ở Chương 3–4.
> Chương này **không** trả lời RQ nào — nó định nghĩa các khái niệm mà RQ1–RQ3 sử
> dụng, và đặc biệt là **metric** (§2.4), vì lựa chọn metric quyết định kết luận.
>
> **Ghi chú về trích dẫn số liệu:** những con số nêu ở §2.3.2 và §2.3.3 là kết quả
> **của chính đồ án này**, lấy từ `results/final/` — không phải số liệu từ tài liệu.

---

## 2.1 Tổng quan về hệ thống gợi ý

Một hệ thống gợi ý nhận đầu vào là hồ sơ người dùng (lịch sử tương tác, hoặc ngữ
cảnh) và đầu ra là một danh sách được xếp hạng các mục (phim, sản phẩm, bài viết).
Theo lịch sử phát triển, các hướng tiếp cận chia thành bốn nhóm:

| Hướng | Ý tưởng cốt lõi | Ví dụ |
|---|---|---|
| **Content-based** | Khớp đặc trưng của mục với hồ sơ người dùng | Người dùng thích phim hành động → gợi ý phim hành động |
| **Collaborative filtering** | Khớp hành vi của người dùng này với người dùng khác | Người giống bạn đã xem phim này |
| **Hybrid** | Kết hợp hai tín hiệu | late fusion, ensemble |
| **Knowledge-based** | Suy luận qua đồ thị tri thức | "người dùng thích phim của đạo diễn X" |

Đồ án này tập trung vào hai hướng đầu, vì chúng là hai hướng có bằng chứng lượng
tính mạnh nhất trên MovieLens và — quan trọng hơn — chúng **thắng ở hai vùng bài
toán khác nhau**, nên so sánh chúng có giá trị thực tiễn.

### 2.1.1 So sánh nhanh

| | **Collaborative Filtering** | **Content-based** |
|---|---|---|
| **Dữ liệu cần** | Tương tác user–item | Đặc trưng nội dung của item |
| **Nguồn tín hiệu** | Hành vi cộng đồng | Đặc trưng nội dung |
| **Ưu** | Khám phá mối liên quan ẩn không ai viết ra; độ chính xác cao khi lịch sử dày | Không cần tương tác; **giải thích được**; phủ rộng; sống được với item mới |
| **Nhược** | Cold-start item = không có tín hiệu; dễ dồn suất; không giải thích | Giới hạn bởi chất lượng đặc trưng; thiếu serendipity; dễ "tự xem lại" |
| **Phụ thuộc** | Mật độ tương tác | Độ bao phủ và chất lượng metadata |
| **Chi phí tính toán** | Cao (ma trận 158.774 × 13.816) | Trung bình–cao (phép nhân thưa) |

Bảng này là khung lý thuyết; các chương sau **đo lại** từng ô bằng số liệu thật.

---

## 2.2 Collaborative Filtering

### 2.2.1 User-based CF

Ý tưởng: tìm những user có lịch sử tương tự user đang xét, rồi gợi ý những phim
họ đã xem mà user đang xét chưa xem. Tương đồng thường đo bằng **cosine** (trên
vector nhị phân hoặc có trọng số) hoặc **Pearson** (hiệu chuẩn hoá theo trung bình
mỗi user, để loại trừ xu hướng thích "phim hay" của cá nhân).

Ưu điểm là trực giác dễ hiểu và giải thích được ("vì người giống bạn đã xem phim
này"). Nhược điểm là **chi phí tính toán và độ thưa**: ma trận tương đồng user×user
(158.774²) là khối lớn, và ở MovieLens 25M phần lớn user chỉ có vài chục tương tác
nên tương đồng giữa hai user thường bằng 0. Đồ án không cài user-based CF vì lý do
này — đây là một lựa chọn có chủ đích, không phải bỏ sót.

### 2.2.2 Item-based CF

Trực giác đảo: thay vì tìm "người giống tôi", tìm "**phim giống phim tôi đã xem**".
Với mỗi cặp phim, tính độ tương đồng giữa các user đã xem cả hai; gợi ý cho user u
là các phim j tương tích với phim i mà u đã tương tác, có trọng số theo mức độ tương
tác của u với i:

\[
S[u, j] = \sum_{i \in \text{seen}(u)} w_{ui} \cdot \mathrm{sim}(i, j)
\]

Item-based scale tốt hơn user-based ở đây vì hai lý do: số phim (13.816) ít hơn
nhiều lần số user đánh giá (158.774), nên ma trận tương đồng item×item nhỏ hơn và
đậm hơn; và mối liên quan giữa hai phim ổn định hơn mối liên quan giữa hai người —
người dùng thay đổi sở thích theo thời gian, còn phim thì không.

Đồ án cài **hai biến thể** để tách hiệu ứng của cách tính trọng số:

- `rs-itemcf` — trọng số `w_ui` = **rating thật**, cho phim user thích điểm cao đóng
  góp nhiều hơn;
- `cf-itemknn` — trọng số **nhị phân**, chỉ quan tâm có tương tác hay không.

Bảng chính dùng `rs-itemcf`; biến thể nhị phân nằm trong registry để thí nghiệm.

### 2.2.3 Matrix Factorization

Matrix factorization (Koren, Bell & Volinsky, 2009) thay ma trận user×item thưa bằng
tích hai ma trận thấp hạn:

\[
\hat{r}_{ui} = p_u^\top q_i, \qquad p_u \in \mathbb{R}^d,\ q_i \in \mathbb{R}^d
\]

Mỗi user và mỗi item được biểu diễn bằng một vector latent **d** chiều; điểm dự
đoán là tích vô hướng. Trực giác: vector latent của một phim chứa các "trục chủ
đề" ẩn (một trục có thể tương ứng với "hành động sci-fi", một trục khác với "tâm
lý trinh thám"), và vector của user chứa mức độ hứng thú với mỗi trục.

Hai lợi ích cấu trúc: (i) ma trận factor dày 158.774 × d và 13.816 × d **nhỏ hơn
nhiều** so với ma trận tương tác thưa về mặt bộ nhớ khi d đủ nhỏ; (ii) điểm dự đoán
cho **mọi cặp** user–item được tính bằng một phép nhân ma trận, không cần truy vấn
hàng xóm.

### 2.2.4 ALS cho implicit feedback

Dữ liệu MovieLens là **explicit** (người dùng chấm điểm), nhưng nhiều tương tác
trong thực tế là **implicit** (click, xem, mua) — chỉ có tín hiệu "có xảy ra", không
có mức độ. Hu, Koren & Volinsky (2008) mở rộng ALS cho trường hợp này bằng cách
gán mỗi quan sát một mức tin cậy:

\[
\min_{P,Q} \sum_{u,i} c_{ui}\,(p_u^\top q_i - r_{ui})^2 + \lambda(\|P\|^2 + \|Q\|^2)
\]

và tối ưu luân phiên: **A**lternating giữa tối ưu \(P\) (với Q cố định) và tối ưu
Q (với P cố định). Mỗi bước là một bài toán least-squares có lời giải đóng.

**Confidence weighting — chi tiết quyết định đúng/sai.** Trong công thức gốc,
\(c_{ui} = 1 + \alpha r_{ui}\) cho dữ liệu implicit, nghĩa là **mức độ tương tác
càng lớn thì tin cậy càng cao**. Áp trực tiếp công thức đó vào dữ liệu rating của
MovieLens sẽ cho một mô hình biết "user thích phim 5 sao hơn phim 4 sao" — đúng ý,
nhưng sai ở chỗ khác: **rating thấp không có nghĩa là user không thích**, nên đưa
cả rating 1–2 sao vào tập relevant với confidence cao là học sai.

Vì vậy đồ án lựa chọn: chỉ lưu tương tác **relevant** (rating ≥ 4,0) và gán
`c_ui = 1 + α` với α cố định, nhưng điều này đòi hỏi hiểu **ngữ nghĩa của thư viện**:
`implicit` coi **giá trị lưu trong ma trận chính là `c_ui`**, rồi dùng `(c−1)` cho
vế tích luỹ và `c` cho vế phải. Sai sót thường gặp là lưu `α` thay vì `1 + α`, hoặc
giữ cả rating thấp với `c = 1` — khi đó số hạng `(c−1)` bằng 0 nhưng `b += 1·factor`
vẫn cộng vào, tức mô hình học "user thích phim này với độ tin cậy 1" cho đúng những
phim user đã chấm điểm thấp.

Đây là lý do ngưỡng relevant 4,0 được dùng **thống nhất** ở ALS, MostPop, ItemKNN-CF
và nhãn test: "relevant" phải nghĩa giống nhau cho mọi thành phần.

### 2.2.5 BPR-MF

BPR (Bayesian Personalized Ranking, Rendle et al., 2009) giải quyết bài toán xếp
hạng trực tiếp bằng mục tiêu **pairwise** thay vì dự đoán rating:

\[
\mathcal{L} = \sum_{(u,i,j)} -\ln \sigma\big(\hat{r}_{ui} - \hat{r}_{uj}\big) + \lambda\|\Theta\|^2
\]

trong đó `j` được lấy bằng **negative sampling** trên các item user chưa tương tác.
Ưu điểm so với ALS: mục tiêu khớp trực tiếp với thứ tự mà ta quan tâm. Nhược điểm
thực tế: mỗi bước tối ưu là SGD, nên kết quả phụ thuộc thứ tự cập nhật tham số.

**Vì sao `num_threads=1`.** Với nhiều luồng, thứ tự cập nhật tham số phụ thuộc lịch
trình hệ điều hành, nên hai lần chạy cùng seed vẫn cho kết quả khác nhau. Đồ án
chọn **tái lập được tuyệt đối** trên tốc độ: `fit_s = 39,84` cho 158.774 user ở giao
thức `full` (nguồn: cột `fit_s` của `results/final/full-summary.csv`) thay vì ~7s khi
chạy nhiều luồng. Đây là đánh đổi đã ghi nhận, được bàn lại ở §5.3.

---

## 2.3 Content-based Filtering

### 2.3.1 Biểu diễn đặc trưng

Ba nguồn đặc trưng được dùng trong đồ án:

| Nguồn | Kích thước | Bản chất |
|---|---|---|
| `genres` | 19 chiều nhị phân | Nhãn thể loại, rất thưa |
| Tag người dùng gán | 30.067 từ sau làm sạch | Mô tả tự do, có timestamp |
| Tag Genome | 1.128 chiều liên quan | Điểm liên quan ngoài động của GroupLens |

Với tag người dùng gán, mỗi phim được biểu diễn bằng một **tag document** — chuỗi
các tag của phim đó — làm đầu vào cho TF-IDF và BM25. Với Tag Genome, ma trận
đặc trưng là mảng số 13.816 × 1.128.

### 2.3.2 TF-IDF và BM25

**TF-IDF** đo tần suất một từ trong tài liệu, điều chỉnh theo độ hiếm của từ đó
trong toàn bộ tập:

\[
tfidf(t,i) = tf(t,i) \cdot idf(t), \qquad idf(t) = \ln\frac{1+N}{1+df(t)} + 1
\]

Trong đó \(N\) là số tài liệu và \(df(t)\) là số tài liệu chứa từ \(t\).

**BM25** (Robertson & Zaragoza, 2009) giữ ý tưởng của TF-IDF nhưng thay hai yếu tố:

\[
idf_t = \ln\frac{N - df_t + 0{,}5}{df_t + 0{,}5} + 1
\qquad
w_{it} = idf_t \cdot \frac{c_{it}(k_1+1)}{c_{it} + k_1(1 - b + b\,\ell_i/\bar{\ell})}
\]

- **Bão hoà tần suất** \(c/(c+k_1)\): một tag được 500 người gán không được áp đảo
  tag được 5 người gán gấp 100 lần.
- **Chuẩn hoá độ dài tài liệu** qua \(b\): phim blockbuster có hàng nghìn tag không
  tự động giống mọi thứ.

Hai tham số \(k_1, b\) được quét trong sweep của CB-2 (21 config). Config tốt nhất
là `k1 = 0,5, b = 0` — tức **tắt chuẩn hoá độ dài tài liệu** (`b = 0`). Điều này
không bất ngờ khi đọc cùng phát hiện ở §3.2.3: **số tag mỗi phim tương quan 0,768
với độ phổ biến**, nên chuẩn hoá độ dài tài liệu ở đây chính là chuẩn hoá theo
popularity — và với bài toán xếp hạng, điều đó làm mất thông tin thay vì thêm thông
tin.

**BM25 hay hơn TF-IDF ở đây, và bằng chứng phải đọc đúng.** Có hai bảng độc lập cùng
nói một chiều:

- Ở bảng phân tầng theo kỷ nguyên (`tag-leakage-strata.csv`), **CB-2 BM25 dẫn CB-1
  TF-IDF ở cả ba tầng**: 0,0824 / 0,1263 / 0,1094 (BM25) so với 0,0707 / 0,1256 /
  0,1083 (TF-IDF).
- Ở giao thức `full` (bảng chính §4.1), BM25 cũng cao hơn: **0,0998 > 0,0937**.

> **Cảnh báo đọc nhầm.** Nhãn "CB-1 TF-IDF" trong **cả hai** bảng trên đều trỏ tới
> **cùng một config time-decay** `cb4-03-timedecay-hl0365` — không phải config TF-IDF
> thuần `cb1-01-add-tags-tfidf` (NDCG@10 ở `dev` = 0,091437). Khi so sánh phải ghi rõ
> **tên config**, không so hai số trần. Tương tự, bảng phân tầng kỷ nguyên phân tầng
> theo **thời kỳ user** — nó **không** phải bằng chứng cho phát biểu chung "BM25 tốt
> hơn TF-IDF" mà phải đọc cùng bảng chính.

**Chi phí tính toán — số cần nêu vì CB thắng cold-start nhưng chậm hơn nhiều.** Cột
`score_s` ở giao thức `full` cho thấy CB-2 BM25 cần **115,14s** và CB-1 TF-IDF
**105,52s** để chấm điểm toàn bộ 158.774 user, so với ALS **24,91s** và BPR
**21,67s**. Lý do: mô hình nội dung phải tính hồ sơ user rồi nhân với toàn bộ ma
trận đặc trưng 13.816 × D cho mỗi lô — một phép toán tuyến tính theo số user lớn hơn
nhiều so với một phép nhân factor.

### 2.3.3 Dựng hồ sơ người dùng

Mô hình nội dung không dự đoán trực tiếp mà đi qua **hồ sơ**: vector trung bình có
trọng số của các item user đã tương tác, trong không gian đặc trưng:

\[
P_u = \mathrm{normalize}\left(\sum_{i \in \text{seen}(u)} w_{ui}\, X_i\right),
\qquad S_u = X P_u^\top
\]

Ba chế độ trọng số được khảo sát:

| Chế độ | Ý nghĩa |
|---|---|
| `binary_pos` | 1 nếu rating ≥ 4,0, 0 ngược lại |
| `rating` | Giữ nguyên giá trị rating |
| `centered` | `r − ū_train` — phim không thích kéo hồ sơ ra xa thể loại đó |

Thêm nữa là tham số `half_life_days` cho hệ số suy giảm theo thời gian
\(0{,}5^{\Delta t / t_{1/2}}\): phim xem lâu rồi đóng góp ít hơn vào hồ sơ.

**Kết quả đo được ở §4.8** và nó khá bất ngờ: `centered` **tệ hơn hẳn**
`binary_pos` ở cả ba nhóm user (chênh −0,023 / −0,020 / −0,031, `p < 1e-44`), còn
`rating` gần như tương đương `binary_pos`. Nghĩa là tín hiệu quan trọng là **có tương
tác hay không**, không phải **mức độ hài lòng** — một phát hiện có giá trị vì nó
giảm tham số cần chọn cho hệ thống thật.

### 2.3.4 Tag Genome và vấn đề metadata hậu kỳ

Tag Genome của GroupLens là 1.128 tag, mỗi tag gắn với mỗi phim một điểm liên quan
thực (relevance). Khác với `tags.csv` — vốn là hành vi gán tag của người dùng, có
timestamp — **`genome-scores.csv` không có timestamp**: GroupLens tính nó ngoài động
trên toàn bộ corpus.

Hệ quả có thể nghiêm trọng: nếu Tag Genome được tính từ dữ liệu bao gồm cả các phim
chưa phát hành tại thời điểm một user đang xem, thì mô hình content-based dựa trên
genome có thể "biết trước" những phim tương lai — và mọi kết luận về độ chính xác
của CB sẽ lạc quan.

Đây là **rủi ro phải nêu trước khi trích dẫn kết quả CB** (§3.2.3 đặt mức ban đầu
của nó; §4.6–§4.7 đo định lượng và phân tầng). Đáng chú ý, `tags.csv` **có**
timestamp nhưng bắt đầu muộn — ngày gán tag đầu tiên là **2005-12-24** — nên cả
hai nguồn đều cần kiểm tra, không chỉ genome.

---

## 2.4 Metric đánh giá bài toán xếp hạng

### 2.4.1 Vì sao không dùng RMSE/MAE

RMSE và MAE đo **sai số dự đoán rating**, phù hợp với bài toán "dự đoán người dùng
sẽ chấm phim này bao nhiêu điểm". Bài toán ở đây là **xếp hạng**: đưa 10 phim lên
đầu. Hai lý do khiến RMSE không phù hợp:

1. **Nó phớt lờ thứ tự.** Hai mô hình có thể cho RMSE bằng nhau nhưng một mô hình
   đặt phim đúng ở vị trí 1, mô hình kia đặt ở vị trí 10 — về mặt sản phẩm đó là
   hai hệ thống rất khác nhau.
2. **Nó phớt lờ mức độ quan tâm.** Một mô hình dự đoán sai 0,2 điểm cho phim đứng
   thứ 500 không quan trọng bằng việc xếp sai phim đáng xem ở vị trí 3.

Metric đánh giá phải phản ánh **thứ tự** và **vị trí**, nên đồ án dùng bốn metric
dựa trên tập relevant nhị phân (rating ≥ 4,0):

### 2.4.2 Precision@k, Recall@k, HR@k, NDCG@k

Với user \(u\), danh sách top-k \(\hat{L}_u\), tập relevant \(R_u\):

| Metric | Định nghĩa | Ý nghĩa |
|---|---|---|
| **Precision@k** | \(\dfrac{\|R_u \cap \hat{L}_u\|}{k}\) | Trong \(k\) gợi ý, bao nhiêu là đúng |
| **Recall@k** | \(\dfrac{\|R_u \cap \hat{L}_u\|}{\|R_u\|}\) | Bao nhiêu phim thật sự muốn xem đã được tìm ra |
| **HR@k** | \(\mathbb{1}[\|R_u \cap \hat{L}_u\| > 0]\) | User nào tìm được **ít nhất một** thứ muốn xem |
| **NDCG@k** | \(\dfrac{DCG@k}{IDCG@k}\) | Phiên bản của HR có **trọng số theo vị trí** |

Trong đó:

\[
DCG@k = \sum_{i=1}^{k} \frac{\mathrm{rel}(i)}{\log_2(i+1)},
\qquad
IDCG@k = \sum_{i=1}^{\min(k,|R_u|)} \frac{1}{\log_2(i+1)}
\]

NDCG (Järvelin & Kekäläinen, 2002) là metric chính của đồ án vì nó phạt **mạnh hơn**
việc xếp sai ở vị trí đầu: đặt phim đúng ở vị trí 1 và vị trí 10 cho DCG khác nhau
nhiều hơn so với vị trí 10 và 11.

> **Định nghĩa đúng của `P@k` là `hits / k`** — phân biệt với `hits / len(top_k)`.
> Với `k = 10` và danh sách luôn đủ 10 phim thì hai cách trùng nhau, nhưng định
> nghĩa đúng phải là `hits/k` và evaluator của đồ án dùng đúng cách đó.

### 2.4.3 Metric ngoài độ chính xác

Một hệ thống có NDCG@10 cao vẫn có thể là hệ thống tệ: nó gợi ý 10 phim giống hệt
nhau cho mọi người. Đồ án bổ sung bốn metric mô tả **hành vi**:

| Metric | Định nghĩa | Câu hỏi trả lời |
|---|---|---|
| **Coverage@k** | Tỉ lệ item từng xuất hiện trong top-k của ít nhất một user | Hệ thống có chạm được bao nhiêu phim trong catalog? |
| **Gini@k** | Bất đẳng thức Gini của mức độ phổ biến trong gợi ý | Gợi ý có dồn suất vào phim phổ biến không? |
| **Novelty@k** | Trung bình \(-\log_2 p(i)\) của các item được gợi ý | Gợi ý có lạ không? |
| **ILD@k** | Trung bình khác biệt nội dung giữa các cặp trong danh sách | Danh sách có đa dạng không? |

Kết quả ở §4.9 cho thấy **coverage đảo ngược với độ chính xác**: ALS chính
xác nhất (NDCG@10 = 0,1375) nhưng chỉ chạm **14,94%** catalog; CB-3 Tag Genome kém
nhất trong nhóm CB (0,0713) nhưng chạm **67,20%**. Đây là phát hiện quan trọng về
thực tiễn — không phải mô hình nào "tốt nhất" nếu không nói mục tiêu là gì.

### 2.4.4 Kiểm định thống kê

Với 158.774 user, chênh lệch NDCG@10 giữa ALS (0,1375) và CB-4 (0,1054) là lớn
về mặt thực tế — nhưng "lớn" là cảm tính. Đồ án dùng hai công cụ:

- **Bootstrap CI 95%** (`bootstrap-ci.csv`): lấy lại mẫu user với hoàn lại (with
  replacement) nhiều lần, mỗi lần tính lại metric. Kết quả: ALS [0,1366; 0,1383]
  không chạm CI của CB-4 [0,1047; 0,1062] — khoảng cách **lớn hơn nhiễu thống kê**.
- **Wilcoxon signed-rank** ghép cặp (`wilcoxon-{cf,cb,coldstart}.csv`): kiểm định
  phân phối khác nhau trên **cùng cặp user**, mạnh hơn khi phân phối metric lệch
  mạnh (đây là trường hợp của NDCG).

> **Giới hạn của cách dùng Wilcoxon ở đây:** kiểm định được chạy **trong từng bảng**
> (CF vs CF, CB vs CB, cold-start vs cold-start), **không** so chéo CF↔CB — vì hai
> nhóm được tinh chỉnh với số config khác nhau, nên so chéo sẽ trộn lẫn hiệu ứng của
> "thuật toán" với hiệu ứng của "mức độ tinh chỉnh". Việc so chéo CF↔CB được nêu là
> hướng phát triển (§5.8).

---

## 2.5 Tóm tắt

| Khái niệm | Cơ sở lý thuyết | Cài đặt cụ thể trong đồ án |
|---|---|---|
| CF dựa trên hành vi | Resnick et al. (1994); Koren et al. (2009) | ALS (§3.8.1), Item-CF (§3.8.2), BPR-MF (§3.8.3) |
| CF cho implicit feedback | Hu et al. (2008) | `c_ui = 1 + α`, chỉ tương tác relevant |
| CB dựa trên nội dung | TF-IDF; BM25 | CB-1, CB-2 (§3.7.2) |
| CB dựa trên Tag Genome | GroupLens genome-scores | CB-3 (§3.7.2) |
| Kết hợp | late fusion | CB-4 (§3.7.2) |
| Đo bài toán xếp hạng | Järvelin & Kekäläinen (2002) | NDCG@10 là metric chính (§2.4.2) |
| Đo hành vi hệ thống | Shani & Gunawardana (2011) | coverage, Gini, novelty, ILD (§2.4.3) |
| Rủi ro rò rỉ metadata | — | Phân tầng 3 kỷ nguyên (§4.6–§4.7) |

---

**Tài liệu tham khảo:** xem [Phần Tài liệu tham khảo](../report/README.md) và danh
sách đầy đủ ở cuối báo cáo.
