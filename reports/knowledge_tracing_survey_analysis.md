# Phân tích bài báo: *Knowledge Tracing: A Survey* (arXiv:2201.06953)

> **Chất lượng PDF:** Tốt (trích xuất được toàn văn; riêng Bảng 7 in xoay ngang nên đã đối chiếu thêm bằng ảnh trang 25) | **Loại bài báo:** Tổng quan (Survey) | **Chế độ phân tích:** standard

**Quy ước nhãn nguồn** (theo skill `paper-analyst`):

| Nhãn | Ý nghĩa |
|---|---|
| `[Nguyên văn]` | Bài báo nói trực tiếp như vậy (có ghi vị trí) |
| `[Suy luận]` | Do người phân tích rút ra (có nêu căn cứ) |
| `[Không nêu rõ]` | Bài báo không cung cấp thông tin này |
| `[Không chắc chắn]` | Thông tin mơ hồ hoặc mâu thuẫn |

**Xếp loại bài báo:** Tổng quan (Survey). Căn cứ:
1. Tiêu đề có chữ "A Survey".
2. Bài tổng hợp 126 tài liệu tham khảo và hơn 25 mô hình KT (Bảng 5).
3. Bài **không đề xuất phương pháp mới và không tự chạy thí nghiệm**. Bảng 7 chỉ gom lại các kết quả do những bài khác đã công bố.

> Vì đây là bài tổng quan, phần "Phương pháp" bên dưới là **cách bài tổng quan được xây dựng** (phạm vi, cách phân loại, các tiêu chí so sánh). Phần "Kết quả" là **các kết quả mà bài tổng hợp lại**, không phải kết quả thí nghiệm của chính tác giả.

---

## 1. Thông tin cơ bản

| Trường | Nội dung |
|---|---|
| Tiêu đề | *Knowledge Tracing: A Survey* (Tổng quan về Truy vết tri thức) |
| Tác giả | Ghodai Abdelrahman, Qing Wang, Bernardo Pereira Nunes |
| Đơn vị | School of Computing, Australian National University (Canberra) |
| Nơi công bố | `[Không nêu rõ]`: mẫu định dạng ACM có ghi "Vol. 1, No. 1", nhưng DOI vẫn là placeholder `10.1145/nnnnnnn.nnnnnnn`, nên chưa xác định được tạp chí hay hội nghị |
| Năm | 2022 (arXiv v1 ngày 8/1/2022) |
| arXiv | 2201.06953v1 [cs.CY] |
| Từ khóa | Knowledge Tracing, Memory Networks, Deep Learning, Sequence Modelling, Key-Value Memory, BKT, Intelligent Education, Factor Analysis, survey |

---

## 2. Dịch tóm tắt và giải thích dễ hiểu

**Bản dịch tóm tắt:**
Khả năng truyền đạt tri thức qua việc dạy học là một khía cạnh cốt lõi của trí thông minh con người. Một giáo viên có thể theo dõi tri thức của học sinh để điều chỉnh cách dạy theo nhu cầu của từng em. Khi các nền tảng giáo dục trực tuyến phát triển, máy tính cũng cần theo dõi tri thức của người học và cá nhân hóa trải nghiệm học tập. Trong tài liệu, bài toán này được gọi là **Knowledge Tracing (KT)**. Giải tốt bài toán KT sẽ mở ra nhiều ứng dụng giáo dục có máy tính hỗ trợ, như hệ thống gia sư thông minh (ITS), học theo lộ trình (curriculum learning) và gợi ý học liệu. Rộng hơn, "người học" có thể là bất kỳ tác tử thông minh nào, kể cả tác tử nhân tạo (một mô hình học máy), nên KT còn áp dụng được cho các bài toán *machine teaching*. Bài báo tổng quan một cách toàn diện và có hệ thống về KT: từ các phương pháp đầu tiên đến các phương pháp học sâu hiện đại; làm rõ khía cạnh lý thuyết của mô hình và đặc điểm của các bộ dữ liệu chuẩn; chỉ ra khác biệt then chốt giữa các phương pháp gần nhau; và cuối cùng bàn về các khoảng trống nghiên cứu và hướng đi tương lai.

**Giải thích dễ hiểu:**
- KT là bài toán: **dựa vào lịch sử làm bài của một học sinh** (câu nào đúng, câu nào sai), **đoán xem em đó có làm đúng câu tiếp theo không**, đồng thời ước lượng em đã nắm từng kỹ năng đến đâu.
- Bài toán khó vì: một câu hỏi có thể cần nhiều kỹ năng; kỹ năng này phụ thuộc kỹ năng khác (phải biết cộng trước khi học nhân); và học sinh **quên** theo thời gian.
- Bài báo đi lại toàn bộ lịch sử phát triển: từ mô hình xác suất cổ điển (BKT), mô hình hồi quy (IRT/PFA), cho đến RNN, memory network, Transformer và GNN. Bài cũng tổng hợp các bộ dữ liệu và bảng kết quả AUC.
- Ý nghĩa: đây là "bản đồ" để người mới vào lĩnh vực biết nên dùng mô hình nào, dữ liệu nào, và cần cẩn trọng gì khi so sánh kết quả.

---

## 3. Bối cảnh

Khái niệm KT do Anderson và cộng sự đưa ra năm 1986, ban đầu phục vụ mô hình hóa nhận thức trong hệ thống gia sư thông minh. Lĩnh vực đã đi qua ba giai đoạn:
1. **Bayesian Knowledge Tracing (BKT):** dùng HMM với giả định đơn giản hóa mạnh, ví dụ mỗi câu chỉ cần một kỹ năng và không có hiện tượng quên.
2. **Mô hình phân tích nhân tố (Factor Analysis):** dựa trên Item Response Theory (IRT) và hồi quy logistic.
3. **Học sâu:** bắt đầu từ DKT (Piech et al., 2015).

Sự bùng nổ của MOOC, ITS và việc học trực tuyến bắt buộc trong COVID-19 khiến nhu cầu theo dõi tiến độ người học tự động trở nên cấp thiết. Đã có một survey trước đó (Liu et al. 2021, [65]). Các tác giả cho rằng bài của họ khác ở bốn điểm: bao phủ mô hình rộng hơn; phân tích sâu hơn cách biểu diễn trạng thái tri thức, hiện tượng quên và knowledge component; mổ xẻ chi tiết các bộ dữ liệu; và tổng hợp bảng kết quả từ nhiều nguồn để hỗ trợ so sánh về sau `[Nguyên văn, §1.1]`.

---

## 4. Phân tích phương pháp (cách xây dựng bài tổng quan)

### 4.1 Phạm vi và chiến lược tìm tài liệu

- **Năm câu hỏi nghiên cứu** `[Nguyên văn, §1.2]`:
  - RQ1: lịch sử các kỹ thuật KT truyền thống
  - RQ2: các kỹ thuật học sâu đã áp dụng cho KT
  - RQ3: các bộ dữ liệu được thu thập, tiền xử lý và dùng làm chuẩn
  - RQ4: các lĩnh vực ứng dụng KT
  - RQ5: cơ hội và thách thức nghiên cứu tương lai
- **Chiến lược tìm kiếm tài liệu:** `[Không nêu rõ]`. Bài không mô tả từ khóa tìm kiếm, cơ sở dữ liệu thư mục, khoảng thời gian hay tiêu chí chọn hoặc loại bài. Đây không phải một systematic review theo chuẩn PRISMA, dù tóm tắt dùng chữ "systematic".
- **Dữ liệu:** "Tất cả bộ dữ liệu công khai đều được tải về, kiểm tra và báo cáo thông tin liên quan" `[Nguyên văn, §3]`. Đây là phần tác giả **tự kiểm tra lại**, không chỉ trích dẫn.

### 4.2 Định nghĩa bài toán

**Định nghĩa 2.1** `[Nguyên văn, §2.2.1]`. Cho chuỗi tương tác `X = ⟨x₁, …, x_{t−1}⟩`, trong đó `xᵢ = (qᵢ, yᵢ)` và `yᵢ ∈ {0,1}`. Bài toán là dự đoán xác suất trả lời đúng câu mới:

```
p_t = P(y_t = 1 | q_t, X)
```

Đây là bài toán **dự đoán chuỗi, phân loại nhị phân**, đánh giá bằng **AUC-ROC**.

### 4.3 Cách phân loại

```
Knowledge Tracing
├── Truyền thống (§2.1)
│   ├── Bayesian KT: Standard BKT · Individualized BKT · Dynamic BKT (DBN)
│   └── Factor Analysis: IRT (1PL/4PL, HIRT, TIRT) · AFM · PFA · KTM (+KTM-DLF)
└── Học sâu (§2.2, Hình 5)
    ├── Sequence modelling: DKT, DKT+, DKT-DSC, Extended-DKT
    ├── Memory-augmented: DKVMN, SKVMN
    ├── Attentive: SAKT, AKT, SAINT, SAINT+, RKT
    ├── Graph-based: GKT, GIKT, SKT
    ├── Text-aware: EERNN (M/A), EKT, (QuesNet)
    └── Forgetting-aware: DKT+forget, KPT/EKPT, HawkesKT, DGMN
```

### 4.4 Ý tưởng cốt lõi của từng nhóm (tóm tắt từ bài)

| Nhóm | Ý tưởng / công thức chính | Hạn chế bài nêu |
|---|---|---|
| **Standard BKT** | HMM hai trạng thái {chưa học, đã học}, 4 tham số cho mỗi kỹ năng: `p(L₀)` (biết từ trước), `p(T)` (học được), `p(S)` (sơ suất), `p(G)` (đoán mò). Cập nhật: `p(Lₙ) = Post(Lₙ₋₁) + (1−Post(Lₙ₋₁))·p(T)` (Eq. 2) | Không có quên; tham số giống nhau cho mọi học sinh; mỗi câu chỉ một kỹ năng |
| **Individualized BKT** | Thêm tham số riêng cho từng học sinh (Bảng 2). Cá nhân hóa `p(T)` có lợi hơn cá nhân hóa `p(L₀)` | Vẫn giữ HMM gốc |
| **Dynamic BKT (DBN)** | Mô hình nhiều kỹ năng cùng lúc và quan hệ tiên quyết giữa chúng; thêm `p(F)` (quên); học trọng số bằng mô hình log-linear | Thường khó tính toán chính xác (intractable) |
| **IRT** | `p_ij = σ(θᵢ − b_j)` (Rasch/1PL); 4PL thêm độ phân biệt `a`, đoán mò `c`, sơ suất `d` | Không xét thứ tự thời gian |
| **AFM / PFA** | AFM: `σ(θᵢ + Σ(β_k + γ_k·T_ik))`. PFA bỏ `θᵢ`, đếm riêng số lần đúng và số lần sai | Như trên |
| **KTM** | Factorization Machine: hạng tuyến tính cộng tương tác bậc hai `⟨v_k, v_l⟩`. IRT, AFM, PFA là các trường hợp đặc biệt. KTM-DLF thêm yếu tố quên | — |
| **DKT** | RNN/LSTM: `h_t = tanh(W_hx·x_t + W_hh·h_{t−1} + b)`, `p_t = σ(W_hy·h_t + b)` | Trạng thái ẩn gộp chung, không tách theo KC; không mô hình quan hệ giữa KC; coi mọi câu trước đó quan trọng như nhau |
| **DKVMN** | Ma trận **key** cố định (biểu diễn KC) và ma trận **value** động (mức thành thạo); đọc bằng `r_t = Σ w_t(i)·M_v(i)` | Không lọc các câu hỏi không liên quan trong chuỗi |
| **SKVMN** | Hop-LSTM: chỉ nối các ô LSTM của những câu hỏi liên quan nhau, giúp bắt phụ thuộc dài hạn | — |
| **SAKT / AKT / SAINT(+) / RKT** | Self-attention theo Transformer. AKT dùng *monotonic attention* với suy giảm mũ theo khoảng thời gian để mô hình hóa việc quên. SAINT dùng encoder–decoder; SAINT+ thêm elapsed time và lag time. RKT nhân trọng số attention với hệ số quan hệ (văn bản câu hỏi) và hệ số quên | — |
| **GKT / GIKT / SKT** | GNN trên đồ thị KC–KC (GKT, SKT) hoặc câu hỏi–KC (GIKT). GKT tự dựng đồ thị bằng thống kê hoặc học end-to-end | GKT và SKT giả định mỗi câu một KC |
| **EERNN / EKT** | Bi-LSTM mã hóa văn bản câu hỏi. EKT dùng ma trận trạng thái tri thức cho nhiều KC | — |
| **Forgetting-aware** | DKT+forget thêm 3 đặc trưng thời gian. KPT dựa trên đường cong học/quên. HawkesKT dùng quá trình Hawkes để mô hình ảnh hưởng chéo theo thời gian. DGMN kết hợp đồ thị KC động, bộ nhớ và cổng (gating) | — |

### 4.5 Các tiêu chí so sánh

**Truyền thống** (§2.1.3, `[Nguyên văn]`):
- **Tham số:** BKT ban đầu có 4 tham số; FA linh hoạt hơn, KTM nhận được tùy ý thông tin phụ.
- **Suy diễn:** BKT dùng xích Markov bậc 1 nên dễ tính nhưng bị đơn giản hóa; FA không suy diễn trạng thái tri thức một cách tường minh.
- **Thời gian:** BKT là mô hình chuỗi; FA không quan tâm thứ tự trả lời.

**Học sâu** (§2.2.7, `[Nguyên văn]`):
- **Trạng thái tri thức:** xu hướng chuyển từ một vector ẩn (DKT) sang ma trận theo từng KC (DKVMN, SKVMN, EKT).
- **Quan hệ giữa KC:** học bằng attention hoặc bằng GNN.
- **Đặc trưng bổ sung** (thời gian, văn bản): giúp tăng độ chính xác nhưng phụ thuộc vào việc dataset có các đặc trưng đó hay không.

**Bảng 5** tổng hợp 29 mô hình theo bốn trục: *mô hình học · KC đơn hay đa · dạng biểu diễn trạng thái tri thức · có xét quên hay không*. Đây là đóng góp hệ thống hóa chính của bài.

---

## 5. Đóng góp

- `[Nguyên văn]` **Phân loại và so sánh kiến trúc các mô hình KT** theo thiết kế mô hình, cách biểu diễn trạng thái tri thức, giả định quan hệ câu hỏi–kỹ năng, và việc có xét hành vi quên hay không.
  Bằng chứng: §1.1 gạch đầu dòng 1; Hình 3, Hình 5; Bảng 3, 4, 5.
- `[Nguyên văn]` **Trình bày tiến trình phát triển theo thời gian** của từng nhóm mô hình và cách phương pháp sau mở rộng phương pháp trước.
  Bằng chứng: §1.1 gạch đầu dòng 2; §2.1.1–2.2.6 (ví dụ: DKT → DKVMN → SKVMN; SAKT → SAINT → SAINT+).
- `[Nguyên văn]` **Tổng hợp đặc điểm các bộ dữ liệu và kết quả AUC** từ nhiều bài khác nhau.
  Bằng chứng: §1.1 gạch đầu dòng 3; §3; Bảng 6, Bảng 7.
- `[Suy luận]` **Kiểm tra dữ liệu độc lập và chỉ ra lỗi cụ thể** là phần có giá trị thực tiễn nhất.
  Căn cứ: tác giả tự tải và kiểm tra dữ liệu (§3), phát hiện timestamp sai trong Algebra 2006-2007, 126.908/179.999 câu của ASSIST2012 không gắn KC, ASSIST2009 bản gốc bị trùng bản ghi, và đề xuất quản lý phiên bản cho dataset (§3.7). Phần lớn các survey khác chỉ trích dẫn số liệu, không tự kiểm tra lại.
- `[Nguyên văn]` **Mở rộng phạm vi ứng dụng của KT sang "Learning to Teach"**: người học có thể là một mô hình học máy, và KT dùng để theo dõi "tri thức" của mô hình đó trong lúc huấn luyện.
  Bằng chứng: tóm tắt; §4.4 (ví dụ KADT [3], một công trình của chính nhóm tác giả).

---

## 6. Kết quả và kết luận

> ⚠️ Mọi số liệu dưới đây là **AUC do các bài gốc công bố**, được bài tổng quan gom vào Bảng 7. Tác giả **không chạy lại** các mô hình. Số trong ngoặc vuông `[n]` là chỉ số tài liệu nguồn trong survey.

### 6.1 Thống kê bộ dữ liệu (Bảng 6, `[Nguyên văn]`)

| Dataset | #Câu hỏi | #Học sinh | #Tương tác | #KC | Ghi chú của bài |
|---|---:|---:|---:|---:|---|
| ASSISTments2009 | 26.688 | 4.217 | 346.860 | 123 | Bản gốc có bản ghi trùng; nên dùng bản "skill-builder"; chỉ khoảng 2/3 số câu có KC |
| ASSISTments2012 | 179.999 | 46.674 | 6.123.270 | 265 | 126.908 câu không có KC, có thể là lý do AUC thấp |
| ASSISTments2015 | 100 | 19.917 | 708.631 | – | Chỉ có 4 thuộc tính, không có KC |
| ASSISTChall | 3.162 | 1.709 | 942.816 | 102 | Thông tin mô tả phong phú nhất |
| STATICS2011 | 1.224 | 335 | 361.092 | 85 | Không công khai (cấp khi có yêu cầu); sau tiền xử lý còn 189.297 tương tác |
| Junyi Academy | 722 | 247.606 | 25.925.992 | 41 | Các bài dùng tập con khác nhau nên không so sánh trực tiếp được |
| Simulated-5 | 50 | 4.000 | 200.000 | 5 | Dữ liệu tổng hợp sinh bằng IRT, dễ so sánh nhất |
| Algebra 2005-06 | 1.084 | 575 | 813.661 | 112 | — |
| Algebra 2006-07 | 90.831 | 1.840 | 2.289.726 | 523 | **Timestamp sai** |
| Bridge to Algebra | 19.258 | 1.146 | 3.686.871 | 493 | — |
| EdNet-KT1…KT4 | 13.169 | đến 784.309 | đến 131.441.538 | 188–293 | Lớn nhất; luyện thi TOEIC ở Hàn Quốc |

### 6.2 AUC tiêu biểu (trích Bảng 7, `[Nguyên văn]`)

Chia train/test là 80/20, trừ DKVMN, SKVMN và DGMN dùng 70/30.

| Mô hình | ASSIST2009 | ASSIST2012 | ASSIST2015 | STATICS | Junyi | Synthetic | EdNet |
|---|---|---|---|---|---|---|---|
| BKT | 0.651 [73] · 0.67 [89] · 0.6571 [120] | 0.623 [73] · 0.6204 [120] | 0.611 [73] · 0.678 [108] | – | 0.68 [89] · 0.831 [108] | 0.54 [89] | 0.6027 [120] |
| IRT | 0.5869 [114] … 0.7651 [116] | 0.634 [114] … 0.743 [73] | 0.672 [73] | – | – | – | – |
| PFA | 0.6849 [111] · 0.703 [73] | 0.670 [73] · 0.669 [35] | 0.689 [73] | – | – | – | – |
| KTM | 0.7169 [120] … 0.8186 [111] | 0.6788 [120] · 0.7535 [114] | – | – | – | – | 0.6888 [120] |
| **DKT** | **0.709 [78] … 0.86 [89]** (9 giá trị) | 0.712 [80] … 0.7308 [114] | 0.707 [73] … 0.7365 [121] | 0.815 [79] … 0.8233 [36] | 0.814 [80] · 0.85 [89] · 0.847 [108] | 0.75 [89] … 0.8255 [121] | 0.6822 [120] · 0.7638 [99] |
| DKT+ | 0.8024 [36] … 0.8227 [121] | – | 0.728 [108] … 0.7371 [121] | 0.8301 [36] … 0.835 [79] | 0.889 [108] | 0.824 [79] · 0.8264 [121] | – |
| DKVMN | 0.8157 [1][125] | – | 0.7268 | 0.8284 | 0.8027 [1] | 0.8273 | – |
| SKVMN | 0.8363 [1] | – | 0.7484 [1] | 0.8485 [1] | 0.8267 [1] | 0.840 [1] | – |
| SAKT | 0.686 [114] … 0.848 [79] | 0.6906 [114] · 0.735 [80] | 0.7212 [36] · 0.854 [79] | 0.8029 [36] · 0.853 [79] | 0.834 [80] | 0.832 [79] | 0.7663 [99] · 0.7671 [21] |
| AKT | 0.7474 [114] · 0.8346 [36] | 0.7555 [114] | 0.7828 [36] | – | – | – | – |
| SAINT / SAINT+ | – | – | – | – | – | – | 0.7811 [21] / **0.7914 [99]** |
| RKT | – | **0.793 [80]** | – | – | 0.860 [80] | – | – |
| GKT | 0.723 [78] | 0.735 [108] | – | – | 0.893 [108] | – | – |
| GIKT | 0.7896 [120] | 0.7754 [120] | – | – | – | – | 0.7523 [120] |
| SKT | – | – | 0.746 [108] | – | **0.908 [108]** | – | – |
| HawkesKT | 0.763 | 0.768 | – | – | – | – | – |
| DGMN | 86.1 [2]* | – | – | 86.4 [2]* | – | 85.9 [2]* | – |

\* DGMN ghi theo thang phần trăm trong bảng gốc, tức 0.861, 0.864, 0.859.

Kết quả trên tập KDDcup (`[Nguyên văn]`, Bảng 7):
- Algebra 2005-06: KTM-DLF đạt 0.837 [35]; DGMN đạt 83.4 [2].
- Bridge 2006-07: HIRT đạt 0.8597 [116] và DKT đạt 0.8901 [116].

### 6.3 Đọc kết quả

- `[Suy luận]` **Cùng một mô hình trên cùng một dataset, AUC dao động rất lớn giữa các bài.** Ví dụ DKT trên ASSIST2009 nằm trong khoảng 0.709 đến 0.86; SAKT trên ASSIST2015 có hai giá trị 0.7212 và 0.854. Độ chênh này **lớn hơn khoảng cách giữa các họ mô hình**, nên không thể xếp hạng mô hình chỉ từ Bảng 7. Tác giả cũng nhận định như vậy (§3.7).
- `[Suy luận]` **Xu hướng chung:** các mô hình học sâu (DKT trở về sau) thường vượt BKT và AFM/PFA; mô hình có attention, đồ thị hoặc bộ nhớ thường nhỉnh hơn DKT. Tuy vậy, IRT/HIRT vẫn cạnh tranh được: trên Bridge 2006-07, HIRT đạt 0.8597 và DKT đạt 0.8901, cả hai số cùng từ nguồn [116]. Điều này khớp với nhận định của tác giả rằng hướng FA "vẫn đang hoạt động và là lựa chọn thay thế" `[Nguyên văn, §1]`.
- `[Suy luận]` Không có mô hình nào được báo cáo đầy đủ trên mọi dataset. Bảng 7 rất thưa, nên không có "SOTA chung" thật sự.

### 6.4 Kết luận của tác giả `[Nguyên văn, §6]`

- Học sâu là bộ công cụ chủ đạo của các phương pháp KT tiên tiến nhất hiện nay.
- Survey cung cấp cách phân loại theo các khía cạnh lý thuyết và thuật toán chung, kèm tiến trình phát triển theo thời gian.
- Dataset có nhiều vấn đề: trùng lặp, timestamp sai, thuật ngữ mơ hồ, định dạng CSV dồn nhiều KC vào một ô bằng ký tự `~~`, thiếu đa dạng (chủ yếu môn toán, chủ yếu từ Mỹ và Hàn Quốc), thiếu thông tin nhân khẩu học, và không quản lý phiên bản (§3.7).
- **Hướng nghiên cứu tương lai** (§5):
  - Biểu diễn đa phương thức và dataset giàu ngữ cảnh: ảnh, công thức toán, đoạn code (RD1–RD3).
  - Học tự giám sát và mô hình tiền huấn luyện cho KT, bao gồm xử lý cold-start (RD4–RD5).
  - KT tương tác: chủ động chọn câu hỏi để hỏi, ví dụ bằng học tăng cường (RD6).
  - AI giải thích được (XAI) cho KT (RD7–RD8).

### 6.5 Hạn chế

**Hạn chế tác giả thừa nhận** `[Nguyên văn]`:
- Kết quả trong Bảng 7 không so sánh trực tiếp được với nhau khi thiết lập thí nghiệm chưa thống nhất (§3.7).
- Kết quả trên dữ liệu chưa khử trùng có thể bị thổi phồng (§3.7, dẫn [116]).

**Hạn chế do người phân tích nhận ra** `[Suy luận]`:
1. **Không có phương pháp luận tìm tài liệu.** Bài không nêu từ khóa, cơ sở dữ liệu, khoảng thời gian hay tiêu chí chọn bài, nên khó đánh giá độ bao phủ và khó lặp lại.
2. **Không tự chạy lại thí nghiệm trên một thiết lập thống nhất.** Chính tác giả chỉ ra vấn đề so sánh không công bằng nhưng không giải quyết nó bằng benchmark riêng. Bảng 7 vì thế chủ yếu mang tính tham khảo.
3. **Mâu thuẫn giữa các bảng** `[Không chắc chắn]`:
   - SAINT+ được đánh dấu *có* xét quên ở Bảng 3 nhưng *không* ở Bảng 5.
   - RKT cũng *có* ở Bảng 3 nhưng *không* ở Bảng 5.
   - DKT+ (vốn chỉ thêm regularization) lại được đánh dấu có xét quên ở Bảng 5.
   - §3.1 viết "(see Table 6)" khi nói về hiệu năng, trong khi hiệu năng nằm ở Bảng 7.
4. **Thiên lệch tự trích dẫn.** SKVMN [1], DGMN [2], KADT [3] đều của nhóm tác giả. DGMN có số liệu duy nhất từ chính bài của nó, ở thang phần trăm, với tỉ lệ chia 70/30 khác các mô hình còn lại.
5. **Mốc thời gian.** Bài dừng ở đầu năm 2022, nên chưa bao phủ các hướng sau đó như mô hình KT dựa trên LLM hay các benchmark chuẩn hóa sau này (ví dụ pyKT). Đây là giới hạn tự nhiên của thời điểm viết, không phải lỗi của bài.
6. **Thiếu chỉ số khác ngoài AUC.** Bài chỉ tổng hợp AUC, không có ACC, RMSE hay chi phí tính toán, nên khó đánh giá đánh đổi giữa hiệu năng và chi phí.

---

*Nguồn:* phân tích dựa trên toàn văn arXiv:2201.06953v1 (36 trang). Không dùng tìm kiếm bên ngoài cho các thông tin về bài báo.
