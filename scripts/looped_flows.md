# Kịch Bản Thuyết Trình: Thinking with Looped Flows

- **Bài báo:** *Thinking with Looped Flows* (arXiv:2609.11801, 2026)
- **Tác giả:** Ayhan Suleymanzade, Chanhyuk Lee, Floor Eijkelboom, Nicholas M. Boffi, İsmail İlkan Ceylan, Jinwoo Kim (EPFL, KAIST, CMU, Amsterdam...)
- **File slide LaTeX tương ứng:** `overleaf/looped_flows/lf_paper.tex` (hoặc `overleaf/looped_flows/main.tex`)
- **Người trình bày:** Hung Manh
- **Thời lượng dự kiến:** 20 – 25 phút (bao gồm Q&A)

---

## 📌 Hướng Dẫn & Phân Bổ Thời Gian
- **Mở đầu & Bối cảnh (Slide Bìa $\to$ Slide 2):** ~5 phút.
- **Phương pháp cốt lõi (Slide 3 $\to$ Slide 6):** ~8 – 9 phút.
- **Thực nghiệm & Phân tích (Slide 7 $\to$ Slide 10):** ~8 phút.
- **Hạn chế & Kết luận (Slide 11 $\to$ Q&A):** ~3 – 5 phút.
- **Yêu cầu văn phong:** Rõ ràng, trực diện, khoa học, trung tính. Sử dụng chính xác thuật ngữ kỹ thuật (*stop-gradient, BPTT, stateful denoiser, spurious attractor, decreasing noise, shared noise, Euler step, ARC-AGI, pass@2*).

---

## 📑 Chi Tiết Kịch Bản Từng Slide

### Trang Bìa & Mục Lục
- **Slide trong TeX:** `main.tex` (Trang bìa & Mục lục)
- **Thời lượng:** ~1 phút
- **Lời thoại:**
  > "Kính thưa quý thầy cô và các bạn, hôm nay em xin trình bày bài báo nghiên cứu: **'Thinking with Looped Flows: Huấn luyện Mô hình Tuần hoàn qua Mục tiêu Khử nhiễu Cục bộ'**, công bố trên arXiv vào tháng 9 năm 2026 bởi nhóm tác giả đến từ EPFL, KAIST, CMU và Đại học Amsterdam.
  >
  > Nội dung bài thuyết trình gồm 4 phần:
  > 1. Bối cảnh nghiên cứu và các hạn chế của mô hình lặp hiện nay.
  > 2. Phương pháp Looped Flows: Bộ khử nhiễu có trạng thái và cơ chế Căn chỉnh theo thời gian.
  > 3. Cơ chế suy luận và kết quả thực nghiệm trên các tác vụ suy luận.
  > 4. Nghiên cứu loại bỏ thành phần (Ablation), hạn chế của mô hình và kết luận."

---

### Slide 0: Looped Flows: Tổng quan bài báo
- **Slide trong TeX:** `lf_paper.tex` (Slide 0)
- **Thời lượng:** ~1.5 phút
- **Mục tiêu:** Tóm tắt thông tin bài báo, ý tưởng kỹ thuật chính và kết quả thực nghiệm nổi bật.
- **Lời thoại:**
  > "Về tổng quan, bài báo đề xuất một phương pháp: **huấn luyện mô hình suy luận vòng lặp (Looped Model) thông qua một chuỗi các mục tiêu khử nhiễu cục bộ (Local Denoising Objectives)** của Flow Matching.
  >
  > Khi huấn luyện mô hình vòng lặp, việc lan truyền ngược qua toàn bộ chuỗi lặp (full BPTT) đòi hỏi nhiều bộ nhớ và dễ gây mất ổn định gradient. Ngược lại, nếu dùng chặn gradient (stop-gradient), các bước sau không truyền được tín hiệu để tối ưu trạng thái ẩn cho các bước trước.
  >
  > Looped Flows xử lý vấn đề này qua ba điểm chính:
  > 1. **Bộ khử nhiễu có trạng thái (Stateful Denoiser):** Vừa thực hiện khử nhiễu dòng, vừa cập nhật trạng thái ẩn phục vụ tính toán tuần tự.
  > 2. **Suy luận bằng tích phân dòng xác suất:** Cho phép tăng lượng tính toán lúc suy luận bằng cách làm mịn lưới thời gian, đồng thời tạo ra nhiều lời giải hợp lệ từ các mẫu nhiễu ban đầu khác nhau.
  > 3. **Kết quả thực nghiệm:** Mô hình có kích thước **5 đến 7 triệu tham số**, đạt **58.8%** trên ARC-AGI-1 (so với TRM là 44.6%) và đạt **97.9%** trên bài toán Sudoku-Extreme."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào tên tác giả và tổ chức]`: Nêu rõ nguồn bài báo (arXiv:2609.11801).
  - `[Chỉ vào các số liệu 5-7M, 58.8%, 97.9%]`: Nêu rõ quy mô tham số và kết quả đạt được.
- **Câu hỏi phản biện dự kiến (Q&A):**
  - *Hỏi:* Vì sao bài báo đánh giá trên mô hình 5-7M tham số thay vì mô hình lớn?
  - *Đáp:* Tác giả kiểm chứng cơ chế suy luận lặp trên các bài toán thuật toán và suy luận logic (ARC, Sudoku). Mô hình nhỏ giúp kiểm soát trực tiếp lượng tính toán nội tại, loại trừ ảnh hưởng của việc ghi nhớ dữ liệu lớn, đồng thời giảm thời gian huấn luyện.

---

### Slide 1: 1. Looped Flows: Bối cảnh và thách thức
- **Slide trong TeX:** `lf_paper.tex` (Slide 1)
- **Thời lượng:** ~2 phút
- **Mục tiêu:** Phân tích bối cảnh bài toán và 3 hạn chế chính của các hướng tiếp cận hiện tại.
- **Lời thoại:**
  > "Về bối cảnh, các bài toán suy luận phức tạp đòi hỏi nhiều bước tính toán hơn tại thời điểm suy luận. Mô hình vòng lặp giải quyết yêu cầu này bằng cách cập nhật trạng thái ẩn lặp đi lặp lại để tích lũy tính toán.
  >
  > Tác giả chỉ ra ba khoảng trống của các hướng nghiên cứu hiện có:
  > 1. **Mô hình vòng lặp như HRM, TRM, FPRM:** Lan truyền ngược qua toàn bộ chuỗi lặp tốn bộ nhớ và không ổn định, nên phải áp dụng toán tử chặn gradient `sg`. Khi đó, hàm mất mát ở các bước sau không truyền được tín hiệu học cho các trạng thái ẩn ở bước trước ($z_{<i}$).
  > 2. **Mô hình dòng và khuếch tán (Flow Matching, Diffusion):** Học qua các mục tiêu khử nhiễu cục bộ độc lập tại từng thời điểm $t$, không cần lan truyền ngược qua quỹ đạo. Tuy nhiên, mô hình không duy trì trạng thái lặp (stateless), do đó khả năng thực hiện tính toán tuần tự bị hạn chế.
  > 3. **Mô hình vòng lặp ngẫu nhiên như PTRM, EqR, GRAM:** Thêm tính ngẫu nhiên nhưng vẫn dựa trên gradient cục bộ để xây dựng trạng thái lặp.
  >
  > **Hậu quả:** Chuỗi lặp dễ bị mất ổn định, rơi vào các điểm hút giả (spurious attractors) — tức là trạng thái bị kẹt ở nghiệm sai — hoặc không hội tụ. Giải pháp của bài báo là kết hợp mục tiêu khử nhiễu cục bộ với trạng thái lặp."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào khái niệm Spurious Attractor]`: Giải thích đây là hiện tượng chuỗi lặp bị kẹt vào nghiệm sai cố định.
  - `[Chỉ vào 3 gạch đầu dòng]`: Nêu rõ sự khác biệt giữa mô hình lặp và mô hình dòng.

---

### Slide 2: 2. Looped Flows: Kiến thức nền
- **Slide trong TeX:** `lf_paper.tex` (Slide 2)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Trình bày công thức toán học của Looped Models và Categorical Flow Matching, chỉ rõ ưu và nhược điểm của từng phương pháp.
- **Lời thoại:**
  > "Về mặt toán học, phương pháp dựa trên hai nền tảng:
  >
  > **1. Mô hình vòng lặp:**
  > Trạng thái ẩn được cập nhật qua: $z_i = f(z_{i-1}; c)$, và đầu đọc giải mã dự đoán: $\hat{x}_i = g(z_i)$.
  > Khi huấn luyện với chặn gradient `sg`:
  > $$\mathcal{L}_{\text{loop}} = \mathbb{E}_{c,x}\Big[\sum_i \text{CE}(\hat{x}_i, x)\Big], \qquad z_i = f\big(\text{sg}(z_{i-1}); c\big)$$
  > Gradient chỉ truyền qua bước $i$ hiện tại, không truyền về $z_{<i}$. Điều này giới hạn khả năng học trạng thái tối ưu cho các bước sau. Ngoài ra, do hàm $f$ xác định, mô hình chỉ tạo ra một lời giải duy nhất cho mỗi đầu vào.
  >
  > **2. Khớp dòng cho dữ liệu phân loại (Categorical Flow Matching):**
  > Với nhãn $x_1$ và nhiễu Gaussian $x_0 \sim \mathcal{N}(0, \sigma^2 I)$, phép nội suy tuyến tính được định nghĩa:
  > $$I_t = (1 - t)x_0 + t x_1, \quad t \in [0, 1]$$
  > Bộ khử nhiễu $D_t(x; c) = \mathbb{E}[x_1 \mid I_t = x, c]$ dự đoán dữ liệu sạch, và trường vận tốc tương ứng là $b_t(x; c) = \frac{D_t(x; c) - x}{1 - t}$.
  > Hàm mục tiêu:
  > $$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t \sim \mathcal{U}[0, 1]}\big[\text{CE}(\hat{x}_t, x_1)\big]$$
  > Điểm thuận lợi là mất mát tại thời điểm $t$ độc lập với các thời điểm khác, không cần lan truyền ngược qua quỹ đạo. Tuy nhiên, mô hình không có trạng thái ẩn để lưu trữ kết quả tính toán trung gian."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào toán tử sg]`: Nêu rõ đây là stop-gradient.
  - `[Chỉ vào công thức nội suy I_t]`: Nêu rõ giá trị tại $t=0$ là nhiễu, tại $t=1$ là nhãn $x_1$.
- **Câu hỏi phản biện dự kiến (Q&A):**
  - *Hỏi:* Tại sao dữ liệu phân loại rời rạc lại áp dụng nội suy liên tục $I_t$?
  - *Đáp:* Token được ánh xạ sang không gian vector liên tục (continuous representation). Mạng dự đoán phân phối xác suất và tính Cross-Entropy với nhãn $x_1$. Cách làm này cho phép sử dụng các thuật toán giải ODE liên tục trong không gian thực.

---

### Slide 3: 3. Looped Flows: Bộ khử nhiễu có trạng thái
- **Slide trong TeX:** `lf_paper.tex` (Slide 3)
- **Thời lượng:** ~2 phút
- **Mục tiêu:** Trình bày cấu trúc toán học của Stateful Denoiser $\hat{D}_{t_i}$, công thức Euler và cấu hình tham số mạng.
- **Lời thoại:**
  > "Ý tưởng kỹ thuật của bài báo là xây dựng bộ khử nhiễu có trạng thái, ký hiệu là $\hat{D}_{t_i}$. Mạng nhận trạng thái dòng $x_{t_i}$ và trạng thái lặp $z_{t_i}$, đồng thời xuất ra hai giá trị:
  > $$\big(\hat{x}_{t_i},\, z_{t_{i+1}}\big) = \hat{D}_{t_i}\big(x_{t_i},\, z_{t_i};\, c\big)$$
  > Trong đó $\hat{x}_{t_i}$ là ước lượng nhãn sạch, còn $z_{t_{i+1}}$ là trạng thái lặp được cập nhật cho bước kế tiếp.
  >
  > Quá trình suy luận sử dụng bước Euler tiến trên lưới thời gian $0 = t_0 < t_1 < \dots < t_n = 1$:
  > $$x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_i)\,\frac{\hat{x}_{t_i} - x_{t_i}}{1 - t_i}$$
  >
  > **Về kiến trúc:**
  > Mô hình phát triển dựa trên kiến trúc TRM, bổ sung lớp chiếu cho vector nội suy nhiễu và vector nhúng thời gian $t_i$. Thay đổi này làm tăng khoảng **5% tham số** so với TRM.
  > Trạng thái $z$ gồm hai thành phần: đặc trưng $h$ để giải mã dự đoán và thành phần $\ell$ để cập nhật $h$. Quá trình huấn luyện chỉ lan truyền ngược gradient trong chu kỳ cuối của mỗi lần gọi mạng.
  >
  > **Quy mô:**
  > - Dùng backbone **MLP-Mixer** với **5 triệu tham số** cho Sudoku.
  > - Dùng backbone **Transformer** với **7 triệu tham số** cho các bài toán còn lại."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào phương trình cập nhật cặp]`: Giải thích việc ghép nối giữa biến dòng $x$ và biến trạng thái $z$.
  - `[Chỉ vào công thức Euler]`: Nêu rõ quy tắc dịch chuyển trạng thái $x$ theo vận tốc ước lượng.

---

### Slide 4: 4. Looped Flows: Căn chỉnh theo thời gian
- **Slide trong TeX:** `lf_paper.tex` (Slide 4)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Trình bày hàm mục tiêu $\mathcal{L}_{\text{LF}}$, hai thiết kế then chốt (Nhiễu giảm dần & Nhiễu dùng chung), và kỹ thuật Mục tiêu giả (pseudo-targets).
- **Lời thoại:**
  > "Để huấn luyện Stateful Denoiser khi gradient bị chặn ở từng bước, tác giả sử dụng cơ chế **Căn chỉnh theo thời gian (Temporal Alignment)**. Hàm mục tiêu được xác định bởi:
  > $$\mathcal{L}_{\text{LF}} = \mathbb{E}\Big[\sum_{i=0}^{k-1} \text{CE}(\hat{x}_{t_i}, x_1)\Big], \quad \text{với } (\hat{x}_{t_i}, z_{t_{i+1}}) = \hat{D}_{t_i}\big(I_{t_i}, \text{sg}(z_{t_i}); c\big)$$
  > Lấy $k+1$ mốc thời gian tăng dần $0 \le t_0 < t_1 < \dots < t_k \le 1$.
  >
  > Cơ chế này hoạt động dựa trên hai thiết kế:
  > 1. **Nhiễu giảm dần (Decreasing Noise Levels):** Do $t_i$ tăng dần, mức nhiễu $(1 - t_i)$ giảm dần qua từng bước. Quá trình tính toán trở thành chuỗi tác vụ khử nhiễu từ mức khó đến mức dễ.
  > 2. **Nhiễu dùng chung (Shared Noise):** Trong mỗi chuỗi lặp, cùng một cặp $(x_0, x_1)$ được giữ cố định cho mọi $t_i$. Vì vậy, đầu vào ở hai bước kế tiếp có khoảng cách gần nhau. Trạng thái ẩn $z_{t_i}$ từ bước trước chứa thông tin phù hợp cho bước tiếp theo. Nhờ đó, dù gradient bị chặn ở mỗi bước, mô hình vẫn học được trạng thái lặp liên tục.
  >
  > **Kỹ thuật Mục tiêu giả (Pseudo-targets):**
  > Với các tập dữ liệu nhỏ (ví dụ 1.000 mẫu Sudoku), ở các bước $t$ lớn, $I_t$ chứa phần lớn thông tin của nhãn $x_1$, dễ dẫn đến hiện tượng quá khớp (overfitting). Tác giả thay thế $x_1$ trong biểu thức nội suy bằng dự đoán của bước trước $\text{sg}(\hat{x}_{t_{i-1}})$. Xác suất áp dụng kỹ thuật này tăng dần từ 0 đến 1 trong 20.000 bước huấn luyện đầu tiên. Hàm mất mát vẫn được tính dựa trên nhãn $x_1$ gốc."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào Shared Noise]`: Nhấn mạnh việc giữ nguyên $(x_0, x_1)$ trong một rollout là yếu tố giúp liên kết thông tin giữa các bước lặp.
  - `[Chỉ vào Pseudo-targets]`: Nêu rõ mục đích là giảm nguy cơ học vẹt khi dữ liệu ít.

---

### Slide 5: 5. Looped Flows: Thời gian tính toán thích ứng và cấu hình
- **Slide trong TeX:** `lf_paper.tex` (Slide 5)
- **Thời lượng:** ~2 phút
- **Mục tiêu:** Trình bày cơ chế Adaptive Computation Time (ACT) với đầu dừng $q$, tối ưu hóa Adam-atan2 và chi phí tính toán.
- **Lời thoại:**
  > "Do mức nhiễu giảm dần, hàm mất mát khử nhiễu có xu hướng giảm nhanh và bão hòa ở các bước cuối. Khi dự đoán $\hat{x}_{t_i}$ đã chính xác, việc tiếp tục tính toán không còn cung cấp tín hiệu gradient hữu ích.
  >
  > Tác giả bổ sung một **Đầu dừng (Halting Head) $q$** nhị phân để phân loại xem dự đoán hiện tại đã đúng với nhãn $x_1$ hay chưa:
  > $$\mathcal{L}_{\text{ACT}} = \mathbb{E}\Big[\sum_{i} \lambda\,\text{BCE}\big(q(z_{t_{i+1}}),\, \mathbb{1}(x_1 = \text{round}(\hat{x}_{t_i}))\big)\Big]$$
  > với hệ số $\lambda = 0.5$.
  > Trong quá trình huấn luyện, khi $q > 0.5$, mô hình ngắt sớm chuỗi tính toán và bỏ qua các bước còn lại. Hàm mất mát tổng hợp là $\mathcal{L} = \mathcal{L}_{\text{LF}} + \mathcal{L}_{\text{ACT}}$.
  >
  > **Cấu hình huấn luyện:**
  > - Huấn luyện qua $k = 16$ bước thời gian.
  > - Sử dụng bộ tối ưu **Adam-atan2** với learning rate $10^{-4}$, batch size 768, hệ số EMA 0.999 và cắt gradient ở ngưỡng 1.0.
  > - **Thời gian huấn luyện:** Mất khoảng **1 đến 5 giờ** đối với Sudoku, Mê cung, N-Queens, Tô màu đồ thị; và khoảng **1 đến 2 ngày** đối với ARC-AGI."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào điều kiện q > 0.5]`: Nêu rõ cơ chế ngắt sớm giúp tiết kiệm tính toán lúc huấn luyện.
  - `[Chỉ vào thông số thời gian]`: Nêu rõ mức tài nguyên thực tế cần sử dụng.

---

### Slide 6: 6. Looped Flows: Cơ chế suy luận
- **Slide trong TeX:** `lf_paper.tex` (Slide 6)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Trình bày 2 phương thức suy luận (ODE xác định vs SDE ngẫu nhiên qua lùi nhiễu), cách mở rộng tính toán lúc suy luận và khả năng sinh đa nghiệm.
- **Lời thoại:**
  > "Tại thời điểm suy luận, mô hình hỗ trợ hai phương thức tích phân:
  >
  > 1. **Tích phân xác định bằng ODE:** Khởi tạo từ nhiễu $x_0$, áp dụng bước Euler tiến từ $t=0$ đến $t=1$, cập nhật đồng thời trạng thái dòng $x$ và trạng thái ẩn $z$.
  > 2. **Tích phân ngẫu nhiên bằng SDE qua cơ chế Lùi nhiễu (Noise Rollback):**
  >    Tại bước $t_i$, mô hình đưa trạng thái $x_{t_i}$ lùi về thời điểm $s = a \cdot t_i$, bổ sung một lượng nhiễu Gaussian mới, sau đó mới thực hiện bước Euler tiến tới $t_{i+1}$. Khi hệ số điều khiển $\gamma = 0$, quá trình quay về ODE xác định; khi $\gamma > 0$, tính ngẫu nhiên được bổ sung để hỗ trợ mô hình thoát khỏi các cực tiểu địa phương.
  >
  > **Mở rộng tính toán lúc suy luận (Inference-time Scaling):**
  > - **Làm mịn lưới thời gian:** Dù chỉ huấn luyện với 16 bước, lúc suy luận có thể tăng số bước Euler lên 32, 64 hoặc **128 bước**, giúp giảm sai số xấp xỉ số học và cải thiện độ chính xác.
  > - **Tổ hợp mẫu theo điểm Q cao nhất (Best-of-N):** Chạy song song 5 quỹ đạo độc lập từ các mẫu nhiễu khác nhau, sử dụng xác suất từ đầu dừng $q$ để chọn ra dự đoán có độ tin cậy cao nhất.
  > - **Sinh nhiều lời giải (Multi-solution):** Việc lấy mẫu từ các vector nhiễu $x_0$ khác nhau cho phép dòng xác suất hội tụ về các nghiệm khác nhau của bài toán, khắc phục tính chất đơn nghiệm của mô hình lặp xác định."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào cơ chế Noise Rollback]`: Giải thích việc lùi thời gian và thêm nhiễu nhằm tăng độ phân tán của quỹ đạo.
  - `[Chỉ vào số bước 16 -> 128]`: Nêu rõ khả năng mở rộng số bước suy luận mà không cần huấn luyện lại.

---

### Slide 7: 7. Looped Flows: Kết quả trên các bài toán suy luận
- **Slide trong TeX:** `lf_paper.tex` (Slide 7)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Phân tích bảng số liệu thực nghiệm trên 4 benchmark (Sudoku, Maze, ARC-1, ARC-2) so với các baseline (HRM, TRM, FPRM, GRAM).
- **Lời thoại:**
  > "Bảng trên slide trình bày kết quả đánh giá trên 4 benchmark tiêu chuẩn so với các mô hình đối chứng:
  >
  > 1. **Sudoku-Extreme:** Looped Flows đạt **97.9%**, cao hơn HRM 27M (55.0%), TRM (87.4%) và FPRM (94.2%).
  > 2. **ARC-AGI-1:** Mô hình đạt **58.8%**, cải thiện so với TRM (44.6%) và cao hơn GRAM (52.0%).
  > 3. **ARC-AGI-2:** Mô hình đạt **12.2%**, cao hơn TRM (7.8%) và GRAM (11.1%).
  > 4. **Maze-Hard (Mê cung):** Looped Flows đạt 86.7%, tương đương TRM (85.3%) và thấp hơn nhẹ so với FPRM (87.0%). Bài báo chưa phân tích nguyên nhân cụ thể cho việc kết quả trên Maze không tạo ra khoảng cách lớn.
  >
  > Khi áp dụng tổ hợp 5 quỹ đạo theo điểm Q:
  > - Sudoku tăng lên **99.3%**.
  > - ARC-AGI-1 đạt **59.5%**.
  > Mức kết quả này tương đương với PTRM (dùng 100 quỹ đạo) và EqR (dùng 128 quỹ đạo).
  >
  > Một điểm cần lưu ý: độ lệch chuẩn trên ARC dao động khoảng $\pm 1.8$ đến $\pm 1.9$ điểm phần trăm qua 3 hạt giống, cho thấy kết quả có độ nhạy với việc khởi tạo ngẫu nhiên."
- **Ghi chú thao tác (Cues):**
  - `[Khoanh vùng hàng Looped Flows và cột ARC-1]`: Nhấn mạnh mức chênh lệch 58.8% so với 44.6% của TRM khi dùng chung số tham số.
  - `[Chỉ vào cột Maze-Hard]`: Nêu rõ kết quả tương đương FPRM.

---

### Slide 8: 8. Looped Flows: Mở rộng lúc suy luận và độ ổn định
- **Slide trong TeX:** `lf_paper.tex` (Slide 8)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Trình bày định lượng về việc mở rộng số bước suy luận (8 $\to$ 128) và phân tích nguyên nhân ổn định hóa chuỗi lặp trên 65.000 bài Sudoku.
- **Lời thoại:**
  > "Slide này phân tích hai đặc tính thực nghiệm quan trọng:
  >
  > **Thứ nhất, ảnh hưởng của số bước tích phân trên Sudoku-Extreme:**
  > Khi tăng số bước Euler từ 8 lên 128:
  > - 8 bước đạt 74.5%.
  > - 16 bước (bằng số bước huấn luyện) đạt 85.2%.
  > - 32 bước đạt 93.0%.
  > - 64 bước đạt 96.7%.
  > - 128 bước đạt **97.9%**.
  > Độ chính xác tăng đều theo số bước, cho thấy mô hình học được trường vận tốc phù hợp để tinh chỉnh nghiệm qua các bước nhỏ hơn.
  >
  > **Thứ hai, độ ổn định của chuỗi lặp trên 65.000 bài Sudoku-Extreme:**
  > Mô hình TRM thất bại ở **12.6%** tổng số bài. Trong số các bài thất bại:
  > - **88.3%** do chuỗi lặp không hội tụ.
  > - **11.7%** do rơi vào điểm hút giả (spurious attractor).
  >
  > Khi cho Looped Flows xử lý tập các bài toán TRM thất bại:
  > - Mô hình giải quyết thành công **90.9%** số bài đó.
  > - Cụ thể, giải quyết được **89.9%** trường hợp không hội tụ và **98.0%** trường hợp bị kẹt vào điểm hút giả.
  > Kết quả này chỉ ra rằng việc kết hợp dòng xác suất và nhiễu ngẫu nhiên giúp chuỗi lặp tránh được việc bị kẹt tại các nghiệm sai cục bộ."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào bảng số bước 8 đến 128]`: Nêu rõ mức tăng độ chính xác theo số bước tính toán lúc suy luận.
  - `[Chỉ vào số liệu giải cứu 90.9%]`: Giải thích cơ chế giải quyết điểm hút giả của TRM.

---

### Slide 9: 9. Looped Flows: Bài toán đa lời giải
- **Slide trong TeX:** `lf_paper.tex` (Slide 9)
- **Thời lượng:** ~2 phút
- **Mục tiêu:** Trình bày kết quả trên bài toán đa nghiệm (N-Queens và Tô màu đồ thị), so sánh trực tiếp với GRAM trên hai chỉ số Accuracy/Conflicts và Coverage.
- **Lời thoại:**
  > "Trên các bài toán có nhiều nghiệm hợp lệ, mô hình được so sánh trực tiếp với GRAM — mô hình lặp có khả năng sinh ngẫu nhiên:
  >
  > Hai chỉ số đánh giá gồm:
  > - **Độ chính xác hoặc Số xung đột ($\downarrow$):** Đo lường mức độ tuân thủ luật của lời giải.
  > - **Độ phủ (Coverage):** Tỷ lệ phần trăm các nghiệm phân biệt đúng được tìm thấy trong tập nghiệm thực tế.
  >
  > **Kết quả:**
  > - **Tô màu đồ thị:** Looped Flows cho kết quả tốt hơn GRAM: với đồ thị 8 đỉnh, số xung đột giảm từ 2.7 xuống **0.7**, độ phủ tăng từ 85.8% lên **89.4%**. Với đồ thị 10 đỉnh, số xung đột giảm từ 3.3 xuống **1.0**, độ phủ đạt **55.2%** (so với GRAM là 51.3%).
  > - **N-Queens 8x8:** Looped Flows đạt độ chính xác **99.9%** và độ phủ **91.4%**, cao hơn GRAM (99.7% và 90.3%).
  > - **N-Queens 10x10:** GRAM đạt kết quả tốt hơn ở cả độ chính xác (96.1% so với 94.4%) và độ phủ (74.3% so với 61.5%).
  >
  > Nhìn chung, Looped Flows đạt kết quả tốt hơn ở 6 trong 8 chỉ số đánh giá trên hai bài toán tổ hợp này."
- **Ghi chú thao tác (Cues):**
  - `[Giải thích chỉ số Độ phủ - Coverage]`: Nêu rõ đây là tỷ lệ số nghiệm đúng tìm được trên tổng số nghiệm lý thuyết.
  - `[Chỉ vào kết quả N-Queens 10x10]`: Nêu rõ trường hợp GRAM dẫn trước.

---

### Slide 10: 10. Looped Flows: Nghiên cứu loại bỏ thành phần
- **Slide trong TeX:** `lf_paper.tex` (Slide 10)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Phân tích mức độ đóng góp của từng thành phần kỹ thuật qua bảng Ablation trên ARC-AGI-1/2 và so sánh SDE vs ODE.
- **Lời thoại:**
  > "Bảng nghiên cứu loại bỏ thành phần trên ARC-AGI làm rõ vai trò của từng cải tiến thiết kế:
  >
  > - Mô hình **Đầy đủ** đạt **58.8%** trên ARC-1 và **12.2%** trên ARC-2.
  > - **Bỏ điều kiện thời gian:** Kết quả giảm xuống 56.4% / 9.9%.
  > - **Bỏ nội suy dòng:** Giảm xuống **51.5%** / 9.9%.
  > - **Bỏ cả hai yếu tố trên:** Độ chính xác giảm xuống **43.6%** / 5.0%, tương đương với mức của mô hình TRM ban đầu (44.6%).
  > - **Bỏ cơ chế nhiễu giảm dần:** Độ chính xác giảm xuống **51.6%** trên ARC-1, cho thấy đây là thành phần có tác động lớn nhất đến hiệu năng.
  > - **Bỏ nhiễu dùng chung:** Độ chính xác giảm nhẹ xuống 56.4% / 10.8%.
  >
  > **Về phương thức suy luận:**
  > Phương thức SDE với lùi nhiễu ($\gamma = 5$) cho kết quả cao hơn ODE xác định một cách nhất quán (Sudoku đạt **97.9%** so với 97.6%; ARC-1 đạt **58.8%** so với 57.5%).
  > Trên bài toán Sudoku, mô hình dòng không có vòng lặp có xu hướng ghi nhớ dữ liệu huấn luyện và giảm khả năng tổng quát hóa; việc kết hợp cấu trúc lặp giúp cải thiện độ chính xác trên dữ liệu kiểm tra."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào dòng Bỏ cả hai]`: Nhấn mạnh khi loại bỏ thành phần dòng, kết quả tương đương TRM (43.6% so với 44.6%).
  - `[Chỉ vào dòng Bỏ nhiễu giảm dần]`: Nêu rõ mức suy giảm từ 58.8% xuống 51.6%.

---

### Slide 11: 11. Looped Flows: Hạn chế, phản biện và kết luận
- **Slide trong TeX:** `lf_paper.tex` (Slide 11) & `main.tex` (Slide kết thúc & Q&A)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Trình bày 4 hạn chế kỹ thuật của bài báo, tóm tắt kết luận và chuyển sang phần Q&A.
- **Lời thoại:**
  > "Về mặt học thuật, bài báo có **bốn hạn chế kỹ thuật cần lưu ý**:
  >
  > 1. **Chưa có chứng minh lý thuyết đầy đủ cho việc căn chỉnh thời gian:** Việc dùng chung nhiễu $x_0$ về nguyên tắc có thể tạo đường tắt triệt tiêu nhiễu; bài báo mới kiểm chứng qua thực nghiệm gián tiếp chứ chưa có chứng minh giải tích toán học.
  > 2. **Lệch phân phối (Distribution Shift):** Có sự khác biệt giữa dữ liệu nội suy $I_t$ dùng trong huấn luyện và trạng thái dòng $x_t$ được tích phân từng bước lúc suy luận thực tế.
  > 3. **Phạm vi thực nghiệm còn hẹp:** Mô hình mới được thử nghiệm ở quy mô 5 đến 7 triệu tham số trên các bài toán câu đố suy luận, chưa áp dụng cho các mô hình ngôn ngữ lớn (LLM) hoặc bài toán sinh chuỗi văn bản tự do.
  > 4. **Quá trình huấn luyện chưa phi mô phỏng (not simulation-free):** Vẫn cần thực hiện tuần tự qua $k$ bước thời gian để cập nhật trạng thái $z$, chưa thể huấn luyện song song hoàn toàn độc lập như Flow Matching chuẩn.
  >
  > **Kết luận:**
  > - Việc kết hợp mục tiêu khử nhiễu cục bộ với trạng thái lặp giúp ổn định quá trình huấn luyện chuỗi lặp khi gradient bị chặn.
  > - Mô hình cho phép mở rộng lượng tính toán tại thời điểm suy luận qua lưới Euler mịn hơn và hỗ trợ tìm nhiều nghiệm hợp lệ.
  > - Hướng nghiên cứu tiếp theo bao gồm xây dựng thuật toán huấn luyện không cần mô phỏng và mở rộng quy mô lên các bài toán lớn hơn.
  >
  > Em xin cảm ơn quý thầy cô và các bạn đã theo dõi. Em xin sẵn sàng nhận các câu hỏi cho phần thảo luận."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào 4 điểm hạn chế]`: Trình bày điềm tĩnh, khách quan.
  - `[Chuyển sang slide Q&A]`: Kết thúc bài báo cáo rõ ràng, sẵn sàng trả lời câu hỏi.
- **Câu hỏi phản biện dự kiến (Q&A):**
  - *Hỏi:* Làm thế nào để áp dụng phương pháp Looped Flows cho LLM sinh văn bản?
  - *Đáp:* Có thể tích hợp phương pháp này vào không gian biểu diễn ẩn (Latent Space) của LLM. Mô hình dùng các vòng lặp để khử nhiễu và tinh chỉnh một vector biểu diễn ý định hoặc kế hoạch suy luận liên tục, sau đó dùng decoder giải mã vector này ra chuỗi token văn bản.
