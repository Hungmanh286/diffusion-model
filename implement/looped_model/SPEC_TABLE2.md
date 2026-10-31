# SPEC: Tái hiện Table 2 — *Thinking with Looped Flows* (arXiv:2609.11801)

Phạm vi: N-Queens (8×8, 10×10) và Graph Coloring (8, 10 đỉnh). Chưa có code ở bước này.

**Quy ước nguồn.** Nguồn đọc từ `papers/2609.11801.pdf` (bản v1, 19 trang). Số trang trong bài trùng với số trang PDF.
- `[§5.2, tr.8]` nghĩa là section 5.2, trang 8.
- `[PDF]` nghĩa là giá trị đã được đối chiếu trực tiếp với ảnh trang, không chỉ với văn bản trích xuất.
- **KHÔNG NÊU** nghĩa là bài không nói. Mục đó được đưa vào Mục 6 (mơ hồ) và Mục 7 (câu hỏi cho bài GRAM). Không tự đoán.
- Dấu `(*)` đánh dấu phép tính hoặc nhận xét của người viết spec, không phải khẳng định của bài.

Bài GRAM = Baek et al., 2026, arXiv:2605.19376. Bài tham chiếu nó cho phần dựng dữ liệu và cho các số baseline [§5.2, tr.8; Table 2 caption, tr.9; App. E.1, tr.16]. Bài này chưa có trong `papers/`.

---

## 1. Mục tiêu

### 1.1 Cấu trúc bảng [Table 2, tr.9, PDF]

- **Hàng**: 9 phương pháp (8 baseline và Looped flows), kèm số tham số.
- **N-Queens**, mỗi cỡ bàn có 2 cột: 8×8 và 10×10.
  - Cột `Accuracy ↑` (%) và `Coverage ↑` (%).
- **Graph Coloring**, mỗi cỡ đồ thị có 2 cột: 8 đỉnh và 10 đỉnh.
  - Cột `Conflicts ↓` (số tuyệt đối, tổng trên các đồ thị test) và `Coverage ↑` (%).
- Caption [tr.9]: *"We report solution accuracy (%) for N-Queens and constraint violations for Graph Coloring, with coverage (%) of distinct valid solutions from 20 inferences. Baseline results are from Baek et al. (2026). For looped flows, we run three repeated tests."*

### 1.2 Metric theo bài toán

| Bài toán | Metric chính | Coverage | Ghi chú nguồn |
|---|---|---|---|
| N-Queens | **Accuracy**: tính trên **mẫu đầu tiên** trong 20 lần suy luận | Có | §5.2, tr.8: *"N-Queens accuracy measures the validity of the first sample"* |
| Graph Coloring | **Conflicts**: chọn **tô màu hoàn chỉnh xuất hiện nhiều nhất** cho mỗi đồ thị test, rồi **cộng số xung đột trên toàn bộ đồ thị test** | Có | §5.2, tr.8 |

Coverage dùng chung công thức cho cả hai bài toán, xem Mục 5. Không có Accuracy cho Graph Coloring trong Table 2.

### 1.3 Giá trị mục tiêu của Looped flows (mean ± std, 3 lần chạy) [Table 2, tr.9, PDF]

| Params | NQ 8×8 Acc | NQ 8×8 Cov | NQ 10×10 Acc | NQ 10×10 Cov | GC-8 Conf | GC-8 Cov | GC-10 Conf | GC-10 Cov |
|---|---|---|---|---|---|---|---|---|
| 7M | 99.9±0.1 | 91.4±0.4 | 94.4±0.7 | 61.5±0.1 | 0.7±0.6 | 89.4±0.7 | 1.0±1.0 | 55.2±0.3 |

Bài khẳng định Looped flows đạt kết quả tốt nhất trên mọi tác vụ và mọi metric [§5.2, tr.8].

### 1.4 Giá trị baseline [Table 2, tr.9, PDF]

Các số này lấy từ Baek et al. (2026), không do bài này chạy lại [Table 2 caption, tr.9; §5.2, tr.8].

| Phương pháp | Params | NQ8 Acc | NQ8 Cov | NQ10 Acc | NQ10 Cov | GC8 Conf | GC8 Cov | GC10 Conf | GC10 Cov |
|---|---|---|---|---|---|---|---|---|---|
| Direct prediction (8 layers) | 27M | 40.4±1.1 | 13.7±1.1 | 13.6±0.5 | 1.6±0.2 | 179.3±4.0 | 19.9±0.2 | 198.7±5.0 | 6.7±0.1 |
| Direct prediction (32 layers) | 100M | 40.2±1.3 | 13.6±1.1 | 13.1±0.4 | 1.6±0.2 | 174.0±18.0 | 19.1±1.7 | 227.7±34.5 | 6.5±1.9 |
| Looped transformer | 7M | 68.4±3.7 | 23.6±1.9 | 50.0±7.6 | 6.2±3.2 | 136.0±16.1 | 20.5±1.5 | 157.3±9.0 | 7.2±0.7 |
| HRM | 27M | 78.7±2.9 | 26.7±1.3 | 37.4±0.3 | 4.7±0.1 | 109.7±1.5 | 21.8±0.3 | 164.3±21.6 | 8.9±1.7 |
| TRM | 7M | 66.8±5.7 | 36.1±22.5 | 17.5±11.2 | 2.0±1.3 | 109.3±3.1 | 22.3±0.6 | 170.7±17.9 | 6.8±0.3 |
| AR transformer | 10.6M | 96.3±1.0 | 84.8±0.8 | 90.0±2.2 | 53.2±0.8 | 19.0±11.3 | 83.0±0.7 | 61.3±8.3 | 40.0±0.3 |
| MDLM | 12.6M | 96.1±1.5 | 87.2±0.6 | 74.3±6.6 | 47.4±2.2 | 2.7±0.6 | 84.5±4.0 | 12.0±7.0 | 48.2±1.4 |
| GRAM | 10M | 99.7±0.3 | 90.3±1.9 | 89.7±2.7 | 57.5±3.4 | 2.7±2.1 | 85.8±0.5 | 3.3±1.5 | 51.3±2.8 |

### 1.5 Số liên quan có thể dùng để kiểm tra chéo (ODE so với SDE)

- **Table 5** [tr.9, PDF], coverage (%), ODE γ=0 so với SDE γ=5:
  - NQ8: 91.4±0.4 so với 91.4±0.4
  - NQ10: 54.7±0.6 so với 61.5±0.1
  - GC8: 88.4±0.7 so với 89.4±0.7
  - GC10: 54.0±1.0 so với 55.2±0.3
- **Table 7** [tr.17]:
  - N-Queens accuracy, ODE so với SDE: 8×8 là 98.4±0.3 so với 99.9±0.1; 10×10 là 73.2±1.4 so với 94.4±0.7.
  - GC conflicts, ODE so với SDE: 8 đỉnh là 0.3±0.6 so với 0.7±0.6; 10 đỉnh là 2.7±1.2 so với 1.0±1.0.
- Cột SDE của Table 5 và Table 7 khớp Table 2 (cùng γ=5).

---

## 2. Dữ liệu

Phần lớn mục này **KHÔNG NÊU** trong bài. Bài chỉ nói *"for the multi-solution tasks, we follow the construction of GRAM"* [App. E.1, tr.16], và mô tả ngắn ở §5.2. Mọi chi tiết còn lại phải lấy từ bài GRAM.

| Hạng mục | Nội dung trong bài | Nguồn |
|---|---|---|
| Quy mô train | **KHÔNG NÊU** | |
| Quy mô test | **KHÔNG NÊU** | |
| Tập validation / chọn checkpoint | **KHÔNG NÊU** | |
| **N-Queens**: cách tạo instance | Mỗi puzzle "hiding 5–7 queens from an 8×8 board or 7–9 queens from a 10×10 board". Phần còn lại của bàn cờ là clue cố định. | §5.2, tr.8 |
| N-Queens: cách chọn bàn hoàn chỉnh ban đầu | **KHÔNG NÊU** | |
| N-Queens: cách chọn quân bị ẩn | **KHÔNG NÊU** (ngẫu nhiên? phân phối của số quân ẩn?) | |
| **Graph Coloring**: cách tạo instance | "Erdős–Rényi graphs with eight or ten vertices that allow a three-coloring" | §5.2, tr.8 |
| GC: xác suất cạnh $p$ | **KHÔNG NÊU** | |
| GC: cách bảo đảm 3-tô màu được (planted hay loại bỏ ngẫu nhiên) | **KHÔNG NÊU** | |
| Số lời giải hợp lệ mỗi instance | Có ví dụ: NQ8 có 4 lời giải, NQ10 có 5 lời giải, GC8 và GC10 đều có 16 lời giải. Đây chỉ là ví dụ minh hoạ, không phải thống kê. | Fig. 7, tr.19 |

### 2.1 Định dạng đầu vào / đầu ra

Định dạng tổng quát [§2, tr.3]:
- Lời giải là chuỗi token $x = (x^l)_{l=1}^{L}$, mỗi token one-hot trong $\mathbb{R}^{|V|}$.
- Mô hình dự đoán chuỗi vector xác suất $\hat{x} \in (\Delta^{|V|-1})^L$. Làm tròn bằng $\text{round}(\hat{x})^l = \text{onehot}(\arg\max \hat{x}^l)$.

Theo từng bài toán [App. A, "Input encoding", tr.14]:

- **N-Queens**: *"the problem specifies part of the completed grid. We keep these values fixed and construct the interpolant over the remaining entries."* Nghĩa là các ô clue được giữ cố định, nhiễu và interpolant chỉ áp dụng lên các ô còn lại.
- **Graph Coloring**: *"the problem is an input grid or an adjacency matrix, given separately from the output grid or vertex colors to be predicted. We embed these problem tokens, apply a bias-free 512 → 512 projection, and add the result to the projected $x_t$."* Đầu vào là ma trận kề, đầu ra là màu của từng đỉnh.

Chi tiết **KHÔNG NÊU**:
- $|V|$ cho N-Queens và Graph Coloring (có token "trống" hay không; có token "chưa tô" hay không).
- Độ dài chuỗi $L$ (có thể là $N^2$ cho N-Queens và số đỉnh cho GC, nhưng bài không nói).
- Cách chuyển ma trận kề thành chuỗi token (phẳng $n^2$ hay theo hàng) và loại position embedding dùng cho nó.
- Có bao nhiêu lời giải được dùng làm nhãn cho mỗi instance lúc train (một lời giải ngẫu nhiên mỗi lần, hay mọi lời giải).

---

## 3. Mô hình và huấn luyện

### 3.1 Kiến trúc [App. A, tr.14]

- Xây dựng trên TRM, thêm lớp chiếu cho interpolant nhiễu và embedding thời gian.
- Mạng dùng chung có **2 lớp, độ rộng 512**.
- Với N-Queens và Graph Coloring (không phải Sudoku): **noncausal attention, 8 đầu, rotary position embedding**.
- Mọi tác vụ: SwiGLU, độ rộng trung gian 1536, RMSNorm sau mỗi phép cộng residual.
- Tổng ≈ **7M tham số** (Table 2 cũng ghi 7M).
- Mã hoá đầu vào: lớp tuyến tính **không bias** ánh xạ mỗi token $|V|$-chiều của $x_t$ sang 512 đặc trưng.
- Embedding thời gian: MLP $1 \to 512 \to 512$ với SiLU, cộng vào **mọi vị trí** sau khi nhân embedding đầu vào với $\sqrt{512}$. Phần chiếu nhiễu và MLP thời gian thêm $512|V| + 263{,}680$ tham số.
- Kiểm tra số tham số `(*)`: mỗi lớp có attention $4 \cdot 512^2 \approx 1.05$M và SwiGLU $3 \cdot 512 \cdot 1536 \approx 2.36$M. Hai lớp cho ≈ 6.8M, cộng ≈ 0.27M, ra ≈ 7.1M. Khớp với "7M" trong bài. Có thể dùng làm kiểm tra tính đúng đắn của cài đặt.

**Trạng thái hồi quy** [App. A, "Recurrent computation", tr.14]:
- $z = (h, \ell)$. Dự đoán được giải mã từ $h$, còn $\ell$ dùng để cập nhật $h$.
- Một lần gọi denoiser lặp chu trình sau **3 lần**:
  $$\ell \leftarrow F_\theta(\ell + h + e_t)\ (m\ \text{lần}), \qquad h \leftarrow F_\theta(h + \ell)$$
  với $m = 4$ cho các tác vụ khác Sudoku ($m = 6$ cho Sudoku).
- Khi huấn luyện, **2 chu trình đầu chạy không gradient**, chỉ lan truyền ngược qua chu trình cuối.
- Giá trị khởi tạo $z_0$: **KHÔNG NÊU** (§3, tr.4 chỉ nói "$z_0$ is fixed").

### 3.2 Thuật toán huấn luyện [§3.1, tr.4–5; Algorithm 1, tr.4]

- Mục tiêu $\mathcal{L}_{LF}$ (Eq. 9, tr.4):
  $$\mathbb{E}\Big[\sum_{i=0}^{k-1} \text{CE}(\hat{x}_{t_i}, x_1)\Big],\quad (\hat{x}_{t_i}, z_{t_{i+1}}) = \hat{D}_{t_i}(I_{t_i}, \text{sg}(z_{t_i}); c)$$
- Interpolant $I_t = (1-t)x_0 + t x_1$, với $x_0 \sim \mathcal{N}(0, \sigma^2 I)$ [Eq. 3, tr.3].
- Các bước $t_0 < \dots < t_k$ được lấy từ phân phối $\mu$ (Mục 3.3). **Cùng một bộ $(c, x_0, x_1)$ được chia sẻ cho mọi $t_i$** [§3.1, tr.5]. Dùng stop-gradient giữa các bước.
- ACT [Eq. 10, tr.5; Algorithm 1 dòng 7–9]: đầu $q$ dự đoán xem $\text{round}(\hat{x}_{t_i}) = x_1$ hay không (BCE với trọng số $\lambda$). Nếu $\hat{q} > 1/2$ thì dừng vòng lặp và bỏ các bước sau.
- Hàm CE là **mean StableMax cross-entropy**, $k = 16$, $\lambda = 0.5$, exploration probability $0.1$ (theo TRM), "bfloat16 precision for forward passes" [App. E.1, tr.16].

### 3.3 Siêu tham số theo tác vụ [Table 6, tr.16, PDF]

| | NQ-8 | NQ-10 | GC-8 | GC-10 | Sudoku (để đối chiếu) |
|---|---|---|---|---|---|
| Noise scale $\sigma$ | 1 | 1 | 1 | $1/\sqrt{\lvert V\rvert}$ | $1/\sqrt{\lvert V\rvert}$ |
| Time sampler $\mu$ | Sorted | **Random-start** | Sorted | Sorted | Sorted |
| Pseudotargets | No | No | No | No | **Yes** |
| LR schedule | Constant | Constant | Constant | Constant | **Cosine** |
| Weight decay | 1.0 | 1.0 | 1.0 | 1.0 | **2.0** |
| Steps $n$ (suy luận) | 128 | 128 | 16 | 16 | 128 |
| Stochasticity $\gamma$ | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |

Nhận xét `(*)`: GC-8 dùng $\sigma = 1$ còn GC-10 dùng $\sigma = 1/\sqrt{|V|}$. Giá trị này đã được kiểm tra trên ảnh trang và khớp văn bản, nhưng khác nhau giữa hai cỡ cùng bài toán nên cần xác nhận lại (xem Mục 6).

**Hai bộ lấy mẫu thời gian** [App. E.1, tr.16]:
- **Sorted**: lấy $k+1$ giá trị độc lập từ $U[0,1]$ rồi sắp xếp để có $t_0 < \dots < t_k$.
- **Random-start**: lấy $t_0 \sim U[0,1]$, rồi sắp xếp $t_0$ cùng $k$ giá trị độc lập từ $U[t_0, 1]$. Cách này làm $t_0$ lớn hơn kỳ vọng so với Sorted.

### 3.4 Thiết lập huấn luyện chung [App. E.1, tr.16]

- Optimizer: **Adam-atan2**, $(\beta_1, \beta_2) = (0.9, 0.95)$, LR đỉnh $10^{-4}$, warmup **2k bước**.
- **Batch size 768**, gradient clipping **1.0**, parameter **EMA decay 0.999**.
- Weight decay theo Table 6.

### 3.5 Phần dùng chung với thiết lập Sudoku, và phần khác

| Hạng mục | Dùng chung với Sudoku? | Nguồn |
|---|---|---|
| Hàm mất mát (StableMax CE, $k = 16$, $\lambda = 0.5$) | **Chung** | App. E.1 |
| Optimizer, LR đỉnh, warmup, batch, clip, EMA, bf16 | **Chung** | App. E.1 |
| Exploration probability 0.1 | **Chung** | App. E.1 |
| Chiều rộng 512, 2 lớp, SwiGLU 1536, RMSNorm | **Chung** | App. A |
| Cấu trúc 3 chu trình, 2 chu trình đầu không gradient | **Chung** | App. A |
| Mã hoá đầu vào, MLP thời gian | **Chung** (cấu trúc), riêng ở cách nhúng token bài toán với GC | App. A |
| Thuật toán suy luận (Algorithm 2) | **Chung** | §3.2, tr.5 |
| Backbone | **KHÁC**: attention + RoPE (Sudoku dùng MLP-Mixer) | App. A |
| Số lần cập nhật $\ell$ mỗi chu trình $m$ | **KHÁC**: $m = 4$ (Sudoku: 6) | App. A |
| Pseudotargets | **KHÁC**: không dùng (Sudoku: có) | Table 6, App. D |
| LR schedule | **KHÁC**: Constant (Sudoku: Cosine → 0.1×) | Table 6, App. E.1 |
| Weight decay | **KHÁC**: 1.0 (Sudoku: 2.0) | Table 6 |
| $\sigma$, time sampler, số bước $n$ | **KHÁC** theo tác vụ | Table 6 |
| Quy mô dữ liệu (Sudoku: 1,000 mẫu train) | **KHÔNG NÊU** cho NQ và GC | App. D |

### 3.6 Điều kiện hoá và việc dùng pseudotargets

Pseudotargets (App. D, tr.16) **chỉ** được mô tả cho bài toán đơn lời giải. Trong bài toán nhiều lời giải, posterior mean không còn là một lời giải duy nhất, nên cơ chế này không áp dụng. Table 6 xác nhận NQ và GC đặt "No".

### 3.7 Tài nguyên tính toán [App. E, tr.17]

- "Representative training runs took roughly **1–5 hours** for Sudoku, Maze, N-Queens, and Graph Coloring."
- Loại GPU: H100, L40S, A40, A10 hoặc RTX 3090 (24–94 GB). **KHÔNG NÊU** GPU nào cho tác vụ nào.
- **Số bước huấn luyện, số epoch, thời điểm dừng**: **KHÔNG NÊU** cho NQ và GC. Chỉ Figure 5 (Sudoku, tr.9) hiển thị trục tới ~125k iteration.

---

## 4. Suy luận

### 4.1 Sampler [§3.2, tr.5; Algorithm 2, tr.4; App. C, tr.15]

Với lưới thời gian $0 = t_0 < \dots < t_n = 1$ và nhiễu $x_0 \sim p_0$:

1. $a \leftarrow [1 - \gamma(t_{i+1} - t_i)]_0^1$ (cắt về $[0, 1]$), $s \leftarrow a\, t_i$.
2. Lấy nhiễu mới $\epsilon \sim p_0$.
3. Lùi về $s$: $\bar{x}_s = a\, x_{t_i} + \sqrt{(1-s)^2 - (a-s)^2}\ \epsilon$ (Eq. 18, tr.15).
4. Gọi denoiser: $(\hat{x}_s, z_{t_{i+1}}) = \hat{D}_s(\bar{x}_s, z_{t_i}; c)$.
5. Bước Euler: $x_{t_{i+1}} = \bar{x}_s + (t_{i+1} - s)\,\dfrac{\hat{x}_s - \bar{x}_s}{1 - s}$ (Eq. 19, tr.15).
6. Kết quả: $\text{round}(x_{t_n})$.

Khi $\gamma = 0$ sampler suy biến thành Euler thuần (Eq. 8). Đây chính là cấu hình "ODE" của Table 5.

### 4.2 Cấu hình cho Table 2 [Table 6, tr.16, PDF]

| Tác vụ | Số bước $n$ | $\gamma$ |
|---|---|---|
| NQ-8 | 128 | 5.0 |
| NQ-10 | 128 | 5.0 |
| GC-8 | 16 | 5.0 |
| GC-10 | 16 | 5.0 |

Cột "SDE γ = 5" của Table 5 khớp với Table 2, nên cấu hình của Table 2 là SDE với γ = 5.

### 4.3 20 lần suy luận [§5.2, tr.8]

- "We perform **20 independent inferences per test instance**."
- Mỗi lần suy luận khởi tạo từ một mẫu nhiễu $x_0 \sim p_0$ độc lập, rồi đi hết sampler ở trên.
- N-Queens: Accuracy chỉ dùng **mẫu thứ nhất**. Coverage dùng cả 20 mẫu.
- Graph Coloring: lấy **tô màu hoàn chỉnh xuất hiện nhiều nhất** trong 20 mẫu cho Conflicts. Coverage dùng cả 20 mẫu.
- **Không dùng best-Q** khi suy luận Table 2? **KHÔNG NÊU**. Bài chỉ nói best-Q (5 quỹ đạo) cho phần ensembling của Table 1 [§3.2, tr.5; §5.1, tr.7; App. E.1, tr.16]. Điểm halting của ACT *không* dùng để dừng tích phân flow [App. E.1, tr.16].

---

## 5. Công thức coverage và cách đếm lời giải hợp lệ

### 5.1 Định nghĩa trong bài [§5.2, tr.8]

> "Coverage divides the number of distinct valid solutions recovered by the number of solutions compatible with the input."

Viết lại `(*)` cho một instance $c$ với $S(c)$ là tập mọi lời giải tương thích với $c$ và $\hat{X}(c)$ là tập 20 mẫu đã làm tròn:
$$\text{Coverage}(c) = \frac{\left|\{\hat{x} \in \hat{X}(c) : \hat{x}\ \text{hợp lệ}\}\ \text{(đã khử trùng lặp)}\right|}{|S(c)|}$$
Sau đó bài báo cáo giá trị theo tỉ lệ phần trăm (Table 2).

### 5.2 Hai thành phần bài nêu và những gì bài không nêu

| Thành phần | Trong bài | Nguồn |
|---|---|---|
| Tử số | "distinct valid solutions recovered" từ 20 lần suy luận | §5.2, tr.8 |
| Mẫu số | "solutions compatible with the input" | §5.2, tr.8 |
| "Hợp lệ" của N-Queens | **KHÔNG NÊU** định nghĩa tường minh (đủ số quân, không tấn công nhau, và khớp clue?). Suy luận hợp lý từ cụm "compatible with the input" là phải khớp clue, nhưng bài không nói. | |
| "Hợp lệ" của Graph Coloring | **KHÔNG NÊU** (mọi đỉnh được tô, không có cạnh nào cùng màu?) | |
| Cách khử trùng lặp ("distinct") | **KHÔNG NÊU** (khớp chính xác sau khi làm tròn? có coi hoán vị màu là một không?) | |
| Cách đếm $|S(c)|$ | **KHÔNG NÊU** (liệt kê vét cạn? hoán vị màu có tính riêng không?) | |
| Gộp qua instance (trung bình theo instance hay tổng tử trên tổng mẫu) | **KHÔNG NÊU** | |
| Conflicts | "sum its conflicts over the test graphs" [§5.2, tr.8]. Định nghĩa "conflict" (số cạnh cùng màu? số đỉnh vi phạm?) **KHÔNG NÊU**. | |

### 5.3 Minh hoạ trong Figure 7 [tr.19]

- NQ8: 4 lời giải hợp lệ. GRAM tìm 3/4, có 3/20 mẫu không hợp lệ. Looped flows tìm đủ 4/4.
- NQ10: 5 lời giải hợp lệ. GRAM tìm 4/5, có 7/20 mẫu không hợp lệ. Looped flows tìm đủ 5/5.
- GC8: 16 lời giải. GRAM tìm 8/16. Looped flows tìm 12/16.
- GC10: 16 lời giải. GRAM tìm 9/16. Looped flows tìm 13/16.
- Caption: "On the graph-coloring problems, no invalid samples were observed".
- Quan sát `(*)`: cả hai ví dụ GC đều có đúng 16 lời giải, và 16 không chia hết cho $3! = 6$. Vì vậy không thể suy ra từ hình xem hoán vị màu có được tính là lời giải khác nhau hay không. Phải kiểm tra trong bài GRAM.

---

## 6. Mơ hồ (mọi điểm bài không nêu rõ)

**Dữ liệu**
1. Số mẫu train, test, validation cho từng cấu hình. Train và test có tách rời không, có loại trùng không.
2. Cách sinh bàn N-Queens hoàn chỉnh, quân nào bị ẩn, phân phối số quân bị ẩn trong khoảng 5–7 (8×8) và 7–9 (10×10).
3. Xác suất cạnh $p$ của đồ thị Erdős–Rényi. Cách bảo đảm đồ thị 3-tô màu được. Có lọc đồ thị không liên thông không.
4. Số lời giải nhãn mỗi instance lúc train.
5. Từ vựng $|V|$, độ dài chuỗi $L$, cách mã hoá ma trận kề, token đặc biệt (nếu có).

**Mô hình / huấn luyện**
6. Số bước huấn luyện, số epoch, lịch dừng và cách chọn checkpoint. Có đánh giá bằng trọng số EMA hay không.
7. Giá trị khởi tạo $z_0$ và cách tạo đầu $q$ của ACT (đầu ra từ $h$? kiến trúc?).
8. "Exploration probability 0.1" chỉ được tham chiếu tới TRM [App. E.1, tr.16], không giải thích lại trong bài.
9. Mô hình huấn luyện riêng cho từng cỡ (NQ-8, NQ-10, GC-8, GC-10) hay dùng chung một mô hình? Table 6 có cột riêng nhưng bài không nói rõ.
10. $\sigma = 1$ cho GC-8 nhưng $\sigma = 1/\sqrt{|V|}$ cho GC-10 (Table 6). Bài không giải thích lý do chọn, và không rõ $|V|$ của hai bài toán có giống nhau không.
11. Lý do NQ-10 dùng Random-start còn ba cấu hình còn lại dùng Sorted. Không có ablation hay giải thích.
12. Có dùng mặt nạ clue khi tính loss không. App. A chỉ nói clue "kept fixed", không nói rõ ảnh hưởng tới CE.

**Suy luận / đánh giá**
13. Lưới thời gian có đều không (Algorithm 2 chỉ ghi $0 = t_0 < \dots < t_n = 1$).
14. Có dùng best-Q hay chọn lọc nào khác trong Table 2 không (xem 4.3).
15. Định nghĩa chính xác của "valid", "distinct", "conflict", và cách gộp coverage (xem Mục 5).
16. "Three repeated tests" [Table 2 caption, tr.9] là ba lần huấn luyện với seed khác nhau, hay ba lần suy luận lại trên một mô hình? Table 1 dùng cụm "three seeds" [tr.7], nên hai cách diễn đạt không nhất quán.
17. Độ lệch chuẩn là độ lệch mẫu hay tổng thể.
18. Với Conflicts, tổng tuyệt đối phụ thuộc kích thước tập test, nhưng kích thước đó không nêu. Vì vậy chỉ so được với baseline nếu dùng đúng tập test của GRAM.

**Baseline**
19. Các baseline được huấn luyện thế nào, trên cùng dữ liệu nào, bao nhiêu bước. Bài này chỉ chép số từ GRAM.

---

## 7. Câu hỏi cần giải quyết bằng bài GRAM (arXiv:2605.19376)

Cần tìm trong bài GRAM (và mã nguồn của nó, nếu có):

**Dữ liệu**
1. **Q1.** Quy trình sinh instance N-Queens: cách sinh bàn hoàn chỉnh, cách chọn quân ẩn, tách train/test, kích thước từng tập.
2. **Q2.** Quy trình sinh đồ thị Graph Coloring: $p$ của Erdős–Rényi, cách bảo đảm 3-tô màu được, kích thước train/test.
3. **Q3.** Mô hình thấy bao nhiêu lời giải nhãn mỗi instance lúc train (một lời giải ngẫu nhiên, hay toàn bộ).
4. **Q4.** Từ vựng và định dạng token cho đầu vào và đầu ra, cách mã hoá ma trận kề, độ dài chuỗi.
5. **Q5.** Cách liệt kê tập lời giải $S(c)$, đặc biệt: hoán vị màu của Graph Coloring được coi là một hay nhiều lời giải (lưu ý Mục 5.3: 16 lời giải ở cả hai ví dụ).

**Metric**
6. **Q6.** Định nghĩa "valid" cho N-Queens (có đòi khớp clue không) và Graph Coloring.
7. **Q7.** Định nghĩa "conflict" cho Graph Coloring, và cách xử lý đỉnh chưa tô.
8. **Q8.** Công thức coverage: khử trùng lặp bằng khớp chính xác hay quy đổi hoán vị màu, tính trung bình theo instance hay gộp.
9. **Q9.** Accuracy của N-Queens: "mẫu đầu tiên" là mẫu nào, và hợp lệ nghĩa là gì.
10. **Q10.** Quy mô tập test dùng cho Conflicts (cần để so sánh số tuyệt đối với baseline).

**Quy trình**
11. **Q11.** "20 independent inferences" của GRAM được thực hiện thế nào (nhiễu độc lập, hay lấy mẫu latent), để dùng cùng quy trình.
12. **Q12.** Số seed và số lần lặp của baseline, và độ lệch chuẩn tính thế nào, để kết quả của ta so sánh được.
13. **Q13.** Ngân sách huấn luyện (số bước, epoch, batch) của GRAM trên hai tác vụ này, làm điểm tham chiếu cho thiết lập khi bài này không nêu.
14. **Q14.** GRAM có công bố mã sinh dữ liệu và đánh giá không. Nếu có thì dùng trực tiếp để bảo đảm khớp tập test của baseline.
