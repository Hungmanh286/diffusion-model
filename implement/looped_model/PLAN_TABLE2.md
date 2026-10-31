# PLAN: Cài đặt và tái hiện Table 2 (N-Queens 8×8 trước)

Bài chính: *Thinking with Looped Flows* (arXiv:2609.11801). Bài tham chiếu: GRAM (arXiv:2605.19376v4).
Tài liệu liên quan: `SPEC_TABLE2.md` (mục tiêu, dữ liệu, mơ hồ), `DEVIATIONS.md` (sai khác so với bài, chờ duyệt).

**Trạng thái: CHỐT cho phiên 30/10/2026. Kế hoạch đã đóng băng, chưa có code. Các mục chờ duyệt nằm ở Mục 8 (danh sách tick) để bạn duyệt ở phiên sau. Chưa bắt đầu M1.**

**Quy ước nhãn bằng chứng**
- `[ĐÃ XÁC MINH <nguồn>]`: đọc trực tiếp trong mã hoặc bài, có vị trí.
- `[CHƯA XÁC MINH]`: chưa có bằng chứng trực tiếp.
- `[GIẢ ĐỊNH]`: lựa chọn của người lập kế hoạch khi nguồn không nêu. Mọi giả định đều nằm trong Mục 7 để bạn quyết.
- Mã TRM đọc ở commit `c01103738605ba39d1430519b1ee0c62f4c707f8` (31/03/2026) của `SamsungSAILMontreal/TinyRecursiveModels`. Đường dẫn dưới đây tính từ gốc repo đó. M0 phải clone lại đúng commit này vào một thư mục tham chiếu cố định, vì bản đã đọc đang nằm trong thư mục tạm của phiên làm việc.

---

## 1. Trạng thái quyết định

| # | Quyết định | Trạng thái |
|---|---|---|
| D1 | "Valid" cho coverage: khớp clue và là bàn hợp lệ | **Đã duyệt** |
| D2 | Đánh giá trên **input duy nhất**, **báo cả hai cách gộp** coverage (theo input duy nhất và theo dòng) | **Đã duyệt** (xem 5, M2) |
| D3 | Từ vựng $\lvert V\rvert = 3$ theo GRAM (0=pad, 1=empty, 2=queen) | **Đã duyệt** |
| D4 | Ô clue giữ cố định; ô còn lại bị nhiễu hoá và dự đoán; loss chỉ tính trên ô không cố định | **Chưa duyệt** (không nằm trong danh sách quyết định lần này; xem 7) |
| D5 | Khởi tạo $z_0$ và đầu ACT | **Chưa chốt**: đã xác minh trong TRM (mục 2.1, 2.2); còn điểm mở (mục 7) |
| D6 | Lưới thời gian suy luận đều | **Chưa duyệt** (không nằm trong danh sách; xem 7) |
| D7 | Đánh giá bằng trọng số EMA | **Chưa chốt**: đã xác minh trong TRM (mục 2.3) |
| D8 | Độ dài huấn luyện | **Chưa chốt**: sẽ quyết sau pilot (mục 5, M6) |
| D9 | Lấy mẫu dữ liệu train đều trên các **dòng** | **Đã duyệt** |
| D10 | Một hạt giống chia dữ liệu cố định, ghi vào config | **Đã duyệt** |

Thay đổi so với bản kế hoạch trước: ước lượng "≈ 28k bước" cho D8 chưa rõ đơn vị. Xem mục 5, M6.

---

## 2. Xác minh nguồn

### 2.1 D5 — Khởi tạo $z_0$ — `[ĐÃ XÁC MINH trong TRM]`, cách LF dùng `[GIẢ ĐỊNH]`

| Điểm | Bằng chứng |
|---|---|
| $z_H$ và $z_L$ khởi tạo bằng hai vector **cố định, không huấn luyện**, kích thước `hidden_size`, lấy từ truncated normal std = 1 | `models/recursive_reasoning/trm.py:153-154`: `nn.Buffer(trunc_normal_init_(torch.empty(hidden_size), std=1), persistent=True)` |
| Truncated normal kiểu JAX, cắt ở $\pm 2$ lần độ lệch hiệu chỉnh | `models/common.py:7-32` |
| Vector được phát (broadcast) cho mọi vị trí và mọi mẫu khi bắt đầu một chuỗi | `trm.py:190-194` (`reset_carry`, `torch.where(reset_flag..., H_init, ...)`) |
| Trạng thái truyền sang bước sau bị cắt gradient | `trm.py:219` (`z_H.detach()`, `z_L.detach()`) |

- Bài chính chỉ nói "$z_0$ is fixed" [§3, tr.4] và "built on TRM" [App. A, tr.14]. Bài **không nói** nó dùng lại $H_{init}$ và $L_{init}$ của TRM. Việc dùng chúng là **giả định** hợp lý nhưng chưa được bài xác nhận.

### 2.2 D5 — Đầu ACT — `[ĐÃ XÁC MINH trong TRM]`; vị trí đọc cho N-Queens `[CHƯA XÁC MINH]`

| Điểm | Bằng chứng |
|---|---|
| Đầu `q_head` là `Linear(hidden, 2, bias=True)` đọc từ $z_H$ ở **vị trí 0** | `trm.py:131`, `trm.py:221` |
| Khởi tạo: trọng số = 0, bias = −5 | `trm.py:158-160` |
| Mặc định `no_ACT_continue=True`: chỉ dùng logit "halt"; dừng khi `q_halt_logits > 0` (tương đương $\hat q > 1/2$) | `trm.py:63`, `config/arch/trm.yaml:26`, `trm.py:280-281` |
| Nhãn của $q$ = chuỗi dự đoán **đúng toàn bộ** các token có nhãn | `models/losses.py:70-71`, `losses.py:88` (BCE-with-logits) |
| Tổng loss = LM loss + 0.5 × (loss halt + loss continue) | `losses.py:102` |
| Exploration: với xác suất 0.1, ép chuỗi chạy ít nhất `min_halt_steps` ∈ {2..`halt_max_steps`} bước | `trm.py:286-287`, `trm.yaml:6-7` (0.1 và 16) |
| Cơ chế dừng chỉ hoạt động khi **huấn luyện**; lúc đánh giá luôn chạy hết số bước tối đa | `trm.py:275-278` |

- Khớp với bài chính: $\lambda = 0.5$, exploration 0.1 "(Jolicoeur-Martineau 2025)", $k = 16$, ngưỡng $\hat q > 1/2$ [Algorithm 1 dòng 9, tr.4; App. E.1, tr.16].
- Khớp thêm: README của TRM đặt `L_cycles=6` cho Sudoku (`README.md:79`) và `L_cycles=4` cho Maze và ARC (`README.md:111,128,143`). Điều này khớp $m=6$ và $m=4$ trong App. A của bài chính.
- **Chưa xác minh**: TRM đọc $z_H$ ở vị trí 0, vốn là một ô puzzle-embedding. Bài chính chỉ dùng puzzle embedding cho ARC [App. E.1, tr.16], nên với N-Queens không có ô đó. Chưa có bằng chứng về vị trí đầu $q$ đọc trong trường hợp này (xem 7, Q1).

### 2.3 D7 — Đánh giá bằng EMA — `[ĐÃ XÁC MINH trong TRM]`; cách LF dùng `[CHƯA XÁC MINH]`

| Điểm | Bằng chứng |
|---|---|
| Mặc định `ema: False`, tốc độ 0.999 | `pretrain.py:82-83`, `config/cfg_pretrain.yaml:40-41` |
| Các lệnh chạy Sudoku, Maze, ARC trong README đều đặt `ema=True` | `README.md:80,93,112,129,144` |
| EMA cập nhật sau **mỗi batch** | `pretrain.py:612-613` |
| Khi bật EMA, đánh giá dùng **bản sao EMA** | `pretrain.py:619-625` |
| Checkpoint được lưu từ `train_state_eval`, tức trọng số EMA | `pretrain.py:641-642` |
| Chi tiết EMA: chỉ các tham số `requires_grad`, khởi tạo từ trọng số bước 0, **không** hiệu chỉnh bias; `H_init` và `L_init` là buffer nên không được EMA | `models/ema.py:9-21` |

- Bài chính chỉ nói "parameter EMA with decay 0.999" [App. E.1, tr.16], **không nói** đánh giá bằng EMA hay trọng số thô.
- Đề xuất, `[GIẢ ĐỊNH]`: đánh giá bằng EMA làm kết quả chính (như TRM) và luôn kèm kết quả trọng số thô (chi phí thấp). Chờ bạn chốt.

### 2.4 StableMax — `[ĐÃ XÁC MINH trong TRM]` (chưa đọc bài Prieto gốc)

| Điểm | Bằng chứng |
|---|---|
| $s(x) = x + 1$ khi $x \ge 0$; $s(x) = 1/(1 - x + \epsilon)$ khi $x < 0$, $\epsilon = 10^{-30}$ | `models/losses.py:11-16` |
| $\log\text{StableMax}(x) = \log\big(s(x) / \sum_j s(x_j)\big)$ | `losses.py:19-21` |
| Cross-entropy tính ở **float64** | `losses.py:25` |
| "Mean": mỗi chuỗi lấy trung bình trên số token có nhãn, rồi tổng trên batch; sau đó nhân $1/\text{global\_batch\_size}$ | `losses.py:68,87`; `pretrain.py:305` |
| Chọn qua cấu hình `loss_type: stablemax_cross_entropy` | `config/arch/trm.yaml:4` |

- Giá trị chuẩn tính tay cho test: logits $(1, 0, -1)$ cho $s = (2, 1, 0.5)$, tổng 3.5, log-xác suất $= \log(2/3.5),\ \log(1/3.5),\ \log(0.5/3.5)$.

### 2.5 Adam-atan2 — **chưa cài được trên máy này, chưa thử trên nơi khác**

| Điểm | Bằng chứng |
|---|---|
| TRM dùng `adam-atan2==0.0.3`, optimizer `AdamATan2(lr, weight_decay, betas)` | `specific_requirements.txt:2`; `pretrain.py:20,150-155` |
| PyPI có bản `0.0.3`, nhưng chỉ có **sdist**, không có wheel dựng sẵn | `pip index versions adam-atan2`; `pip download` chỉ trả `adam_atan2-0.0.3.tar.gz` |
| Gói **biên dịch CUDA extension** `adam_atan2_backend`, cần `torch` và `nvcc`, kiến trúc hỗ trợ `{80, 86, 89, 90}`; bình luận trong mã: "Needs testing on consumer GPUs" | `adam_atan2-0.0.3/setup.py:9,21-44` |
| Máy này: **chưa có `torch`**, **chưa có `nvcc`** | `python3 -c "import torch"` báo lỗi; `which nvcc` rỗng |
| Quy tắc cập nhật trong kernel (a = b = 1, không có hằng 1.27): $w \leftarrow w(1 - \text{lr}\cdot\text{wd})$; $m \leftarrow \text{lerp}(m, g, 1-\beta_1)$; $v \leftarrow \text{lerp}(v, g^2, 1-\beta_2)$; $w \leftarrow w - \frac{\text{lr}}{1-\beta_1^t}\,\text{atan2}\!\big(m,\ \sqrt{v}/\sqrt{1-\beta_2^t}\big)$ | `csrc/adam_atan2.cu:47-53`, `:77-86` |

**Phương án**, ghi tại `DEVIATIONS.md` mục DEV-01, **chờ bạn duyệt**:
- (A) Cài bản CUDA chuẩn khi có `torch` và CUDA toolkit. Nằm trong kiến trúc hỗ trợ cho GPU 8.6 (máy này) và 9.0 (H100).
- (B) Tự viết bản PyTorch thuần **khớp đúng công thức kernel** (khuyến nghị nếu (A) không cài được).
- (C) Gói `adam-atan2-pytorch==0.4.1`: **không** nên dùng nguyên trạng. Mặc định `a=1.27`, và bias correction khác kernel: lấy $\text{atan2}(m/\text{bc}_1,\ \sqrt{v/\text{bc}_2})$ thay vì $\frac{1}{\text{bc}_1}\text{atan2}(m,\ \sqrt{v/\text{bc}_2})$.
- Chênh lệch cụ thể ở bước đầu (tính tay từ các dòng trên, cần xác nhận bằng số trong test): với $g > 0$, kernel cho bước $\approx 0.9967\ \text{lr}$ (vì $\text{atan2}(0.1g, |g|) = \arctan 0.1$ và nhân $\text{lr}/0.1$), còn bản (C) với $a=1$ cho $\pi/4 \approx 0.7854\ \text{lr}$.

---

## 3. Phần cứng

Dòng `[GPU: ...]` bạn gửi **để trống**. **Kế hoạch hiện chưa phụ thuộc phần cứng.** M5 và M6 được viết theo kiểu *đo trước, quyết sau*, không cố định số bước hay kích thước micro-batch.

Quan sát (không phải thông tin bạn cung cấp, chỉ để tham khảo): `nvidia-smi` trên máy này báo **NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, compute capability 8.6**, driver 595.99.02. Ổ đĩa còn 51 GB. Chưa có `torch`, `pytest`, `nvcc`. Nếu đây là máy huấn luyện, batch 768 có thể không vừa bộ nhớ và cần gradient accumulation. Đó là điều pilot M6 sẽ đo, không phải kết luận.

Khi bạn điền dòng GPU, tôi sẽ cập nhật mục này và chỉnh M5, M6.

---

## 4. Cấu trúc file

```
implement/looped_model/
├── SPEC_TABLE2.md                 (đã có)
├── PLAN_TABLE2.md                 (file này)
├── DEVIATIONS.md                  sai khác so với bài, mỗi mục có trạng thái duyệt
├── configs/nq8.yaml               siêu tham số từ Table 6 và App. E.1; không hard-code
├── ref/                           thư mục tham chiếu cố định (M0): TRM đúng commit c011037
├── lf/
│   ├── data_nqueens.py            sinh lời giải, instance, chia theo input, mã hoá token
│   ├── metrics.py                 is_valid_board, accuracy, coverage (cả hai cách gộp)
│   ├── model.py                   denoiser có trạng thái (h, ℓ), RoPE, SwiGLU, đầu ACT
│   ├── flow.py                    interpolant, time sampler, StableMax CE, loss ACT
│   ├── sampler.py                 Algorithm 2 (noise-backtracking, γ)
│   ├── optim.py                   Adam-atan2 (chọn theo DEV-01) và EMA
│   ├── train.py                   Algorithm 1, checkpoint
│   └── eval_table2.py             20 lần suy luận mỗi input, in cả hai cách gộp
├── tests/
│   ├── golden_nqueens.py          từ nq_check.py đã chạy
│   └── test_*.py                  pytest, chạy được trên CPU
└── runs/                          log, checkpoint, kết quả (không commit)
```

---

## 5. Các mốc, test và cổng

Mỗi mốc phải qua cổng mới sang mốc sau. Mốc đầu tiên tôi sẽ làm khi bạn duyệt là M1.

### M0. Môi trường (không cần GPU)

- Tạo môi trường Python, cài `torch`, `pytest`, `numpy`.
- Clone TRM đúng commit `c011037...` vào `ref/` (read-only, chỉ để đối chiếu).
- Thử cài `adam-atan2==0.0.3` (phương án A của DEV-01). Ghi kết quả vào `DEVIATIONS.md`. **Nếu không cài được, dừng và chờ bạn duyệt phương án B** trước khi làm M3 trở đi.
- Cổng: `pytest` chạy được trên CPU. Kết quả của việc cài Adam-atan2 đã được ghi lại.

### M1. Bộ sinh dữ liệu (`data_nqueens.py`)

Giá trị chuẩn lấy từ phép liệt kê đã chạy (`nq_check.py`), không lấy từ con số trong bài:

| Test | Kỳ vọng |
|---|---|
| Số lời giải đầy đủ | **92** |
| Tổng số dòng (lời giải × tập quân bị bỏ, $k \in \{5,6,7\}$) | **8,464** |
| Số input duy nhất | **5,148** |
| Số lời giải trung bình mỗi input duy nhất | **1.644** (GRAM ghi 1.65) |
| Số lời giải tối đa của một input | **18** |
| Tỉ lệ input có đúng một lời giải | **0.655** |
| Chia theo input: giao của tập input train và test | **rỗng** |
| Tỉ lệ dòng test | ≈ 14.65% (GRAM: 1,240 / 8,464) |
| Mỗi dòng: nhãn là bàn hợp lệ và khớp clue | 100% |
| Mã hoá | độ dài 64, giá trị trong {1, 2} (0 chỉ dành cho pad) |
| Mã hoá rồi giải mã | khôi phục đúng bàn |

Số dòng train và test sẽ không khớp đúng 7,224 / 1,240 vì không có hạt giống của GRAM (xem DEV-02).

### M2. Bộ đếm metric (`metrics.py`)

- **`is_valid_board`**: 8 queens, không tấn công theo hàng, cột, chéo, và khớp clue. Test bằng 92 lời giải thật (đều hợp lệ), bàn có 7 quân, bàn có hai quân cùng đường chéo, bàn vi phạm clue (đều không hợp lệ).
- **Coverage trên input đã biết**: đưa đủ tập lời giải tương thích cho coverage 1.0; một nửa tập cho 0.5; lặp cùng một lời giải 20 lần chỉ tính một lần; mẫu không hợp lệ không đổi tử số; lời giải hợp lệ nhưng sai clue không tính (D1).
- **Hai cách gộp** (D2): tính cả hai trong mọi lần đánh giá và **báo cả hai**. Công tắc chỉ chọn cách nào là số chính trong bảng tóm tắt.
  - (a) theo input duy nhất: trung bình cộng của coverage từng input duy nhất.
  - (b) theo dòng: mỗi dòng test tính một lần, input trùng lặp được tính lặp.
- **Accuracy**: chỉ xét mẫu đầu tiên trong 20 mẫu.
- Cổng: mọi test trên qua, và hai cách gộp cho cùng kết quả khi mọi input chỉ có một lời giải.

### M3. Mô hình (`model.py`) và optimizer (`optim.py`)

| Test | Kỳ vọng |
|---|---|
| Hình dạng | vào $(B, 64, \lvert V\rvert)$ và $z$; ra $(B, 64, \lvert V\rvert)$ và $z$ mới |
| Số tham số | ≈ **7.08M** theo tính tay: $6{,}815{,}744 + 512\lvert V\rvert + 263{,}680$, chưa tính đầu ra và đầu ACT; chấp nhận 6.9–7.3M |
| Noncausal | đổi token $j$ làm đổi đầu ra ở vị trí $i \neq j$ |
| Cắt gradient | 2 chu trình đầu không có gradient, chu trình cuối có (`trm.py:208-216`) |
| Stop-gradient giữa các bước | trạng thái truyền sang bước sau đã `detach` (`trm.py:219`) |
| Số lần gọi $F_\theta$ | 3 × (4 + 1) = 15 mỗi lần gọi denoiser |
| Khởi tạo $z_0$ | hai vector cố định (buffer, không có gradient), phát cho mọi vị trí (mục 2.1) |
| Đầu ACT khi khởi tạo | trọng số 0, bias −5, nên $\hat q \approx \sigma(-5) \approx 0.0067$, tức không dừng sớm |
| Optimizer, bước đầu | khớp công thức kernel: độ lớn bước đầu $\approx 0.9967\ \text{lr}$ (mục 2.5); kiểm bằng số |

### M4. Lõi flow (`flow.py`, `sampler.py`)

Dùng **denoiser oracle** $D_t(x) = x_1$ đã biết:

- **Interpolant**: $t=0$ cho nhiễu thuần, $t=1$ bằng đúng $x_1$.
- **Time sampler**: `sorted` ra $k+1$ giá trị tăng; `random-start` có $t_0 \sim U[0,1]$ và các giá trị còn lại $\ge t_0$.
- **Nhiễu chia sẻ**: cùng $x_0$ cho mọi $t_i$ trong một rollout.
- **Ô clue** (nếu D4 được duyệt): bằng đúng giá trị clue ở mọi $t$.
- **Sampler với oracle**: với $\gamma = 0$ và $\gamma = 5$ đều ra **đúng $x_1$ tại $t=1$**; $\gamma=0$ suy biến về Euler (Eq. 8); mô phỏng Monte Carlo cho $\bar x_s$ trung bình $s\,x_1$ và độ lệch chuẩn $\sigma(1-s)$ (App. C, tr.15).
- **StableMax CE**: khớp giá trị chuẩn ở mục 2.4, tính ở float64; bỏ qua token không có nhãn.
- **ACT**: nhãn $q$ bằng 1 khi chuỗi dự đoán đúng toàn bộ; vòng lặp dừng khi $\hat q > 1/2$.
- **Exploration**: với xác suất 0.1 ép chạy ít nhất $m_{\min} \sim U\{2,\dots,16\}$ bước; kiểm tần suất thực nghiệm và chuỗi không bao giờ dừng trước $m_{\min}$.

### M5. Huấn luyện nhỏ (`train.py`)

Theo Algorithm 1 [tr.4]: **một bước gradient cho mỗi bước denoising** (dòng 8 nằm trong vòng lặp `for i`, đã xác nhận bằng ảnh trang), giống TRM (`pretrain.py:303-322`: mỗi batch một lần forward, backward, optimizer step). Cách gộp batch xem 7, Q2.

1. **Overfit 1 batch**: 32 dòng, loss tiến về ≈ 0 và độ chính xác 1 bước ở $t$ lớn đạt 100%.
2. **Overfit tập nhỏ một lời giải**: khoảng 64 input có đúng một lời giải; lấy mẫu từ nhiễu đạt accuracy 100% khi chạy đủ bước.
3. **Overfit một input nhiều lời giải**: ví dụ một input có 3 lời giải. 20 mẫu phải phủ **gần đủ** các lời giải với tỉ lệ không quá lệch. Test này chứng minh phần "nhiều lời giải" hoạt động, nằm trước khi tốn tính toán đầy đủ.
4. **Ablation định tính nhanh**: tắt chia sẻ nhiễu hoặc tắt nhiễu giảm dần để thấy ít nhất một hướng giảm (Table 3 của bài).
- Hardware: chưa phụ thuộc phần cứng. Mọi test chạy được trên CPU với mô hình thu nhỏ. Khi có dòng GPU, thêm một lần chạy kiểm tra trên GPU và đo bộ nhớ ở batch 768.

### M6. Pilot rồi mới quyết ngân sách (D8)

**Không chạy 3 seed trước khi bạn duyệt ngân sách.**

*Đơn vị cần định nghĩa rõ*: bài chính không nêu số bước huấn luyện cho N-Queens 8×8. Số "3K epoch" của GRAM (Table 7, tr.20) là tham chiếu của bài khác. Với Algorithm 1, một lượt qua dữ liệu (≈ 7.2k dòng / 768 ≈ 9.4 batch) có thể sinh tới 16 bước tối ưu hoá mỗi batch. Vì vậy "3K epoch" có thể là 28k batch hay tới ≈ 450k bước tối ưu hoá tuỳ cách đếm. Con số "≈ 28k bước" trong bản kế hoạch trước **không đủ căn cứ** và được rút lại. Pilot đo cả hai đơn vị.

- **P0, smoke**: chạy vài trăm bước, đo giây/bước, bộ nhớ đỉnh ở batch 768 và ở batch nhỏ hơn kèm gradient accumulation.
- **P1, pilot một seed**: chạy đến khi **hết 3,000 batch hoặc 6 giờ**, tuỳ cái nào đến trước (đề xuất, chờ bạn chốt). Mỗi 250 batch đánh giá trên tập con cố định gồm 100 input duy nhất, 20 mẫu mỗi input, $n = 128$, $\gamma = 5$.
- **Ghi lại**: loss, giây/batch, bộ nhớ, accuracy mẫu đầu, coverage (cả hai cách gộp), phân phối số bước halting, độ hiệu chuẩn của $q$.
- **Quy tắc chọn ngân sách** (đề xuất): nếu cuối pilot coverage còn tăng hơn 1 điểm phần trăm trong 25% cuối, nội suy đường cong và đề xuất ngân sách dài hơn; nếu đã đi ngang, đề xuất ngân sách ≈ 1.5 lần điểm bắt đầu đi ngang. Tôi trình đường cong và ước tính thời gian cho bạn, **bạn chốt ngân sách**.
- **Sau khi bạn chốt**: chạy 3 seed theo Table 6 ($\sigma=1$, `sorted`, hằng số LR, weight decay 1.0, batch 768, EMA 0.999), kèm trọng số thô để so sánh với EMA (D7).

### M7. Kết quả và đối chiếu

| So với | Giá trị mục tiêu |
|---|---|
| Table 2, NQ 8×8 | Accuracy 99.9±0.1, Coverage 91.4±0.4 |
| Table 5 và 7, ODE ($\gamma=0$) | Coverage 91.4±0.4, accuracy 98.4±0.3 (chỉ đổi $\gamma$ lúc suy luận) |

Báo cáo trung thực nếu lệch, và báo cả hai cách gộp coverage. Tập test của ta **không trùng** tập của GRAM (DEV-02), nên so với baseline chỉ ở mức gần đúng.

Sau NQ 8×8: NQ 10×10 (khác ở `random-start`, dữ liệu lớn hơn), rồi Graph Coloring (có dataset công khai trên HuggingFace). Hai phần này cần plan riêng.

---

## 6. Rủi ro còn lại

1. **Adam-atan2 chưa cài được**. Nếu phải dùng bản tự viết, cần test số học chặt (M3) và bạn duyệt DEV-01.
2. **Không có hạt giống chia của GRAM**: so sánh tuyệt đối với baseline chỉ gần đúng (DEV-02).
3. **Ngân sách huấn luyện chưa rõ**: bài chính **KHÔNG NÊU**. Pilot giảm rủi ro nhưng không loại bỏ.
4. **Bộ nhớ**: batch 768 có thể không vừa trên GPU nhỏ. Gradient accumulation thay đổi thống kê batch-norm nếu có (mô hình không dùng batch-norm, nên chủ yếu ảnh hưởng thời gian).
5. **Coverage nhạy với định nghĩa**: đã giảm bằng việc báo cả hai cách gộp.
6. **Cách gộp batch cho rollout có halting** chưa được bài nêu (Q2).
7. **Baseline trong Table 2 của bài chính lệch Table 1 của GRAM v4** ở một số ô (liệt kê trong báo cáo trước); khi so sánh phải ghi rõ nguồn.
8. **Hai lần đọc mã TRM đều ở một commit**; nếu bạn dùng phiên bản khác, các số dòng ở mục 2 sẽ lệch.

---

## 7. Câu hỏi mở (cần bạn quyết)

| # | Câu hỏi | Lựa chọn và khuyến nghị |
|---|---|---|
| Q1 | Đầu $q$ đọc từ đâu khi N-Queens không có ô puzzle-embedding (TRM đọc vị trí 0, vốn là ô puzzle-embedding: `trm.py:221`)? | (a) trung bình mean-pool của $h$ trên các vị trí (không thêm tham số, **khuyến nghị**, `[GIẢ ĐỊNH]`); (b) đọc vị trí 0 (một ô bàn cờ, ít hợp lý); (c) thêm một token toàn cục học được (khác với bài chính) |
| Q2 | Cách gộp batch khi mỗi mẫu dừng ở bước khác nhau: Algorithm 1 viết cho một mẫu, bài không nêu | (a) rollout đồng bộ, mẫu đã dừng bị che khỏi loss (đơn giản, dễ test, **khuyến nghị cho bản đầu**); (b) kiểu TRM, mỗi vị trí batch mang một bước riêng và được thay mẫu mới khi dừng (`trm.py:249-297`; hiệu quả hơn nhưng phức tạp, cần lưu lịch $t_0..t_k$ riêng cho từng vị trí) |
| Q3 | Eq. (9) là tổng loss rồi **một** bước cập nhật mỗi rollout, nhưng Algorithm 1 cập nhật **mỗi bước** | Khuyến nghị theo Algorithm 1 (khớp TRM). Bài không giải thích sự khác nhau |
| Q4 | Có duyệt D4 (ô clue cố định, loss chỉ trên ô không cố định) không? | Khuyến nghị: có. Cơ sở: App. A tr.14 "keep these values fixed"; phần loss là lựa chọn của tôi |
| Q5 | Có duyệt D6 (lưới thời gian đều khi suy luận) không? | Khuyến nghị: có. Algorithm 1–2 chỉ nêu lưới tăng |
| Q6 | D7: đánh giá bằng EMA làm số chính, kèm trọng số thô? | Khuyến nghị: có (mục 2.3) |
| Q7 | DEV-01: nếu không cài được Adam-atan2 chuẩn, dùng bản tự viết khớp kernel (B)? | Khuyến nghị (B), kèm test số học. Chờ bạn duyệt trong `DEVIATIONS.md` |
| Q8 | "Three repeated tests" [Table 2 caption, tr.9] là ba lần huấn luyện hay ba lần suy luận? | Chưa rõ. Kế hoạch dùng ba lần huấn luyện với seed khác nhau, và ghi rõ |
| Q9 | Dòng GPU | Còn trống (mục 3) |
| Q10 | Quy tắc pilot ở M6 (3,000 batch hoặc 6 giờ; đánh giá mỗi 250 batch trên 100 input) | Là đề xuất. Chờ bạn chỉnh |

---

## 8. Danh sách duyệt cho phiên sau

Tick vào ô để duyệt. Mục không tick thì áp dụng **mặc định** ghi ở cột phải, chỉ khi mặc định đó không chặn mốc nào. Mục "chặn" nghĩa là không thể làm mốc ghi bên cạnh nếu chưa có quyết định.

### 8.1 Bắt buộc trước khi làm các mốc tương ứng

| ☐ | Mục | Chặn mốc | Đề xuất | Chọn của bạn |
|---|---|---|---|---|
| ☐ | **DEV-01**: Adam-atan2 (xem `DEVIATIONS.md`) | M3 trở đi | B nếu A không cài được | A / B / C: ____ |
| ☐ | **Q1**: đầu $q$ đọc từ đâu khi không có ô puzzle-embedding | M3 | mean-pool của $h$ | a / b / c: ____ |
| ☐ | **Q2**: cách gộp batch có halting | M5 | rollout đồng bộ, che mẫu đã dừng | a / b: ____ |
| ☐ | **Q3**: cập nhật mỗi bước denoising (Algorithm 1) hay một lần mỗi rollout (Eq. 9) | M5 | theo Algorithm 1 | ____ |
| ☐ | **Q9**: dòng GPU của máy huấn luyện | M5, M6 | (không có) | ____ |
| ☐ | **Q10**: quy tắc pilot M6 (3,000 batch hoặc 6 giờ, đánh giá mỗi 250 batch, 100 input) | M6 | như đề xuất | ____ |
| ☐ | **D8**: ngân sách huấn luyện đầy đủ | sau pilot M6 | do pilot quyết | ____ |

### 8.2 Có mặc định, duyệt để chốt

| ☐ | Mục | Mặc định nếu không tick |
|---|---|---|
| ☐ | **D4**: ô clue cố định, loss chỉ trên ô không cố định (Q4) | **Chưa dùng**, cần quyết trước M4 (chặn M4) |
| ☐ | **D6**: lưới thời gian đều khi suy luận (Q5) | Đều |
| ☐ | **D7**: đánh giá bằng EMA làm số chính, kèm trọng số thô (Q6) | EMA chính, thô phụ |
| ☐ | **Q8**: "three repeated tests" hiểu là 3 lần huấn luyện khác seed | 3 lần huấn luyện |

### 8.3 Câu hỏi bổ sung (chưa có trong các mục trước)

| ☐ | # | Câu hỏi | Vì sao quan trọng | Đề xuất |
|---|---|---|---|---|
| ☐ | Q11 | Có lưu bản GRAM v4 vào `papers/` không? | Kế hoạch dựa vào GRAM nhưng bản PDF hiện chỉ nằm trong thư mục tạm của phiên và sẽ mất | Có: `papers/2605.19376v4.pdf` |
| ☐ | Q12 | Phiên bản GRAM mà bài chính trích (v1, v2, v3 hay v4)? Nếu cần so từng ô baseline, tải các bản cũ để đối chiếu | Table 2 của bài chính lệch Table 1 của GRAM v4 ở một số ô | Chỉ làm nếu cần số baseline chính xác |
| ☐ | Q13 | Khi báo kết quả, số "chính" cho coverage là cách gộp theo input duy nhất hay theo dòng? (D2 đã duyệt báo cả hai, nhưng chưa nói số nào làm tiêu đề) | Quyết định số nào đặt cạnh 91.4 của bài | Theo input duy nhất |
| ☐ | Q14 | Có chạy thêm hàng ODE ($\gamma=0$) của Table 5 và 7 sau khi có kết quả chính không? | Chi phí thấp, dùng lại mô hình đã huấn luyện, là kiểm tra chéo tốt | Có |
| ☐ | Q15 | Bao nhiêu thời gian GPU tối đa bạn chấp nhận cho 3 seed? | Cần để chốt D8 và quyết có chạy NQ 10×10 hay không | ____ giờ |
| ☐ | Q16 | Môi trường chạy: máy này (RTX 3050 4 GB), server H100, hay cả hai (unit test ở máy này, huấn luyện ở H100)? | Quyết định M0 cài gì và M6 đo gì | Unit test cục bộ, huấn luyện trên GPU lớn hơn |
| ☐ | Q17 | Sau NQ 8×8, ưu tiên NQ 10×10 hay Graph Coloring trước? | Cần plan riêng cho phần tiếp theo | NQ 10×10 (khác ít nhất) |

---

## 9. Cách tiếp tục ở phiên sau

1. Mở `PLAN_TABLE2.md` Mục 8, tick các mục duyệt, gửi lại.
2. Nếu bạn chưa muốn quyết hết, cho phép làm **M0, M1, M2** trước. Ba mốc này không cần GPU và không phụ thuộc Q1–Q3, Q9, DEV-01, D8. M3 trở đi thì bị chặn như ghi ở 8.1.
3. Bắt đầu bằng M0 (môi trường), sau đó M1 theo các giá trị chuẩn trong mục 5.

### Những thứ sẽ **mất** khi phiên này kết thúc

Các file sau nằm trong thư mục tạm của phiên (`/tmp/claude-1000/.../scratchpad/`), không nằm trong repo:

| Thứ cần | Cách lấy lại |
|---|---|
| Mã TRM (để đối chiếu số dòng ở Mục 2) | `git clone` repo `SamsungSAILMontreal/TinyRecursiveModels`, rồi checkout đúng commit `c01103738605ba39d1430519b1ee0c62f4c707f8`. Nếu dùng commit khác, các số dòng ở Mục 2 sẽ lệch |
| Bản PDF GRAM v4 | Tải từ `arxiv.org/pdf/2605.19376`. Nếu tải bản mới hơn v4 thì số trang và số liệu có thể khác. Xem Q11 |
| Mã nguồn `adam-atan2` 0.0.3 và `adam-atan2-pytorch` 0.4.1 (đã đọc để lập DEV-01) | Tải trực tiếp tarball (hoặc wheel) từ trang PyPI của gói rồi giải nén để đọc. Lưu ý: với `adam-atan2`, lệnh `pip download` trên sdist **chạy bước metadata của `setup.py`** (phiên này đã xảy ra, và mã đó chỉ `import torch` trong khối `try`). `adam-atan2-pytorch` có wheel thuần Python nên không có bước này |
| Script liệt kê N-Queens (`nq_check.py`) tạo ra các giá trị chuẩn ở M1 | **Không được lưu.** Các con số chuẩn đã nằm trong bảng M1. Script chỉ ~30 dòng: backtracking ra 92 lời giải; với mỗi lời giải và mỗi tổ hợp $k \in \{5,6,7\}$ quân bị bỏ, tạo một dòng; đếm số input duy nhất và số lời giải tương thích mỗi input. M1 sẽ viết lại |

**Việc đã thực hiện nhưng chưa được duyệt**: không có thay đổi nào ngoài hai file tài liệu (`PLAN_TABLE2.md`, `DEVIATIONS.md`) và `SPEC_TABLE2.md` từ trước. Không cài gói nào, không tạo môi trường.

---

**Điểm dừng:** kế hoạch đã chốt cho phiên này. Việc đầu tiên ở phiên sau là duyệt Mục 8.
