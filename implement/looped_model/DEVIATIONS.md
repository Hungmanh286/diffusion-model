# DEVIATIONS: sai khác so với bài báo khi tái hiện Table 2

Mỗi mục ghi: bài nói gì, ta làm gì khác, lý do, ảnh hưởng, trạng thái duyệt.
Trạng thái: **ĐỀ XUẤT** (chờ duyệt), **ĐÃ DUYỆT**, **TỪ CHỐI**, **THÔNG TIN** (không tránh được, chỉ ghi lại).

---

## DEV-01 — Optimizer Adam-atan2

- **Trạng thái: ĐÃ DUYỆT 31/10/2026**: thử A trước; nếu A không cài hoặc không chạy được thì dùng B ngay, không cần dừng chờ duyệt lại. Ghi kết quả của A (thành công hoặc lỗi nguyên văn) vào mục "Kết quả M0" bên dưới.
- **Bài nói**: dùng Adam-atan2 (Everett et al., 2024), $(\beta_1, \beta_2) = (0.9, 0.95)$, LR đỉnh $10^{-4}$, warmup 2k bước, weight decay theo Table 6 [App. E.1, tr.16]. Bài **không** nêu hằng số $a, b$ của atan2.
- **Tham chiếu thực tế**: TRM dùng gói `adam-atan2==0.0.3` (`specific_requirements.txt:2`; `pretrain.py:20,150`).
- **Điều kiện cài trên server** (kiểm ngày 31/10/2026): gói chỉ có sdist, biên dịch CUDA extension cần `torch` và `nvcc`, hỗ trợ kiến trúc `{80, 86, 89, 90}` (`setup.py:9,21-44`). Server có `nvcc` 13.1 (`/usr/local/cuda/bin/nvcc`), GPU RTX 4090 compute capability 8.9 (nằm trong danh sách), driver 580 (CUDA 13.0). Kho PyTorch có wheel `torch 2.14.1+cu130`. Chưa cài `torch`. Rủi ro còn lại: gói 0.0.3 có thể không biên dịch được với torch 2.14; tác giả gói ghi chú "needs testing on consumer GPUs".
- **Quy tắc cập nhật của kernel** (`csrc/adam_atan2.cu:47-53,77-86`), $a=b=1$:
  - $w \leftarrow w\,(1 - \text{lr}\cdot\text{wd})$
  - $m \leftarrow \text{lerp}(m, g, 1-\beta_1)$, $v \leftarrow \text{lerp}(v, g^2, 1-\beta_2)$
  - $w \leftarrow w - \dfrac{\text{lr}}{1-\beta_1^t}\ \text{atan2}\!\Big(m,\ \dfrac{\sqrt{v}}{\sqrt{1-\beta_2^t}}\Big)$

### Phương án

| | Mô tả | Nhận xét |
|---|---|---|
| **A** | Cài bản CUDA chuẩn `adam-atan2==0.0.3` | **Chọn trước.** Không sai khác |
| **B** | Tự viết bản PyTorch thuần **khớp đúng công thức kernel ở trên**, kèm test số học | **Dự phòng đã duyệt.** Sai khác chỉ ở độ chính xác số học và tốc độ |
| **C** | Gói `adam-atan2-pytorch==0.4.1` nguyên trạng | **Không dùng.** Mặc định `a=1.27`; bias correction khác kernel (đặt $a=1$ vẫn khác ở các bước đầu) |

- **Chênh lệch bước đầu, tính tay từ mã, xác nhận bằng số ở test M3**: với $g > 0$ và bước đầu tiên, $m = 0.1g$, $v = 0.05g^2$, $\text{bc}_1 = 0.1$, $\text{bc}_2 = 0.05$.
  - Kernel: $\dfrac{\text{lr}}{0.1}\,\text{atan2}(0.1g, |g|) = 10\,\text{lr}\cdot\arctan(0.1) \approx 0.9967\ \text{lr}$.
  - Bản (C) với $a=1$: $\text{lr}\cdot\text{atan2}(g, |g|) = \text{lr}\cdot\pi/4 \approx 0.7854\ \text{lr}$.
- **Ảnh hưởng dự kiến** nếu dùng B: không đáng kể; test M3 so B với A trên cùng gradient nếu A cài được.
- **Kết quả M0**: (chưa chạy)

---

## DEV-02 — Tập train/test không trùng GRAM

- **Trạng thái: THÔNG TIN** (không tránh được).
- **Bài nói**: dữ liệu "theo cách dựng của GRAM" [App. E.1, tr.16]. GRAM chia 85:15 theo input duy nhất, ra 7,224 train và 1,240 test [GRAM App. C.2, tr.21]. Hạt giống chia và danh sách input test **không được công bố**; tập N-Queens không có link tải [GRAM Table 17, tr.32].
- **Ta làm**: tự sinh dữ liệu theo mô tả (8,464 dòng, 5,148 input duy nhất) và chia theo input với một hạt giống cố định (D10). Số dòng train và test có thể lệch nhẹ khỏi 7,224 và 1,240.
- **Ảnh hưởng**: so sánh số tuyệt đối với baseline chỉ ở mức gần đúng; thứ hạng giữa các phương pháp vẫn có ý nghĩa nếu cùng tập test.
- **Duyệt**: không cần.
- **Thực tế đo được (31/10/2026, M1)**: với `split_seed=0`, chia ra **7,248 train / 1,216 test** dòng (14.37%), so với 7,224 / 1,240 (14.65%) của GRAM.

---

## DEV-03 — Các lựa chọn lấp chỗ bài không nêu

- **Trạng thái: ĐÃ DUYỆT 31/10/2026.** Đây không phải sai khác với điều bài nói, mà là lựa chọn ở chỗ bài im lặng. Ghi lại để khi báo kết quả nói rõ.

| Mục | Bài nói | Ta chọn | Cơ sở |
|---|---|---|---|
| Q1, đầu $q$ | Không nêu vị trí đọc | Đọc $h$ ở **vị trí 0** (ô bàn cờ đầu tiên) | Hành vi gốc của mã TRM khi không có puzzle embedding: `q_head(z_H[:, 0])`, `trm.py:221`; bài chỉ dùng puzzle embedding cho ARC [App. E.1, tr.16] |
| Q2, gộp batch có halting | Algorithm 1 viết cho một mẫu | **Kiểu TRM**: mỗi vị trí batch mang rollout riêng, thay mẫu mới khi dừng | "One denoiser call uses the computation of a single TRM supervision step. We replace TRM's outer supervision loop…" [App. A, tr.14]; vòng ngoài TRM: `trm.py:251-297` |
| D4, ô clue | "keep these values fixed and construct the interpolant over the remaining entries" [App. A, tr.14] | Ô cố định = **chỉ** các ô có quân clue (1–3 ô). Cố định ở interpolant, ở bước thêm nhiễu (Eq. 18) và bước Euler (Eq. 19). CE và nhãn ACT chỉ tính trên ô còn lại | Phần loss và nhãn ACT là lựa chọn của ta |
| D6, lưới suy luận | Chỉ nêu $0 = t_0 < \dots < t_n = 1$ | Lưới đều | |
| D7, trọng số đánh giá | Chỉ nêu "parameter EMA with decay 0.999" | EMA là số chính, kèm trọng số thô | TRM: `pretrain.py:619-625` |
| Q8, "three repeated tests" | Không rõ | 3 lần huấn luyện, seed 0, 1, 2 | Table 1 dùng "three seeds" [tr.7] |
| Số học | "bfloat16 precision for forward passes" [App. E.1, tr.16] | Forward bf16; tham số và trạng thái optimizer float32; StableMax CE float64 | TRM: `forward_dtype: bfloat16`, `losses.py:25` |
