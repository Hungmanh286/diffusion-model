# DEVIATIONS: sai khác so với bài báo khi tái hiện Table 2

Mỗi mục ghi: bài nói gì, ta làm gì khác, lý do, ảnh hưởng, trạng thái duyệt.
Trạng thái: **ĐỀ XUẤT** (chờ duyệt), **ĐÃ DUYỆT**, **TỪ CHỐI**, **THÔNG TIN** (không tránh được, chỉ ghi lại).

---

## DEV-01 — Optimizer Adam-atan2

- **Trạng thái: ĐỀ XUẤT, chờ duyệt.**
- **Bài nói**: dùng Adam-atan2 (Everett et al., 2024), $(\beta_1, \beta_2) = (0.9, 0.95)$, LR đỉnh $10^{-4}$, warmup 2k bước, weight decay theo Table 6 [App. E.1, tr.16]. Bài **không** nêu hằng số $a, b$ của atan2.
- **Tham chiếu thực tế**: TRM dùng gói `adam-atan2==0.0.3` (`specific_requirements.txt:2`; `pretrain.py:20,150`).
- **Vấn đề**: gói chỉ có sdist, biên dịch CUDA extension cần `torch` và `nvcc`, hỗ trợ kiến trúc `{80, 86, 89, 90}` (`setup.py:9,21-44`). Máy này hiện **chưa có `torch` và `nvcc`**, nên **chưa cài được**. Chưa thử trên máy khác (ví dụ H100).
- **Quy tắc cập nhật của kernel** (`csrc/adam_atan2.cu:47-53,77-86`), $a=b=1$:
  - $w \leftarrow w\,(1 - \text{lr}\cdot\text{wd})$
  - $m \leftarrow \text{lerp}(m, g, 1-\beta_1)$, $v \leftarrow \text{lerp}(v, g^2, 1-\beta_2)$
  - $w \leftarrow w - \dfrac{\text{lr}}{1-\beta_1^t}\ \text{atan2}\!\Big(m,\ \dfrac{\sqrt{v}}{\sqrt{1-\beta_2^t}}\Big)$

### Phương án

| | Mô tả | Nhận xét |
|---|---|---|
| **A** | Cài bản CUDA chuẩn khi có `torch` và CUDA toolkit | Không sai khác. GPU 8.6 (RTX 3050) và 9.0 (H100) nằm trong kiến trúc hỗ trợ; tác giả ghi chú còn "needs testing on consumer GPUs" |
| **B** | Tự viết bản PyTorch thuần **khớp đúng công thức kernel ở trên**, kèm test số học | **Khuyến nghị nếu A không cài được.** Sai khác chỉ ở độ chính xác số học và tốc độ |
| **C** | Gói `adam-atan2-pytorch==0.4.1` nguyên trạng | **Không khuyến nghị.** Mặc định `a=1.27`; bias correction khác kernel (đặt $a=1$ vẫn khác ở các bước đầu) |

- **Chênh lệch bước đầu, tính tay từ mã, cần xác nhận bằng số ở test M3**: với $g > 0$ và bước đầu tiên, $m = 0.1g$, $v = 0.05g^2$, $\text{bc}_1 = 0.1$, $\text{bc}_2 = 0.05$.
  - Kernel: $\dfrac{\text{lr}}{0.1}\,\text{atan2}(0.1g, |g|) = 10\,\text{lr}\cdot\arctan(0.1) \approx 0.9967\ \text{lr}$.
  - Bản (C) với $a=1$: $\text{lr}\cdot\text{atan2}(g, |g|) = \text{lr}\cdot\pi/4 \approx 0.7854\ \text{lr}$.
  - Hai bản chỉ gần nhau khi $m/(\text{bc}_1 \cdot \text{denom})$ nhỏ, tức sau vài chục bước đầu.
- **Ảnh hưởng dự kiến**: nhỏ đến vừa ở đoạn khởi động, không đáng kể về sau; chưa được đo.
- **Duyệt**: ☐ A ☐ B ☐ C ☐ Khác: ________

---

## DEV-02 — Tập train/test không trùng GRAM

- **Trạng thái: THÔNG TIN** (không tránh được).
- **Bài nói**: dữ liệu "theo cách dựng của GRAM" [App. E.1, tr.16]. GRAM chia 85:15 theo input duy nhất, ra 7,224 train và 1,240 test [GRAM App. C.2, tr.21]. Hạt giống chia và danh sách input test **không được công bố**; tập N-Queens không có link tải [GRAM Table 17, tr.32].
- **Ta làm**: tự sinh dữ liệu theo mô tả (8,464 dòng, 5,148 input duy nhất) và chia theo input với một hạt giống cố định (D10). Số dòng train và test có thể lệch nhẹ khỏi 7,224 và 1,240.
- **Ảnh hưởng**: so sánh số tuyệt đối với baseline chỉ ở mức gần đúng; thứ hạng giữa các phương pháp vẫn có ý nghĩa nếu cùng tập test.
- **Duyệt**: không cần.
