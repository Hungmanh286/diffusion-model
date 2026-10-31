# Báo cáo chạy thử S1 trên laptop và các sửa đổi sau đó

Ngày: 31/10/2026. Máy: laptop RTX 3050 Laptop 4 GB, i5-12500H, RAM 15 GB,
torch `2.14.1+cu130`.

> **Phạm vi:** đây là **smoke test**, không phải số liệu cho Table 2. Cấu hình
> khác bài báo ở hai điểm: batch 64 thay vì 768 [App. E.1, tr.16]; forward
> float32 thay vì bfloat16 (lúc chạy S1, mã chưa có bf16; xem Mục 5).

---

## 1. Cấu hình S1

| Mục | Giá trị | Nguồn |
|---|---|---|
| Config | `configs/nq8_laptop.yaml`: giống `nq8.yaml`, chỉ đổi `optim.batch_size: 64` | `diff` với `nq8.yaml` |
| Lý do chọn batch | B=64 dùng 1.99 GB VRAM; B=128 OOM (fp32) | đo trước khi chạy, Mục 5.3 |
| Số bước | 2,000 (vừa hết warmup, lr chạm 1e-4) | `optim.warmup_steps: 2000` |
| Forward | float32 (mã lúc đó chưa có bf16) | `lf/train.py` trước khi sửa |
| Lệnh | `timeout 40m nice -n 10 python -m lf.train --config configs/nq8_laptop.yaml --device cuda --steps 2000 --out runs/laptop_s1 --ckpt-every 250 --log-every 50` | |
| Thời gian | 15:47:00 → 16:05:45 (18'45"), exit code 0, 1.78–1.85 it/s | `runs/laptop_s1/train.log` |
| Checkpoint | 9 file × 130 MB (bước 250…2000 và `ckpt_final.pt`) | `runs/laptop_s1/` |

## 2. An toàn máy

Watchdog kiểm mỗi 5 giây (log: `runs/laptop_s1/watchdog.log`, 219 mẫu). Nếu vượt
ngưỡng thì gửi SIGTERM, 10 giây sau gửi SIGKILL.

| Chỉ số | Cao nhất | Ngưỡng dừng | Ghi chú |
|---|---|---|---|
| GPU | **84°C** | 90°C ngay, hoặc 85°C liên tục 30 giây | 41/219 mẫu ở mức ≥ 84°C (khoảng 3.4 phút) |
| CPU (x86_pkg) | **94°C** | 95°C liên tục 30 giây | khoảng 92°C suốt phần lớn thời gian |
| Công suất GPU | khoảng 60 W | – | chạm trần ở mọi batch, kể cả B=32 |
| VRAM | 2.3 / 4 GB | – | |
| RAM trống | ≥ 9.7 GB | 1.5 GB | |

Watchdog không phải dừng lần nào. **Cả GPU và CPU chỉ cách ngưỡng khoảng 1°C.**
Chạy khoảng 20 phút thì chịu được. Chạy dài hơn thì cần làm mát thêm (Mục 7).

Nhận xét: CPU nóng vì tiến trình python chiếm 100% một nhân (để phát lệnh CUDA),
và trên laptop CPU với GPU dùng chung tản nhiệt.

## 3. Kết quả huấn luyện

### 3.1 Log từng bước bị nhiễu theo pha rollout

Log của S1 in số liệu của **một bước**, không phải trung bình (`lf/train.py`
trước khi sửa). Lúc đầu mọi carry chạy đồng bộ 16 bước (`flow.k_steps: 16`) nên
số liệu dao động theo pha. Bằng chứng: ở bước 400 và 800 (chia hết cho 16) có
`halt 1.00`, `lm` khoảng 0.035, `q` cao (4.13 và 1.19). Từ khoảng bước 1000,
`halt` > 0 làm các carry lệch pha nhau, nên dao động giảm.
*(Suy luận từ mẫu số liệu; khớp với cách `_init_carry` ép reset toàn batch ở
bước đầu, `lf/train.py` `c_halted = ones`.)*

### 3.2 So các bước cùng pha (bước chia 16 dư 8)

| Bước | 200 | 600 | 1000 | 1400 | 1800 |
|---|---|---|---|---|---|
| `lm` | 0.251 | 0.241 | 0.219 | 0.185 | 0.191 |

`lm` giảm chậm, đúng với việc lr vẫn đang warmup. `halt` cuối khoảng 0.08. Nếu
mọi carry đều chạy đủ 16 bước thì `halt` bằng 1/16 ≈ 0.0625, nên con số này cho
thấy đã có một ít carry dừng sớm. Không có NaN hay Inf, không OOM.

### 3.3 Độ chính xác dự đoán một bước (`Trainer.one_step_accuracy`)

Đo trên `ckpt_final.pt` (bước 2000). Đây là một lần gọi denoiser từ trạng thái
khởi tạo tại t cố định, và một chuỗi chỉ tính là đúng khi **mọi** ô không phải
clue đều đúng. Dùng 1,216 dòng test và 1,216 dòng train lấy ngẫu nhiên.
Baseline là lấy argmax thẳng của $I_t$, không qua mô hình, trên cùng dòng test
(trung bình 5 lần lấy nhiễu).

| t | Test EMA | Test raw | Train EMA | Train raw | Baseline argmax($I_t$) |
|---|---|---|---|---|---|
| 0.5 | 0.0000 | 0.0033 | 0.0000 | 0.0082 | 0.0000 |
| **0.8** | **0.9383** | **0.9671** | 0.9548 | 0.9679 | **0.7602** |
| 0.9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.99 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Cách đọc:
- Ở t ≥ 0.9, chỉ cần lấy argmax của $I_t$ đã đúng 100%, nên con số này **không
  nói lên gì** về mô hình.
- Ở t = 0.8, mô hình vượt baseline rõ (0.94–0.97 so với 0.76). Nghĩa là mô hình
  đã học được nhiều hơn việc chép lại tín hiệu trong $I_t$.
- Train và test gần như bằng nhau, chưa thấy overfit.
- EMA thấp hơn trọng số thô. Điều này hợp lý: decay 0.999 lấy trung bình trên
  khoảng 1,000 bước gần nhất, trong khi mô hình còn đang thay đổi nhanh.
- Ở t = 0.5 gần như bằng 0. Lời giải đầy đủ cần cả sampler 128 bước, và việc đó
  chưa được đo.

**Kết luận S1:** pipeline huấn luyện chạy được trên GPU từ đầu đến cuối và mô
hình đã học được chút ít. Chưa nói được gì về coverage hay accuracy của Table 2.

## 4. Lỗi tìm thấy và đã sửa: resume trên GPU

`Trainer.load` dùng `torch.load(..., map_location=self.device)`, làm RNG state
(ByteTensor) bị đưa lên GPU, rồi `torch.set_rng_state` báo
`TypeError: RNG state must be a torch.ByteTensor`. Như vậy **`--resume` trên GPU
chưa bao giờ chạy được.** Test cũ chỉ chạy trên CPU nên không bắt được.

- Sửa: `lf/train.py` `load()` gọi `.cpu()` cho `torch_rng` và `gen_rng`.
- Test mới: `tests/test_train.py::test_resume_on_cuda`.
- Kiểm thật qua CLI: chạy 100 bước, rồi `--resume ckpt_100.pt` chạy tiếp đến bước
  150 trên cuda, kết quả đạt (`runs/laptop_cli_check/`).

## 5. Các sửa đổi theo yêu cầu (mục 1, 3, 4)

### 5.1 Mục 1: đánh giá `one_step_accuracy`

Đã làm, kết quả ở Mục 3.3. Không đổi mã đánh giá. Chỉ chia batch 256 để giữ VRAM
thấp (đỉnh 844 MiB, mất 96 giây).

### 5.2 Mục 3: log trung bình theo cửa sổ (`lf/train.py`, `main()`)

- `loss`, `lm`, `q`, `halt` giờ là **trung bình** trên `--log-every` bước. `lr`
  là giá trị của bước cuối.
- Dòng đầu log ghi rõ: `log: trung bình cửa sổ N bước; forward dtype ...`.
- Sửa thêm: trước đây it/s tính bằng `step / thời gian`, nên sai khi `--resume`
  vì `step` gồm cả số bước của lần chạy trước. Giờ tính bằng
  `(step - step_lúc_bắt_đầu) / thời gian`.
- `print(..., flush=True)` để `tail -f` thấy log ngay.
- Định dạng dòng giữ nguyên (`step N loss … lm … q … lr … halt …`), nên lệnh
  grep cũ vẫn dùng được.

### 5.3 Mục 4: forward bfloat16 (DEV-03)

Lệch đã sửa: `nq8.yaml` (`precision.forward: bfloat16`) và `DEVIATIONS.md:58` ghi
forward chạy bf16, nhưng mã cũ chạy toàn bộ bằng float32.

Cách làm theo TRM, không dùng `torch.autocast`:

| Thành phần | Cách làm | Nguồn TRM |
|---|---|---|
| Tham số | giữ float32 | `ref/models/layers.py:51-57` |
| Linear | ép trọng số về dtype của input (lớp `Linear` của ta đã làm vậy từ trước) | `layers.py:59-60` |
| Trạng thái z (h, l) | `initial_state` trả ra `forward_dtype`; carry lưu ở bf16 | `trm.py:153-154`, `trm.py:183-186` |
| Đầu vào | interpolant $I_t$ tính ở float32, rồi ép sang `forward_dtype` khi vào `_injection` | TRM nhận token nên không có bước này; **đây là lựa chọn của ta** |
| RMSNorm, RoPE | tính bằng float32 bên trong rồi trả lại dtype gốc (giữ nguyên như cũ) | `layers.py:163-169` |
| Logits và CE | logits ở bf16; StableMax CE tự đổi sang float64 (`lf/flow.py`, `stablemax_cross_entropy`) | `losses.py:25` |
| Đầu q | `.to(float32)` (giữ nguyên như cũ) | |

Mã:
- `lf/model.py`: thêm tham số `forward_dtype`; ép kiểu trong `initial_state`,
  `_injection` và đầu `forward`. State float32 từ checkpoint cũ vẫn nạp được vì
  được ép kiểu khi vào.
- `lf/train.py`: thêm `TrainConfig.forward_dtype`, mặc định `"float32"` để test cũ
  giữ nguyên hành vi; `from_yaml` đọc `precision.forward`.
- Vì vậy mọi lần chạy dùng `nq8.yaml` hoặc `nq8_laptop.yaml` **từ giờ đều chạy
  bf16**.

Đo trên laptop (10 bước, cùng cấu hình 7M):

| Batch | dtype | VRAM đỉnh | ms/bước |
|---|---|---|---|
| 64 | float32 | 1,988 MiB | 548 |
| 64 | bfloat16 | 1,255 MiB | **262** (nhanh khoảng 2.1 lần) |
| 128 | bfloat16 | 2,249 MiB | 513 |
| 192 | bfloat16 | 3,257 MiB | 756 |

Sai số so với float32 cùng trọng số, đo lúc khởi tạo ngẫu nhiên: sai tương đối
2–7% trên logits, argmax trùng 97–100%. Sai số cộng dồn qua 15 lần gọi
$F_\theta$ ($3$ chu trình $\times (m+1)$, $m=4$).

### 5.4 Test

`pytest tests`: **62/62 đạt** (58 test cũ cộng 4 test mới):
- `test_model.py::test_bf16_forward_dtype`: dtype của state, logits và tham số;
  nạp được state float32; sai tương đối so với fp32 < 15%.
- `test_train.py::test_bf16_train_steps`: 30 bước bf16, loss hữu hạn và giảm,
  carry ở bf16, tham số ở fp32.
- `test_train.py::test_from_yaml_reads_precision`
- `test_train.py::test_resume_on_cuda`

Lúc đầu test bf16 dùng ngưỡng sai tuyệt đối 0.25 do tao đặt tùy tiện, và nó fail
(sai tối đa 0.58). Tao đã đổi sang ngưỡng sai tương đối dựa trên số đo ở trên,
không phải nới ngưỡng cho test qua.

## 6. File thay đổi

| File | Trạng thái |
|---|---|
| `configs/nq8_laptop.yaml` | mới |
| `lf/model.py` | sửa (bf16) |
| `lf/train.py` | sửa (bf16, log theo cửa sổ, lỗi resume trên GPU) |
| `tests/test_model.py`, `tests/test_train.py` | thêm test; `__main__` của `test_train.py` giờ tự chạy mọi `test_*` |
| `REPORT_LAPTOP_S1.md` | mới (file này) |
| `runs/laptop_s1/`, `runs/laptop_cli_check/` | output, đã gitignore |

Chưa commit.

## 7. Việc cần làm tiếp

**Cần người dùng quyết:**
1. **Commit** các thay đổi trên (chưa commit).
2. **Có ghi vào `DEVIATIONS.md` không:** (a) S1 chạy batch 64 và fp32, chỉ là
   smoke test; (b) interpolant được ép float32 → bf16 khi vào model (Mục 5.3),
   chi tiết TRM không có.
3. **Có chạy lại S1 bằng bf16 trên laptop không.** Lý do: S1 cũ chạy fp32 nên
   không so được với các lần chạy sau. Ước tính: B=64 2000 bước mất khoảng 9 phút;
   B=128 mất khoảng 17 phút. Chưa đo bf16 có làm GPU/CPU mát hơn không (công suất
   có thể vẫn chạm trần 60 W).
4. **Muốn chạy dài hơn trên laptop thì phải làm mát thêm:** kê đế tản nhiệt, hoặc
   khóa xung bằng `sudo nvidia-smi -lgc 0,1200` (chưa thử, cần sudo).
5. **Dọn checkpoint:** `runs/laptop_s1/` chiếm khoảng 1.2 GB; có thể chỉ giữ
   `ckpt_final.pt`.

**Kỹ thuật, chưa làm:**
6. Đánh giá bằng sampler đầy đủ (Algorithm 2, 128 bước, γ=5) trên một tập test
   nhỏ, dùng `lf/eval_table2.py::evaluate_model`. Hiện chưa có CLI cho việc này.
7. Tùy chọn: ghi số liệu từng bước ra CSV để phân tích theo pha mà không làm
   nhiễu log.
8. Phần còn lại trên server (PLAN_TABLE2.md): M0 (venv cu130, thử adam-atan2 phương
   án A), chạy M5 thật với batch 768 bf16, pilot P0/P1 để chốt D8, 3 seed, M7.
   Ghi chú: với bf16, batch 768 có thể cần ít VRAM hơn nhiều so với ước lượng cũ.
   Suy từ số đo laptop (khoảng 15.5 MiB/mẫu), batch 768 cần khoảng 12 GB. Đây
   là ngoại suy, chưa đo trên server.
