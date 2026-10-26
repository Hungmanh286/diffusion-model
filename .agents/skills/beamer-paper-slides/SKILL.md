---
name: beamer-paper-slides
description: >-
  Hệ thống quy chuẩn và template mô-đun hóa để tạo slide báo cáo bài báo nghiên cứu
  khoa học (AI/ML) bằng LaTeX Beamer, cho phép tạo các file slide bài báo riêng lẻ
  và nhập (import qua \input) vào bài thuyết trình chính (main.tex).
---

# Beamer Paper Slides: Chuẩn Hóa Slide Báo Cáo Bài Báo AI/ML

Kỹ năng này hướng dẫn quy trình, cấu trúc và phong cách chuẩn để tóm tắt một bài báo khoa học thành các slide thuyết trình LaTeX Beamer dạng mô-đun hóa, giúp bạn hoặc trợ lý AI có thể dễ dàng tạo file bài báo mới và nhập vào `main.tex`.

---

## 1. Kiến Trúc Mô-đun Hóa (Modular Architecture)

Mỗi buổi báo cáo thường bao gồm nhiều bài báo hoặc nhiều chủ đề. Cấu trúc project chuẩn bao gồm:

```text
project-folder/
├── main.tex                 # File chính: chứa preamble, cấu hình theme, mục lục, trang kết luận
├── elf_paper.tex            # Slide bài báo 1 (ELF)
├── fmlm_paper.tex           # Slide bài báo 2 (FMLM)
├── <new_paper>_paper.tex    # Slide bài báo mới tiếp theo
└── template_paper.tex       # Template mẫu để nhân bản khi có bài báo mới
```

### Nguyên tắc bất di bất dịch của file con:
1. **KHÔNG** chứa `\documentclass`, `\usepackage`, hay bất kỳ khai báo preamble nào.
2. **KHÔNG** chứa `\begin{document}` và `\end{document}`.
3. **KHÔNG** chứa các lệnh trang bìa độc lập (`\title`, `\author`, `\titlepage`). Thay vào đó, bắt đầu bằng `\section{Tên Bài Báo}` để tích hợp mượt mà vào mục lục của `main.tex`.

---

## 2. Quy Chuẩn Văn Phong & Thuật Ngữ

### 2.1 Quy tắc bắt buộc: không mở ngoặc, dịch thuật ngữ sang tiếng Việt

1. **Tuyệt đối không dùng ngoặc đơn để kèm tiếng Anh** trong tiêu đề slide, tiêu đề khối, gạch đầu dòng, ô bảng. Ví dụ sai: `Đại lộ Gradient (Gradient Highway)`, `bộ khử nhiễu (denoiser)`. Ví dụ đúng: `Đại lộ gradient`, `bộ khử nhiễu`.
2. **Dịch mọi thuật ngữ chuyên ngành của bài báo sang tiếng Việt** theo bảng ở mục 2.3, kể cả tên cơ chế do bài báo tự đặt (state recursion, shared memory, gradient highway, two-time denoiser...). Chỉ giữ tiếng Anh cho các loại ở mục 2.2.
3. **Dùng một cách dịch duy nhất xuyên suốt** một file và giữa các file. Thuật ngữ mới chưa có trong bảng: chọn bản dịch ngắn, tự nhiên, rồi **thêm vào bảng 2.3 của skill này** để lần sau dùng lại.
4. **Không giải thích thuật ngữ bằng cách chèn tiếng Anh vào ngoặc.** Nếu người nghe cần nhìn nguyên văn, đưa vào một slide thuật ngữ riêng hoặc vào ghi chú, không đặt cạnh bản dịch.
5. Ngoặc đơn vẫn dùng cho nội dung toán học (như `$f(x)$`), cho số liệu phụ (như `3,6 đến 5,0 lần`), và cho viết tắt tên mô hình khi định nghĩa lần đầu ở slide tổng quan (như `Transformer vòng lặp dự đoán, gọi tắt LPT`, viết bằng dấu phẩy chứ không bằng ngoặc).

### 2.2 Những gì được giữ nguyên tiếng Anh

- **Tên riêng:** tên bài báo, tên mô hình và phương pháp (LPT, FMLM, FLM, TRM, MDLM, Duo, Looped Flows), tên bộ dữ liệu và benchmark (FineWeb-Edu, LM1B, OWT, Sudoku-Extreme, ARC-AGI-1), tên tác giả, tên tổ chức, tên phần cứng (H100).
- **Thuật ngữ đã Việt hóa trong cộng đồng AI:** Transformer, gradient, token, softmax, KV cache, batch, GPU, FLOPs, entropy, cross-entropy, KL, one-hot, argmax, Gaussian, Euler, ODE, SDE, PCA, EMA, Sudoku.
- **Ký hiệu toán và viết tắt chỉ số:** PPL (đứng riêng, không kèm ngoặc), Gen. PPL, Acc, R, $\omega$, $\delta_{s,t}$.

### 2.3 Bảng thuật ngữ chuẩn (Anh sang Việt)

| Tiếng Anh trong bài báo | Dùng trong slide |
|---|---|
| autoregressive model | mô hình tự hồi quy |
| discrete diffusion | khuếch tán rời rạc |
| continuous flow | dòng liên tục |
| flow matching | khớp dòng |
| flow map | ánh xạ dòng |
| denoiser | bộ khử nhiễu |
| two-time denoiser | bộ khử nhiễu hai thời điểm |
| stateful denoiser | bộ khử nhiễu có trạng thái |
| simplex | đơn hình xác suất |
| factorization error | lỗi phân rã nhân tử |
| semigroup condition | điều kiện nửa nhóm |
| distillation | chưng cất |
| one-step / few-step generation | sinh một bước / sinh ít bước |
| look-ahead | nhìn trước điểm cuối |
| guidance, autoguidance, reward guidance | dẫn hướng, tự dẫn hướng, dẫn hướng theo phần thưởng |
| time reparameterization | tái tham số hóa thời gian |
| boundary sampling | lấy mẫu biên |
| decoding error rate | tỉ lệ lỗi giải mã |
| mode collapse | sụp đổ mode |
| looped model / looped transformer | mô hình vòng lặp / Transformer vòng lặp |
| recursion | vòng lặp |
| state recursion | vòng lặp trạng thái |
| prediction recursion | vòng lặp dự đoán |
| shared memory | bộ nhớ dùng chung |
| local window / sliding window | cửa sổ cục bộ / cửa sổ trượt |
| gradient highway | đại lộ gradient |
| detach | cắt gradient |
| inductive bias | thiên kiến quy nạp |
| context memory | bộ nhớ ngữ cảnh |
| hybrid model | mô hình lai |
| linear attention | chú ý tuyến tính |
| hidden state / recurrent state | trạng thái ẩn / trạng thái lặp |
| stop-gradient | chặn gradient |
| temporal alignment | căn chỉnh theo thời gian |
| shared noise | nhiễu dùng chung |
| decreasing noise | nhiễu giảm dần |
| adaptive computation time | thời gian tính toán thích ứng |
| inference-time scaling | mở rộng tính toán lúc suy luận |
| best-Q ensembling | tổ hợp chọn mẫu theo điểm Q cao nhất |
| pseudotarget | mục tiêu giả |
| spurious attractor | điểm hút giả |
| failure-to-converge | không hội tụ |
| coverage | độ phủ lời giải |
| multi-solution | đa lời giải |
| ablation study | nghiên cứu loại bỏ thành phần |
| baseline | phương pháp đối chứng |
| state-of-the-art | tốt nhất hiện nay |
| perplexity | độ rối |
| throughput | thông lượng |
| batch capacity | sức chứa batch |
| inference, training | suy luận, huấn luyện |
| loss function | hàm mất mát |
| objective | hàm mục tiêu |
| bottleneck | nút thắt |
| zero-shot accuracy | độ chính xác zero-shot |

### 2.4 Bố cục slide

- Dùng gạch đầu dòng phân cấp (`\begin{itemize}` với `\textbf{...}` cho tiêu đề phụ).
- Công thức toán viết ngắn gọn với `$$...$$` hoặc `\begin{equation*}`.
- Mỗi slide chỉ chứa 1 thông điệp chính, tránh dồn quá nhiều chữ.
- **Số liệu phải khớp báo cáo** trong `reports/` của đúng bài báo. Không viết "nats" cho độ rối PPL, không viết "hoàn toàn" khi bài báo chỉ chứng minh một phần. Giữ nguyên các phản biện quan trọng trong báo cáo (ví dụ entropy thấp hơn dữ liệu thật, gap công bằng của LPT so với LT).
- Slide nên có một slide cuối về **hạn chế và phản biện**, không chỉ ca ngợi.

---

## 3. Cấu Trúc Khung Chuẩn Một Bài Báo (8 - 12 Slides)

Mỗi bài báo nên được tóm tắt trong khoảng 8 đến 12 slide. Tách riêng một slide kết quả cho mỗi nhóm bảng số liệu và luôn có slide cuối về hạn chế và phản biện:

1. **Slide 0: Tổng quan bài báo:** Tên bài báo, tác giả/lab, ý tưởng cốt lõi, 3 đóng góp chính.
2. **Slide 1: Bối cảnh và thách thức:** Hạn chế của các phương pháp cũ, lý do bài toán khó.
3. **Slide 2: Ý tưởng đột phá và khung làm việc:** Điểm khác biệt mấu chốt, mô hình hóa tổng quan.
4. **Slide 3: Phương pháp luận toán học:** Định nghĩa, công thức toán then chốt, tính chất cốt lõi.
5. **Slide 4: Hàm mục tiêu và huấn luyện:** Hàm mất mát, kỹ thuật lấy mẫu, tái tham số hóa.
6. **Slide 5: Cơ chế suy luận và dẫn hướng:** Cách lấy mẫu, thuật toán sinh ít bước, dẫn hướng (nếu có).
7. **Slide 6: Kết quả thực nghiệm:** So sánh với các phương pháp đối chứng qua bảng số liệu hoặc gạch đầu dòng then chốt (PPL, tốc độ, độ chính xác).
8. **Slide 7: Nghiên cứu loại bỏ thành phần:** Kiểm chứng các quyết định thiết kế.
9. **Slide cuối: Hạn chế, phản biện và kết luận:** Giả định ngầm, điểm thiết lập chưa công bằng, quy mô còn nhỏ, bài học chính.

---

## 4. Hướng Dẫn Thêm Bài Báo Mới

Khi muốn thêm bài báo mới vào bài thuyết trình:
1. Nhân bản file mẫu:
   ```bash
   cp template_paper.tex <ten_bai_bao>_paper.tex
   ```
2. Điền nội dung vào `<ten_bai_bao>_paper.tex` theo mẫu chuẩn.
3. Mở `main.tex` và thêm dòng import:
   ```latex
   \input{<ten_bai_bao>_paper.tex}
   ```
4. Biên dịch lại `main.tex`.

