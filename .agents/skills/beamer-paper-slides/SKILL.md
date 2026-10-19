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

- **Tuyệt đối không viết cặp từ song song Anh-Việt:** Tránh viết dạng `bộ khử nhiễu (Denoiser)` hay `lỗi phân rã (Factorization Error)`.
- **Xử lý thuật ngữ chuyên ngành:** 
  - Giữ nguyên tiếng Anh cho các thuật ngữ chuẩn quốc tế: *autoregressive LLMs, discrete diffusion, continuous flows, denoiser, flow map, simplex, cross-entropy, few-step regime, inference-time guidance, look-ahead, ablation study*.
  - Dịch sang tiếng Việt tự nhiên cho các khái niệm thông dụng: *bối cảnh, thách thức, cơ chế huấn luyện, kết quả thực nghiệm, tính toán song song, hàm mục tiêu*.
- **Bố cục slide:**
  - Dùng gạch đầu dòng phân cấp (`\begin{itemize}` với `\textbf{...}` cho tiêu đề phụ).
  - Công thức toán viết ngắn gọn, rõ ràng với môi trường `$$...$$` hoặc `\begin{equation*}`.
  - Mỗi slide chỉ chứa 1 thông điệp chính, tránh dồn quá nhiều chữ.

---

## 3. Cấu Trúc Khung Chuẩn Một Bài Báo (6 - 8 Slides)

Mỗi bài báo nên được tóm tắt trong khoảng 6 đến 8 slide:

1. **Slide 0: Tổng quan bài báo (Overview):** Tên bài báo, tác giả/lab, ý tưởng cốt lõi, 3 đóng góp chính.
2. **Slide 1: Bối cảnh & Thách thức (Motivation & Problem):** Hạn chế của các phương pháp cũ (SOTA baselines), lý do bài toán khó.
3. **Slide 2: Ý tưởng đột phá / Khung làm việc (Core Idea / Framework):** Điểm khác biệt mấu chốt, mô hình hóa tổng quan.
4. **Slide 3: Phương pháp luận toán học (Theoretical Methodology):** Định nghĩa, công thức toán then chốt, tính chất cốt lõi.
5. **Slide 4: Hàm mục tiêu & Huấn luyện (Training & Loss Objectives):** Hàm loss, kỹ thuật lấy mẫu, tái tham số hóa.
6. **Slide 5: Cơ chế suy luận / Dẫn hướng (Inference & Guidance):** Cách lấy mẫu, thuật toán sinh ít bước, steering/guidance (nếu có).
7. **Slide 6: Kết quả thực nghiệm (Experimental Results):** So sánh với baselines qua bảng số liệu hoặc gạch đầu dòng then chốt (PPL, tốc độ, độ chính xác).
8. **Slide 7: Nghiên cứu triệt tiêu & Kết luận (Ablations & Takeaways):** Kiểm chứng các quyết định thiết kế (ablation study), tóm tắt bài học và hạn chế.

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

