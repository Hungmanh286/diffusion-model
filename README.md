# Diffusion Models & Looped Architectures: Paper Analysis & Slides

Dự án tổng hợp nghiên cứu, phân tích chuyên sâu và tạo slide thuyết trình (LaTeX Beamer) cho các bài báo khoa học tiên phong về **Diffusion/Flow Matching Models** và **Looped Transformers**.

---

## 📁 Cấu trúc thư mục (Repository Structure)

```text
├── papers/                  # Các bài báo nghiên cứu gốc dạng PDF
│   ├── 2602.16813v3.pdf     # Flow Map Language Models (KAIST & CMU)
│   ├── 2610.02383.pdf       # Shared Memory in Looped Transformers (Cornell & IBM)
│   └── lf.pdf               # Looped Flows
│
├── reports/                 # Báo cáo đọc hiểu và kế hoạch trình bày chi tiết (Markdown)
│   ├── 2602.16813_flow_map_language_models.md
│   ├── 2610.02383_shared_memory_looped_transformers.md
│   ├── knowledge_tracing_survey_analysis.md
│   ├── 2602.16813_extract/  # Dữ liệu bóc tách cấu trúc & hình ảnh paper
│   └── 2610.02383_extract/  # Dữ liệu bóc tách cấu trúc & hình ảnh paper
│
├── overleaf/                # Source code slide thuyết trình LaTeX Beamer
│   ├── main.tex             # Slide tổng hợp (kết hợp các bài báo)
│   ├── main.pdf             # File PDF slide hoàn chỉnh đã biên dịch
│   ├── elf_paper.tex        # Slide mô-đun: Embedded Language Flows (ELF)
│   ├── fmlm_paper.tex       # Slide mô-đun: Flow Map Language Models (FMLM)
│   ├── lpt_paper.tex        # Slide mô-đun: Looped Prediction Transformers (LPT)
│   ├── template_paper.tex   # Template mẫu chuẩn để nhân bản khi có bài báo mới
│   ├── fmlm/                # Project slide con độc lập cho FMLM
│   ├── looped transformer/  # Project slide con độc lập cho Looped Transformers
│   └── archive/             # Lưu trữ các bản slide PDF đã xuất trước đây
│
├── assets/                  # Hình ảnh và sơ đồ minh họa dùng chung
│   ├── fmlm.png
│   └── looped.png
│
├── .agents/skills/          # Các kỹ năng tự động hóa của Antigravity AI
│   ├── beamer-paper-slides/ # Chuẩn hóa tạo slide Beamer dạng mô-đun
│   ├── paper-presenter/     # Lập dàn ý thuyết trình từ URL / PDF paper
│   └── paper-analyst/       # Phân tích học thuật chuyên sâu
│
└── .gitignore               # Tự động loại bỏ các file phụ trợ của LaTeX (*.aux, *.log...)
```

---

## 🛠 Hướng dẫn biên dịch slide (Compiling Slides)

### 1. Biên dịch toàn bộ slide trên máy local:
```bash
cd overleaf
xelatex main.tex
```

### 2. Biên dịch từng project con độc lập:
* **FMLM Slide:**
  ```bash
  cd "overleaf/fmlm"
  xelatex main.tex
  ```
* **Looped Transformer Slide:**
  ```bash
  cd "overleaf/looped transformer"
  xelatex main.tex
  ```

