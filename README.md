# Diffusion Models & Looped Architectures: Paper Analysis, Slides & Reproduction

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX_Beamer-008080?logo=latex&logoColor=white)
![Status](https://img.shields.io/badge/status-in_progress-orange)

> **EN:** Research notes, Beamer slides, and a from-scratch PyTorch reproduction of *Thinking with Looped Flows* (Table 2, N-Queens 8×8) — flow matching on a looped transformer, with a 58-test pytest suite and documented deviations from the paper.

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
├── implement/looped_model/  # Tái hiện Table 2 của *Thinking with Looped Flows* (PyTorch)
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

---

## 🧪 Tái hiện *Thinking with Looped Flows* — Table 2 (N-Queens 8×8)

Mã nguồn tại [`implement/looped_model/`](implement/looped_model/). **Đang thực hiện:** mã đã viết và kiểm trên CPU, chưa có kết quả huấn luyện đầy đủ.

| Mô-đun | Nội dung |
|---|---|
| `lf/data_nqueens.py` | Sinh dữ liệu N-Queens, chia train/test theo seed |
| `lf/model.py`, `lf/optim.py` | Backbone looped transformer (6,815,744 tham số), Adam-atan2 |
| `lf/flow.py`, `lf/sampler.py` | Interpolant flow matching, StableMax CE, sampler (Algorithm 2) |
| `lf/train.py` | Vòng huấn luyện (Algorithm 1), checkpoint và resume |
| `lf/metrics.py`, `lf/eval_table2.py` | Coverage và accuracy theo Table 2 |
| `tests/` | Bộ pytest chạy được trên CPU |

Tài liệu: [`SPEC_TABLE2.md`](implement/looped_model/SPEC_TABLE2.md) (đặc tả) · [`PLAN_TABLE2.md`](implement/looped_model/PLAN_TABLE2.md) (kế hoạch) · [`DEVIATIONS.md`](implement/looped_model/DEVIATIONS.md) (sai khác so với bài báo).

```bash
cd implement/looped_model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # chạy bộ test trên CPU
python -m lf.train --config configs/nq8.yaml --steps 20000 --out runs/seed0 --device cuda --seed 0
```
