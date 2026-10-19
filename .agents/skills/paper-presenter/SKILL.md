---
name: paper-presenter
description: Đọc một paper nghiên cứu từ URL (arXiv, OpenReview, ACL Anthology, DOI, trang web) hoặc file PDF, rồi liệt kê toàn bộ nội dung cần có để trình bày paper đó (problem, key insight, pipeline, từng module, loss, thí nghiệm, ablation, hạn chế, câu hỏi phản biện) kèm outline từng slide và khuyến nghị cách làm slide chuyên nghiệp. Dùng skill này bất cứ khi nào người dùng muốn trình bày/thuyết trình/báo cáo một paper, journal club, paper reading, seminar, bảo vệ đồ án có paper tham khảo, "làm slide cho paper này", "tóm tắt paper để present", "outline trình bày paper", hoặc dán link arxiv/pdf và hỏi nên nói gì, kể cả khi họ không nói rõ chữ "skill" hay "slide".
---

# Paper Presenter

Biến một paper (URL hoặc PDF) thành **kế hoạch trình bày**: người dùng đọc xong biết nên nói gì, theo thứ tự nào, đặt hình/công thức nào lên slide, và làm slide ra sao. Skill này lập kế hoạch; việc dựng file slide chỉ làm khi người dùng yêu cầu (xem bước 6).

Trả lời bằng ngôn ngữ người dùng đang viết (thường là tiếng Việt), giữ nguyên thuật ngữ kỹ thuật tiếng Anh (attention, ablation, baseline...).

## Vì sao cần cách làm này

Người nghe không cần nghe lại cả paper; họ cần hiểu **ý tưởng cốt lõi và vì sao nó hoạt động**. Slide tệ thường do đi theo đúng thứ tự section của paper, dán nguyên hình/đoạn text, và giải thích "cái gì" mà không giải thích "tại sao". Vì vậy việc đọc paper ở đây là đọc *có chủ đích*: tìm đúng những mảnh thông tin để kể một câu chuyện mạch lạc.

## Quy trình

### 1. Lấy nội dung paper

**Người dùng đưa URL**
- arXiv (`arxiv.org/abs/<id>`): thử `WebFetch` bản HTML `arxiv.org/html/<id>` trước (giữ cấu trúc section và caption hình), nếu không được thì `arxiv.org/pdf/<id>`, cuối cùng mới là trang `abs`. Shell trong workspace thường không ra được ngoài allowlist, nên dùng WebFetch thay vì curl/wget.
- OpenReview, ACL Anthology, trang project, DOI: `WebFetch`. Nếu bị chuyển host (redirect) thì gọi lại với URL mới. Nếu trang bị chặn hoặc paywall, nói rõ và nhờ người dùng đính kèm PDF; không tìm đường vòng.
- Nếu chỉ lấy được abstract, nói thẳng giới hạn đó và chỉ lập outline ở mức những gì đã đọc được, không suy đoán chi tiết method.
- Khi hữu ích, `WebSearch` thêm repo code / project page / bản video của paper để gợi ý tài liệu tham khảo cho người trình bày.

**Người dùng đưa PDF** (đính kèm hoặc đường dẫn)
- Chạy script trích xuất cấu trúc:
  ```bash
  python3 -I <skill-dir>/scripts/extract_paper.py <file.pdf> --out <thư-mục-mới> --render figures
  ```
  Script tạo `paper.txt` (text theo trang, có đánh dấu `=== Page N ===`), `structure.md` (heading, caption Figure/Table, số công thức, abstract) và ảnh các trang có hình trong `pages/`. Đọc `structure.md` trước để nắm bố cục, sau đó đọc `paper.txt` ở các phần cần.
- Xem ảnh trang (Read file png) để chọn hình đáng đưa lên slide, nhất là Figure tổng quan kiến trúc và bảng kết quả chính.
- Script báo "ít text" nghĩa là PDF scan: dùng skill `pdf` (OCR) hoặc `pdf-reading` rồi mới tiếp tục.

### 2. Đọc có chủ đích

Đọc theo thứ tự: Abstract → Introduction (đặc biệt danh sách contributions) → Figure tổng quan → Method → Experiments và Ablation → Limitations/Conclusion → Appendix (chỉ để tìm chi tiết cho backup). Trong lúc đọc, thu thập các mục ở checklist dưới đây. Mục nào paper không nói rõ thì ghi "paper không nêu", đừng tự điền.

Checklist thông tin cần trích:
- **Bài toán**: input/output, setting, ký hiệu chính.
- **Động lực & hạn chế của cách làm cũ** (theo lời tác giả), kèm trích dẫn cụ thể paper nào bị chê điểm gì.
- **Key insight**: ý tưởng chính gói trong 1–2 câu.
- **Contributions** theo tác giả.
- **Method**: mỗi module/bước gồm input → xử lý → output, *lý do thiết kế*, công thức cốt lõi, hình tương ứng.
- **Training / inference**: loss, dữ liệu, điểm khác nhau giữa lúc train và lúc chạy, chi tiết triển khai đáng chú ý.
- **Thí nghiệm**: dataset, metric, baseline, kết quả chính (con số + vị trí bảng), ablation (bỏ gì thì giảm bao nhiêu), phân tích định tính, chi phí tính toán.
- **Hạn chế / failure case / giả định** (tác giả nêu và điều bạn quan sát được, ghi rõ cái nào là của ai).
- **Tái lập**: code, dữ liệu, tài nguyên huấn luyện.

Nếu paper thuộc loại đặc thù (survey, theory, benchmark/dataset, system/LLM, empirical study), đọc `references/paper-types.md` để điều chỉnh cấu trúc trình bày cho hợp.

### 3. Thời lượng và đối tượng nghe

Thời lượng và đối tượng nghe (lab nội bộ, hội đồng, cuộc thi, journal club) quyết định độ sâu. Nếu người dùng chưa nói, giả định **20 phút, người nghe cùng lĩnh vực nhưng chưa đọc paper**, ghi rõ giả định ở đầu kết quả và nói họ chỉnh lại nếu khác. Không dừng lại để hỏi trước khi làm.

### 4. Xuất kết quả theo mẫu

Dùng đúng khung này (bỏ mục nào paper không có, ví dụ paper lý thuyết không có training):

```
# Kế hoạch trình bày: <tên paper>
<tác giả, venue/năm, link, code nếu có> · Giả định: <thời lượng>, <đối tượng>

## 1. Tóm tắt 30 giây
3 câu: bài toán, ý tưởng chính, kết quả nổi bật.

## 2. Thông điệp cần người nghe nhớ
1–3 ý. Mọi slide phải phục vụ các ý này.

## 3. Outline slide (bảng)
| # | Tiêu đề slide (câu khẳng định) | Nội dung chính | Hình/công thức dùng | ~Thời gian |
Phân bổ gợi ý: Motivation và related ~15%, Key idea + overview + method ~35–40%,
thí nghiệm ~25%, hạn chế và takeaway ~10%, phần còn lại để đệm. Mỗi slide ~1–2 phút.

## 4. Chi tiết phần Method (từng module)
Với mỗi module: Input → Xử lý → Output; Vì sao cần; Công thức cốt lõi (chú thích từng ký hiệu);
Bằng chứng từ ablation nếu có.

## 5. Hình, bảng, công thức nên dùng
Bảng: Figure/Table/Eq số mấy (trang mấy) → dùng thế nào (vẽ lại gọn / cắt một phần / chỉ lấy số / bỏ vào backup).

## 6. Kết quả thí nghiệm cần nêu
2–3 bảng/hình quan trọng nhất với con số chính (kèm vị trí trong paper) và điều nên rút ra.

## 7. Hạn chế và câu hỏi phản biện dự kiến
5–8 câu hỏi người nghe có thể hỏi (complexity, giả định, so sánh công bằng, khái quát hóa...) kèm gợi ý trả lời.

## 8. Backup slides nên chuẩn bị

## 9. Khuyến nghị làm slide
Tóm tắt 6–10 gạch đầu dòng áp dụng riêng cho paper này (không chép nguyên guideline chung).
```

Phần 9 lấy từ `references/slide-design.md`: đọc file đó, rồi chọn ra những điểm thực sự liên quan (ví dụ paper có pipeline nhiều khối → progressive reveal; paper nặng toán → công thức chọn lọc và ví dụ số; paper nhiều bảng → làm nổi bật con số chính). Kèm gợi ý công cụ vẽ hình và dựng slide phù hợp với loại paper và thói quen người dùng.

### 5. Giữ tính chính xác

- Số liệu, tên module, siêu tham số chỉ lấy từ paper và ghi vị trí (ví dụ `[Table 2, tr. 6]`). Không bịa hoặc làm tròn bừa. Nếu cần tính thêm (ví dụ chênh lệch so với baseline), nói rõ đó là phép tính của bạn.
- Tách rõ **tác giả claim** và **nhận xét của bạn**. Đánh giá phê bình (thiếu baseline, so sánh không cùng điều kiện...) hữu ích cho phần phản biện, nhưng phải có căn cứ trong paper.
- Diễn đạt lại bằng lời mình; không dán nguyên đoạn dài từ paper. Nhắc người dùng ghi nguồn cho mọi hình lấy từ paper và nên vẽ lại hình kiến trúc cho gọn thay vì chụp nguyên.

### 6. Kết thúc

Kết quả là bản kế hoạch (trong câu trả lời; nếu người dùng muốn giữ lại hoặc chia sẻ, đề xuất làm thành tài liệu). Cuối cùng đề xuất một bước tiếp theo cụ thể, ví dụ "dựng bản nháp slide theo outline này" hoặc "viết lời thoại cho từng slide". Chỉ dựng file slide khi người dùng đồng ý hoặc đã yêu cầu; khi đó dùng công cụ/skill tạo slide mà phiên làm việc đang có (Slides artifact hoặc skill pptx), lấy bảng outline ở mục 3 làm nội dung.
