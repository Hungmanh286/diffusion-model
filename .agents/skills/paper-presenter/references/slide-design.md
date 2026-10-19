# Hướng dẫn thiết kế slide trình bày paper

Đọc file này ở bước viết mục "Khuyến nghị làm slide". Chọn những điểm liên quan đến paper đang xử lý, đừng liệt kê hết.

## Mục lục
1. Cấu trúc kể chuyện cho phần Method
2. Nguyên tắc thiết kế slide
3. Hình và công thức
4. Cách nói
5. Công cụ gợi ý
6. Lỗi thường gặp

## 1. Cấu trúc kể chuyện cho phần Method

Mục tiêu là người nghe hiểu ý tưởng và lý do nó hoạt động, không phải tái hiện cả paper. Thứ tự nên kể:

1. **Problem setup**: bài toán, input/output, ký hiệu chính (1 slide).
2. **Hạn chế cách cũ → key insight**: ý tưởng chính trong 1–2 câu. Đây là slide quan trọng nhất của phần Method.
3. **Overview pipeline**: một hình tổng quan duy nhất để người nghe có bức tranh lớn trước khi vào chi tiết.
4. **Từng module**: mỗi module 1–2 slide theo mẫu *input → xử lý → output*, kèm lý do thiết kế ("tại sao cần module này").
5. **Training và inference**: loss, dữ liệu, chỗ khác nhau giữa train và chạy thật.
6. **Design choices**: nối với ablation, ví dụ "bỏ module X thì giảm Y điểm", để thấy mỗi phần có lý do tồn tại.
7. **Tóm tắt** các đóng góp chính (1 slide).

Phần Method thường chiếm 30–40% thời lượng; mỗi slide nói khoảng 1–2 phút.

## 2. Nguyên tắc thiết kế slide

- **Một slide, một ý.** Tiêu đề là câu khẳng định ("Cross-attention hợp nhất đặc trưng ảnh và văn bản") thay vì nhãn chung chung ("Method"). Người chỉ lướt tiêu đề vẫn theo được mạch.
- **Hình nhiều hơn chữ.** Tối đa khoảng 5 dòng ngắn mỗi slide, cỡ chữ từ 20pt, font sans-serif. Chi tiết phụ (hyperparameter, kích thước layer, công thức phụ) để vào Backup/Appendix.
- **Hiện dần (progressive reveal)** cho pipeline phức tạp: làm nổi bật từng khối theo lời nói, các khối chưa nhắc tới thì làm mờ.
- **Nhất quán**: cùng khái niệm dùng cùng ký hiệu và cùng màu trên mọi slide (ví dụ feature ảnh luôn màu xanh).
- **Palette hạn chế**: 2–3 màu chính, nền sáng hoặc tối đơn giản, tránh hiệu ứng chuyển slide màu mè. Dùng màu để mã hóa ý nghĩa, không để trang trí.
- **Số liệu**: với bảng kết quả, chỉ giữ cột/hàng cần so sánh, in đậm hoặc tô màu con số cần chú ý, ghi rõ đơn vị và chiều tốt hơn (↑/↓). Một hình biểu đồ thường đọc nhanh hơn một bảng lớn.
- **Số slide**: ước lượng = số phút / 1,5. Slide nhiều hơn mức đó thường nghĩa là quá tải.
- **Có slide "roadmap" thu nhỏ** ở các lần chuyển phần để người nghe không lạc.

## 3. Hình và công thức

- **Tự vẽ lại hình kiến trúc** cho gọn: bỏ chi tiết thừa, chữ đủ to, màu theo palette của bài. Đừng chụp nguyên Figure của paper rồi để người nghe tự đọc. Nếu dựa vào hình gốc, ghi nguồn (tên paper, số Figure).
- **Công thức chọn lọc**: chỉ đưa công thức cốt lõi (loss chính hoặc phép biến đổi chính). Tô màu từng thành phần và ghi chú ý nghĩa ngay cạnh. Nếu được, thêm một ví dụ số nhỏ để người nghe "thấy" công thức làm gì.
- **Ví dụ minh họa**: với module trừu tượng, một ví dụ cụ thể (câu, ảnh, trạng thái, đồ thị nhỏ) đi qua cả pipeline có giá trị hơn một đoạn mô tả.
- **Hình định tính (qualitative)** của paper thường rất thuyết phục: chọn 1–2 ví dụ tốt và 1 failure case để vừa chứng minh vừa thể hiện sự trung thực.

## 4. Cách nói

- Mở mỗi module bằng một câu hỏi hoặc vấn đề ("Làm sao để mô hình biết vùng nào quan trọng?") rồi mới đưa giải pháp.
- Dùng ví dụ cụ thể hoặc analogy cho khái niệm trừu tượng.
- Chuyển slide bằng câu nối ("Có đặc trưng rồi, bước tiếp theo là…").
- Chuẩn bị sẵn slide dự phòng cho câu hỏi về complexity, giả định, hạn chế, so sánh baseline.
- Luyện thử với người chưa đọc paper và bấm giờ; cắt slide khi vượt thời lượng, không nói nhanh hơn.

## 5. Công cụ gợi ý

| Nhu cầu | Gợi ý |
|---|---|
| Slide thông thường, làm nhanh, cộng tác | PowerPoint, Google Slides, Keynote |
| Nhiều công thức, thuần học thuật, muốn kiểm soát chính xác | LaTeX Beamer hoặc Typst |
| Sơ đồ pipeline/kiến trúc | Vẽ trực tiếp bằng shape trong PowerPoint/Keynote, hoặc draw.io, Figma, Excalidraw; TikZ nếu dùng Beamer |
| Công thức trong PowerPoint/Slides | Equation editor có sẵn, hoặc xuất LaTeX ra SVG/PNG nền trong suốt |
| Biểu đồ kết quả | Vẽ lại từ số liệu trong paper (matplotlib, Excel) với palette thống nhất, thay vì chụp hình gốc |
| Cần bản nháp nhanh từ outline | Nhờ Claude dựng bản nháp slide từ bảng outline, sau đó chỉnh tay phần hình |

## 6. Lỗi thường gặp

- Đi đúng thứ tự section của paper thay vì kể theo logic dễ hiểu.
- Dán nguyên đoạn text hoặc công thức dài từ paper.
- Nhồi cả pipeline vào một slide với chữ nhỏ li ti.
- Giải thích *cái gì* mà không nói *tại sao*.
- Quên nói rõ điểm mới so với các công trình trước.
- Chỉ nêu kết quả tốt, bỏ qua ablation và hạn chế, khiến người nghe nghi ngờ hoặc không có gì để thảo luận.
- Ký hiệu và màu thay đổi giữa các slide.
