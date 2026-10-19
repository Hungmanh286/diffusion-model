# Điều chỉnh cấu trúc trình bày theo loại paper

Cấu trúc chuẩn trong SKILL.md phù hợp nhất với paper đề xuất phương pháp mới. Với các loại khác, giữ tinh thần "kể theo logic người nghe, không theo section" nhưng đổi trọng tâm như sau.

## Đề xuất phương pháp (model / thuật toán)
Cấu trúc chuẩn. Nhấn mạnh key insight, overview pipeline, từng module và ablation. Thí nghiệm: kết quả chính, ablation, phân tích định tính.

## Paper nặng lý thuyết
- Dành nhiều thời gian hơn cho **phát biểu bài toán và trực giác** của định lý, ít cho chứng minh.
- Mỗi định lý chính: phát biểu → ý nghĩa bằng lời → ví dụ nhỏ hoặc trường hợp đặc biệt → giả định nào là quan trọng. Đưa chứng minh vào backup, trừ khi chứng minh chứa ý tưởng mới.
- Không có phần training; thay bằng "điều kiện áp dụng" và "độ chặt của kết quả".

## Benchmark / dataset
- Trọng tâm là **thiết kế dữ liệu**: nguồn, cách thu thập, gán nhãn, kiểm soát chất lượng, thống kê.
- Nêu rõ benchmark đo năng lực gì mà benchmark cũ chưa đo, và bằng chứng (các mô hình xếp hạng thế nào, lỗi điển hình).
- Thí nghiệm: bảng so sánh các mô hình, phân tích lỗi, giới hạn của dataset (thiên lệch, rò rỉ dữ liệu).

## Survey / tổng quan
- Đừng đi qua từng công trình. Dùng **taxonomy** làm xương sống: 1 slide hình taxonomy, mỗi nhánh chính 1–2 slide với đại diện tiêu biểu.
- Kết bằng open problems và xu hướng; đây thường là phần người nghe quan tâm nhất.

## Nghiên cứu thực nghiệm / phân tích (empirical study)
- Câu hỏi nghiên cứu (RQ) làm trục: mỗi RQ một khối gồm thiết lập thí nghiệm, kết quả, kết luận.
- Chú ý nêu rõ thiết lập có công bằng không (cùng ngân sách, cùng dữ liệu) và mức độ chắc chắn của kết luận.

## System / LLM / kỹ thuật huấn luyện quy mô lớn
- Kiến trúc hệ thống và quyết định thiết kế quan trọng hơn công thức. Có hình luồng dữ liệu.
- Nêu rõ **dữ liệu, quy mô tính toán, chi phí** và những điều cần thiết để tái lập.
- Đánh giá: benchmark nào, so với mô hình nào, có rủi ro contamination hay không; phần an toàn/hạn chế nếu paper có.

## Paper ứng dụng (applied / domain-specific)
- Bắt đầu bằng **bối cảnh thực tế và ràng buộc** (dữ liệu hiếm, độ trễ, chi phí, quy định).
- Nêu những chỗ phương pháp được điều chỉnh cho miền ứng dụng và kết quả theo metric mà người dùng thật quan tâm.
- Hạn chế khi triển khai và bài học thực tế thường là phần có giá trị thảo luận cao.
