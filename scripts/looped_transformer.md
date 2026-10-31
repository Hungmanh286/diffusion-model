# Kịch Bản Thuyết Trình: The Surprising Effectiveness of Shared Memory in Looped Transformers

- **Bài báo:** *The Surprising Effectiveness of Shared Memory in Looped Transformers* (arXiv:2610.02383, 2026)
- **Tác giả:** Giovanni Monea, Keshav Ramji, Yousef El-Kurdi, Luis A. Lastras, Yoav Artzi, Nathan Godey, Ramón Fernandez Astudillo (Cornell University & IBM Research)
- **File slide LaTeX tương ứng:** `overleaf/looped transformer/lpt_paper.tex` (hoặc `overleaf/looped transformer/main.tex`)
- **Người trình bày:** Hung Manh
- **Thời lượng dự kiến:** 20 – 25 phút (bao gồm Q&A)

---

## 📌 Hướng Dẫn & Phân Bổ Thời Gian
- **Mở đầu & Bối cảnh (Slide Bìa $\to$ Slide 1):** ~3.5 phút.
- **Kiến trúc & Thiết lập thực nghiệm (Slide 2 $\to$ Slide 3):** ~4.5 phút.
- **Kết quả & Độ ổn định (Slide 4 $\to$ Slide 5):** ~5 phút.
- **Giải mã hai cơ chế cốt lõi (Slide 6 $\to$ Slide 7):** ~5 phút.
- **Đo thực tế GPU & Độ tin cậy (Slide 8 $\to$ Slide 9):** ~4 phút.
- **Hạn chế & Kết luận (Slide 10 $\to$ Q&A):** ~3 – 5 phút.
- **Yêu cầu văn phong:** Khoa học, chính xác, trung tính, trực diện. Tránh dùng từ ngữ cảm thán hoặc phóng đại. Giữ nguyên các thuật ngữ kỹ thuật chuyên môn (*KV cache, recursion, effective depth, sliding window, perplexity, Gated DeltaNet, gradient highway, unigram perplexity, FLOPs, FlexAttention*).

---

## 📑 Chi Tiết Kịch Bản Từng Slide

### Trang Bìa & Mục Lục
- **Slide trong TeX:** `main.tex` (Trang bìa & Mục lục)
- **Thời lượng:** ~1 phút
- **Lời thoại:**
  > "Kính thưa quý thầy cô và các bạn, hôm nay em xin trình bày bài báo nghiên cứu: **'The Surprising Effectiveness of Shared Memory in Looped Transformers'**, được công bố vào tháng 10 năm 2026 bởi nhóm nghiên cứu từ Đại học Cornell và IBM Research.
  >
  > Bài thuyết trình bao gồm 4 phần:
  > 1. Bối cảnh của mô hình Transformer vòng lặp và vấn đề mở rộng bộ nhớ KV cache.
  > 2. Kiến trúc LPT và biến thể mô hình lai H-LPT với cơ chế bộ nhớ dùng chung.
  > 3. Kết quả thực nghiệm, tính ổn định và hai cơ chế giải thích: đại lộ gradient cùng sự phân hóa vai trò giữa các vòng lặp.
  > 4. Đo đạc thực tế trên phần cứng GPU H100, các giới hạn kỹ thuật và kết luận."

---

### Slide 0: LPT: Tổng quan bài báo
- **Slide trong TeX:** `lpt_paper.tex` (Slide 0)
- **Thời lượng:** ~1.5 phút
- **Mục tiêu:** Tóm tắt tác giả, ý tưởng kỹ thuật then chốt và các kết quả định lượng chính của công trình.
- **Lời thoại:**
  > "Về tổng quan, bài báo đặt ra câu hỏi về cách quản lý bộ nhớ ngữ cảnh trong mô hình Transformer vòng lặp (Looped Transformers).
  >
  > Ý tưởng cốt lõi của bài báo là: **huấn luyện Transformer vòng lặp từ đầu với cơ chế bộ nhớ dùng chung (shared memory)**. Cụ thể, chỉ vòng lặp đầu tiên ghi toàn bộ KV cache của ngữ cảnh, trong khi các vòng lặp sau tái sử dụng lại KV cache này kết hợp với một cửa sổ trượt cục bộ ngắn.
  >
  > Các đóng góp chính của công trình gồm:
  > 1. **Chia sẻ bộ nhớ giúp cải thiện chất lượng mô hình:** Khác với quan niệm trước đây cho rằng chia sẻ bộ nhớ làm giảm chất lượng, mô hình LPT giảm perplexity từ 0.38 đến 1.09 so với Transformer vòng lặp thông thường, đồng thời giảm dung lượng KV cache từ 2.0 đến 4.7 lần.
  > 2. **Biến thể mô hình lai H-LPT:** Khi chạy ở cấu hình 5 vòng lặp, H-LPT đạt perplexity thấp hơn Transformer tiêu chuẩn từ 1.12 đến 1.82, trong khi sử dụng ít hơn 76% đến 79% bộ nhớ ngữ cảnh.
  > 3. **Giải thích cơ chế:** Bài báo chứng minh hai cơ chế giải thích cho hiện tượng này: sự hình thành của đại lộ gradient (gradient highway) và sự phân hóa chức năng rõ rệt giữa vòng lặp xây bộ nhớ và các vòng lặp dự đoán."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào tên tác giả và tổ chức]`: Nêu rõ công trình từ Cornell và IBM Research (arXiv:2610.02383).
  - `[Chỉ vào số liệu PPL và KV cache]`: Nhấn mạnh việc giảm đồng thời cả Perplexity lẫn dung lượng bộ nhớ.
- **Câu hỏi phản biện dự kiến (Q&A):**
  - *Hỏi:* Điểm khác biệt cơ bản giữa cách chia sẻ bộ nhớ của LPT và các nghiên cứu trước đây là gì?
  - *Đáp:* Các nghiên cứu trước (như Ouro) chỉ tái sử dụng hoặc cắt giảm KV cache tại thời điểm suy luận sau khi đã huấn luyện mô hình thông thường, dẫn đến suy giảm chất lượng nghiêm trọng. LPT huấn luyện mô hình từ đầu với cấu trúc chia sẻ bộ nhớ, biến nó thành một thiên kiến quy nạp (inductive bias) của kiến trúc.

---

### Slide 1: 1. LPT: Bối cảnh và cái bẫy bộ nhớ KV cache
- **Slide trong TeX:** `lpt_paper.tex` (Slide 1)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Phân tích nguyên lý hoạt động của Looped Transformer, công thức tính dung lượng KV cache và mâu thuẫn về bộ nhớ ở ngữ cảnh dài.
- **Lời thoại:**
  > "Về mặt nguyên lý, Transformer vòng lặp (Looped Transformer) áp dụng cùng một khối gồm $L$ tầng Transformer lặp lại $R$ lần cho mỗi token:
  > $$h^{(t)} = M_L\big(h_{\text{in}}^{(t)}\big), \qquad h_{\text{in}}^{(t+1)} = N_{\text{rec}}\big(h^{(t)}\big), \qquad t = 1, \dots, R$$
  > Trong đó $N_{\text{rec}}$ là tầng chuẩn hóa chuyển tiếp trạng thái giữa các vòng lặp. Mô hình có độ sâu hiệu dụng tương đương $L \cdot R$ tầng, nhưng số lượng tham số lưu trên bộ nhớ chỉ bằng một khối $L$ tầng.
  >
  > **Vấn đề bộ nhớ KV cache:**
  > Trong kiến trúc vòng lặp thông thường, mỗi vòng lặp $t$ lại tính toán và lưu trữ một bảng KV cache độc lập cho toàn bộ chuỗi độ dài $T$. Do đó, tổng dung lượng bộ nhớ KV cache bằng:
  > $$M_{\text{KV}}^{\text{LT}} = L \cdot R \cdot T$$
  > Dung lượng này tăng tuyến tính theo số vòng lặp $R$. Khi làm việc với ngữ cảnh dài (ví dụ $T = 4096$ hoặc dài hơn), dung lượng bộ nhớ dành cho KV cache nhanh chóng vượt qua dung lượng bộ nhớ lưu trọng số mô hình. Điều này làm mất đi ưu thế tiết kiệm bộ nhớ ban đầu của kiến trúc vòng lặp.
  >
  > **Hạn chế của các giải pháp trước đây:**
  > Các phương pháp trước đây thường tìm cách bỏ bớt hoặc dùng lại KV cache chỉ tại lúc suy luận. Ví dụ, trong mô hình Ouro, nếu chỉ giữ lại cache của vòng lặp đầu tiên lúc suy luận thì chất lượng mô hình sụt giảm nghiêm trọng. Chia sẻ bộ nhớ trước đây chỉ được xem là một biện pháp kỹ thuật nhằm tiết kiệm dung lượng và chấp nhận giảm hiệu năng."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào công thức M_{KV}^{LT} = L * R * T]`: Nhấn mạnh sự tăng tuyến tính theo số vòng lặp $R$.
  - `[Nêu bật mâu thuẫn]`: Mục đích của Looped Transformer là tiết kiệm bộ nhớ tham số, nhưng lại bị nghẽn ở bộ nhớ ngữ cảnh KV cache.

---

### Slide 2: 2. LPT: Kiến trúc Transformer vòng lặp dự đoán
- **Slide trong TeX:** `lpt_paper.tex` (Slide 2)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Trình bày cấu trúc LPT, cơ chế phân tách vòng trạng thái ($t=1$) và vòng dự đoán ($t \ge 2$), công thức bộ nhớ KV và biến thể lai H-LPT.
- **Lời thoại:**
  > "Để giải quyết vấn đề trên, kiến trúc **LPT (Looped Prediction Transformer)** thực hiện phân tách rõ ràng nhiệm vụ xây dựng bộ nhớ và nhiệm vụ dự đoán theo trục vòng lặp:
  >
  > 1. **Vòng lặp trạng thái ($t = 1$):** Chạy qua chuỗi đầu vào và ghi lại bảng KV cache đầy đủ trên toàn bộ ngữ cảnh, ký hiệu là $C^{(1)}$.
  > 2. **Các vòng lặp dự đoán ($t \ge 2$):** Không ghi thêm KV cache đầy đủ nữa, mà đọc lại bảng cache dùng chung $C^{(1)}$, kết hợp với một cửa sổ trượt cục bộ ngắn gồm $\omega = 64$ token gần nhất do chính vòng lặp đó tạo ra:
  > $$\mathcal{A}_{\text{LPT}}^{(t)}(i) = C^{(1)}_{\le i} \;\cup\; C^{(t)}_{[i-\omega,\, i]}, \qquad t = 2, \dots, R$$
  > Cả phần bộ nhớ dùng chung và cửa sổ cục bộ này được đưa vào cùng một phép tính Softmax duy nhất trong cơ chế Attention.
  >
  > **Dung lượng bộ nhớ KV cache:**
  > $$M_{\text{KV}}^{\text{LPT}} = L\,\big[\,T + (R-1)\omega\,\big]$$
  > Với độ dài ngữ cảnh $T = 4096$, kích thước cửa sổ $\omega = 64$ và số vòng lặp $R = 5$, dung lượng KV cache của LPT nhỏ hơn **4.7 lần** so với mô hình vòng lặp thông thường.
  >
  > **Biến thể mô hình lai H-LPT:**
  > Tác giả kết hợp cơ chế này vào kiến trúc lai Gated DeltaNet (trong đó cứ 4 tầng chú ý tuyến tính thì có 1 tầng chú ý đầy đủ). Các tầng chú ý tuyến tính duy trì một trạng thái ẩn kích thước cố định qua các vòng lặp, trong khi các tầng chú ý đầy đủ áp dụng cơ chế chia sẻ bộ nhớ của LPT.
  >
  > **Triển khai kỹ thuật:** Quá trình huấn luyện sử dụng mặt nạ khối FlexAttention của PyTorch để ghép nối các vector khóa-giá trị dùng chung và cục bộ, cho phép tính toán song song hiệu quả."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào công thức tập truy cập \mathcal{A}_{LPT}^{(t)}]`: Giải thích rõ hai thành phần: $C^{(1)}_{\le i}$ (toàn bộ quá khứ từ vòng 1) và $C^{(t)}_{[i-\omega, i]}$ (64 vị trí gần nhất của vòng hiện tại).
  - `[Chỉ vào hệ số giảm 4.7 lần]`: Nêu rõ con số định lượng khi $T=4096, \omega=64, R=5$.

---

### Slide 3: 3. LPT: Thiết lập thực nghiệm
- **Slide trong TeX:** `lpt_paper.tex` (Slide 3)
- **Thời lượng:** ~2 phút
- **Mục tiêu:** Trình bày cấu hình dữ liệu, 3 quy mô mô hình (S, M, L), các baseline đối chứng và các tiêu chí đánh giá.
- **Lời thoại:**
  > "Về thiết lập thực nghiệm, bài báo đánh giá mô hình trên các điều kiện tiêu chuẩn:
  >
  > - **Dữ liệu huấn luyện:** Tập dữ liệu FineWeb-Edu với độ dài ngữ cảnh 4096 token.
  > - **Quy mô mô hình:** Ba kích thước:
  >   - Cỡ **S** (152 triệu tham số, huấn luyện trên 20 tỷ token).
  >   - Cỡ **M** (454 triệu tham số, huấn luyện trên 20 tỷ token).
  >   - Cỡ **L** (1.01 tỷ tham số, huấn luyện trên 50 tỷ token).
  > - **Số vòng lặp khảo sát:** $R \in \{2, 3, 5\}$, và mở rộng lên 9 vòng lặp trong phần phụ lục.
  > - **Các mô hình đối chứng:**
  >   - Transformer chuẩn (Standard Transformer).
  >   - Transformer sâu-hẹp (khớp cả số tham số và độ sâu hiệu dụng).
  >   - Transformer vòng lặp thông thường (Regular Looped Transformer).
  >   - Các kiến trúc vòng lặp khác: PLT (vòng lặp song song), LOOPIE, và mô hình lai LT2.
  > - **Hàm mục tiêu:** So sánh giữa mất mát ở vòng lặp cuối cùng (final loop loss) và mất mát thích ứng có trọng số của Ouro.
  > - **Tiêu chí đánh giá:** Perplexity (PPL) trên 10 triệu token kiểm tra, độ chính xác zero-shot trung bình trên 5 tác vụ chuẩn (ARC-Easy, HellaSwag, PIQA, SciQ, LAMBADA), cùng dung lượng bộ nhớ và số phép tính FLOPs tính theo giải tích.
  > - Đặc biệt, tác giả giữ nguyên cấu hình siêu tham số mặc định của nanoGPT cho tất cả các mô hình để đảm bảo tính công bằng."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào 3 cỡ mô hình S, M, L]`: Nêu rõ mô hình lớn nhất có quy mô 1 tỷ tham số.
  - `[Chỉ vào danh sách 5 tác vụ benchmark]`: Khẳng định đánh giá cả năng lực mô hình ngôn ngữ (PPL) và suy luận zero-shot.

---

### Slide 4: 4. LPT: Kết quả chính ở cỡ 1B, 5 vòng lặp
- **Slide trong TeX:** `lpt_paper.tex` (Slide 4)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Phân tích bảng kết quả so sánh ở quy mô 1 tỷ tham số và 5 vòng lặp, làm rõ mức giảm PPL và dung lượng KV cache.
- **Lời thoại:**
  > "Bảng số liệu trên slide thể hiện kết quả so sánh chính ở quy mô **1 tỷ tham số với 5 vòng lặp ($R = 5$)**:
  >
  > - **Transformer chuẩn:** Đạt PPL 11.35, độ chính xác zero-shot 62.5%, dung lượng KV cache quy ước là $1.00\times$.
  > - **Transformer vòng lặp thông thường:** PPL là 11.39, độ chính xác 63.4%, nhưng dung lượng KV cache tăng gấp **5 lần** ($5.00\times$).
  > - **Mô hình LPT:** Đạt PPL **10.32**, độ chính xác **68.2%**, trong khi dung lượng KV cache chỉ tăng nhẹ lên $1.06\times$.
  > - **Mô hình lai H-LPT:** Đạt PPL tốt nhất là **10.23**, độ chính xác 68.1%, và dung lượng bộ nhớ ngữ cảnh chỉ bằng **$0.24\times$** (tức là giảm tới 76% so với Transformer chuẩn).
  >
  > **Các kết luận rút ra trên toàn bộ các quy mô (S, M, L):**
  > 1. So với Transformer vòng lặp thông thường, LPT đạt PPL thấp hơn từ 0.38 đến 1.09, độ chính xác cao hơn từ 0.4 đến 4.8 điểm, dung lượng KV cache nhỏ hơn từ 2.0 đến 4.7 lần, trong khi số lượng FLOPs chỉ tăng dưới 1%.
  > 2. So với Transformer tiêu chuẩn, mô hình lai H-LPT ở 5 vòng lặp có PPL thấp hơn từ 1.12 đến 1.82, độ chính xác cao hơn từ 3.1 đến 5.6 điểm, đồng thời giảm được từ 76% đến 79% bộ nhớ ngữ cảnh."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào dòng LPT và H-LPT]`: Nêu rõ mức cải thiện PPL từ 11.35 xuống 10.32 và 10.23.
  - `[Chỉ vào cột KV cache của H-LPT 0.24x]`: Nhấn mạnh đây là mức giảm bộ nhớ ngữ cảnh đáng kể so với Transformer chuẩn.

---

### Slide 5: 5. LPT: Mở rộng theo độ sâu và độ ổn định huấn luyện
- **Slide trong TeX:** `lpt_paper.tex` (Slide 5)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Phân tích khả năng mở rộng theo số vòng lặp và hiện tượng sụp đổ của mô hình vòng lặp thường khi huấn luyện bằng mất mát vòng cuối.
- **Lời thoại:**
  > "Slide số 5 phân tích khả năng mở rộng khi tăng số vòng lặp $R$ và tính ổn định trong quá trình huấn luyện:
  >
  > **Khả năng mở rộng theo độ sâu:**
  > Đối với Transformer vòng lặp thông thường, hiệu năng mô hình ngừng cải thiện khi tăng từ 3 vòng lặp lên 5 vòng lặp trên tất cả các kích thước mô hình. Ngược lại, LPT tiếp tục cải thiện độ chính xác và giảm perplexity khi tăng số vòng lặp lên $R = 5$.
  >
  > **Hiện tượng mất ổn định với hàm mất mát ở vòng cuối (Final Loop Loss):**
  > - Khi huấn luyện Transformer vòng lặp thông thường bằng hàm mất mát chỉ tính ở vòng lặp cuối cùng, quá trình huấn luyện trở nên kém ổn định khi tăng số vòng lặp. Ở cấu hình 5 vòng lặp với chuẩn hóa của Ouro, mô hình cỡ S bị sụp đổ hoàn toàn về mức phân phối unigram (PPL tăng vọt lên mức 2040). Do đó, các mô hình vòng lặp trước đây buộc phải dùng hàm mất mát trung bình có trọng số qua các vòng lặp.
  > - Ngược lại, LPT duy trì tính ổn định khi huấn luyện trực tiếp bằng mất mát vòng cuối. Thực nghiệm cho thấy mất mát vòng cuối giúp LPT đạt PPL tốt hơn từ 0.16 đến 0.50 so với mất mát thích ứng. Lý do là các vòng lặp trung gian không bị ép phải thực hiện dự đoán sớm, mà có thể tập trung hoàn toàn vào việc tinh chỉnh trạng thái biểu diễn.
  >
  > **Khi mở rộng lên 9 vòng lặp ($R = 9$):**
  > LPT cải thiện nhẹ ở cỡ S nhưng giảm 1.7 điểm độ chính xác ở cỡ L. Tuy nhiên, biến thể mô hình lai H-LPT tiếp tục cải thiện hiệu năng ở cả ba quy mô S, M, L.
  >
  > Kết luận: Cơ chế chia sẻ bộ nhớ giúp ổn định quá trình huấn luyện vòng lặp sâu, nhưng việc tăng số vòng lặp lên quá cao vẫn cần các kiến trúc lai để duy trì hiệu quả."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào hiện tượng sụp đổ PPL 2040]`: Giải thích việc huấn luyện chuỗi lặp sâu bằng final loss rất dễ gây phân kỳ nếu không có cơ chế chia sẻ bộ nhớ.
  - `[Chỉ vào kết quả R=9]`: Nêu rõ giới hạn thực nghiệm khi tăng số vòng lặp lên mức cao.

---

### Slide 6: 6. LPT: Giải mã cơ chế 1 -- Đại lộ gradient
- **Slide trong TeX:** `lpt_paper.tex` (Slide 6)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Trình bày cơ chế Đại lộ gradient (Gradient Highway) truyền tín hiệu trực tiếp về vòng lặp đầu tiên và thí nghiệm kiểm định nhân quả bằng cách cắt gradient.
- **Lời thoại:**
  > "Để trả lời câu hỏi vì sao việc chia sẻ bộ nhớ lại cải thiện chất lượng mô hình, tác giả phân tích cơ chế thứ nhất: **Đại lộ gradient (Gradient Highway)**.
  >
  > **Đường truyền gradient trực tiếp về vòng lặp đầu:**
  > Trong mô hình vòng lặp thông thường, gradient từ hàm mất mát ở vòng cuối muốn truyền về vòng đầu tiên phải đi ngược tuần tự qua toàn bộ các trạng thái ẩn trung gian. Khi tăng số vòng lặp, độ lớn gradient suy giảm mạnh qua từng bước lặp.
  > Trong LPT, do tất cả các vòng lặp dự đoán ($t \ge 2$) đều trực tiếp chú ý vào KV cache của vòng đầu tiên $C^{(1)}$, gradient từ hàm mất mát cuối có một đường tắt truyền thẳng về các tham số của vòng đầu tiên. Thực nghiệm đo đạc cho thấy ở cấu hình 5 vòng lặp, phần lớn gradient truyền về vòng lặp đầu đi qua đường kết nối bộ nhớ dùng chung này.
  >
  > **Thí nghiệm kiểm định quan hệ nhân quả (Cắt gradient):**
  > Để kiểm chứng xem đường truyền gradient này có thực sự tạo ra sự cải thiện hay không, tác giả thực hiện thí nghiệm: giữ nguyên toàn bộ tính toán ở chiều thuận (forward pass), nhưng áp dụng toán tử chặn gradient (`stop-gradient`) trên KV cache dùng chung.
  > Kết quả:
  > - Perplexity tăng mạnh trên tất cả các cấu hình: ở cỡ S với 5 vòng lặp, PPL tăng từ 14.91 lên 33.21.
  > - Ở cỡ M với 3 và 5 vòng lặp, mô hình không thể hội tụ và rơi về mức unigram.
  >
  > **Điểm cần lưu ý về mặt phân tích:**
  > Tác giả cũng thận trọng chỉ ra rằng: tỷ lệ gradient đi qua KV dùng chung vốn đã cao hơn ngay từ thời điểm khởi tạo ngẫu nhiên trước khi huấn luyện. Do đó, thí nghiệm này xác nhận việc duy trì đường gradient là cần thiết, nhưng chưa phân biệt được hoàn toàn giữa ảnh hưởng của chiều dài đường truyền gradient và sự thay đổi về độ lớn hay hướng của vector gradient."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào sơ đồ gradient highway]`: Minh họa đường truyền gradient trực tiếp từ output về vòng 1 qua KV cache.
  - `[Chỉ vào số liệu cắt gradient PPL 14.91 -> 33.21]`: Nhấn mạnh kết quả kiểm định nhân quả thực nghiệm.

---

### Slide 7: 7. LPT: Giải mã cơ chế 2 -- Phân hóa vai trò và cửa sổ cục bộ
- **Slide trong TeX:** `lpt_paper.tex` (Slide 7)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Trình bày cơ chế thứ hai: sự phân hóa vai trò giữa vòng trạng thái và vòng dự đoán qua trọng số attention và phân tích PCA, cùng khảo sát kích thước cửa sổ $\omega$.
- **Lời thoại:**
  > "Cơ chế thứ hai được tác giả chỉ ra là: **Sự phân hóa chức năng giữa các vòng lặp**.
  >
  > **Tỷ lệ chú ý vào bộ nhớ dùng chung:**
  > Khi đo phân phối trọng số Attention của các vòng lặp dự đoán:
  > - Các vòng lặp dự đoán đầu dành từ 57% đến 71% trọng số chú ý vào bộ nhớ dùng chung $C^{(1)}$.
  > - Ở vòng lặp cuối cùng, tỷ lệ này tăng lên từ 73% đến 80%.
  > Điều này diễn ra ngay cả khi cửa sổ cục bộ được mở rộng ra toàn bộ chiều dài ngữ cảnh, cho thấy mô hình chủ động lựa chọn đọc thông tin từ bộ nhớ dùng chung thay vì bộ nhớ cục bộ.
  >
  > **Phân tích không gian biểu diễn (PCA):**
  > Phân tích thành phần chính (PCA) trên các vector khóa (keys) cho thấy: các vector khóa dùng chung từ vòng 1 và các vector khóa cục bộ từ các vòng sau phân tách thành các cụm riêng biệt trong không gian biểu diễn.
  > Trong khi đó, ở mô hình vòng lặp thông thường, các vector khóa giữa các vòng lặp có độ chồng lấn cao hơn.
  > Điều này cho thấy: vòng lặp đầu tiên chuyên biệt hóa vào nhiệm vụ xây dựng bộ nhớ biểu diễn ngữ cảnh, trong khi các vòng lặp sau chuyên biệt hóa vào việc khai thác bộ nhớ đó để suy luận và đưa ra dự đoán.
  >
  > **Khảo sát kích thước cửa sổ cục bộ $\omega$ (từ 0 đến 4096):**
  > Tác giả quét kích thước cửa sổ $\omega$ từ 0 đến toàn bộ ngữ cảnh 4096:
  > - LPT cho PPL tốt hơn mô hình vòng lặp thông thường ở mọi giá trị $\omega$.
  > - Việc thay đổi độ rộng cửa sổ có tác động tương đối nhỏ đến PPL; yếu tố quyết định chất lượng là việc được truy cập vào bộ nhớ dùng chung.
  > - Tác giả lựa chọn $\omega = 64$ để cân bằng tối ưu giữa việc cung cấp ngữ cảnh lân cận và giảm thiểu dung lượng bộ nhớ."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào tỷ lệ attention 73-80%]`: Nhấn mạnh việc các vòng sau ưu tiên đọc bộ nhớ từ vòng 1.
  - `[Chỉ vào biểu đồ phân cụm PCA]`: Nêu rõ sự phân hóa biểu diễn giữa state keys và prediction keys.

---

### Slide 8: 8. LPT: Đo thực tế trên GPU H100
- **Slide trong TeX:** `lpt_paper.tex` (Slide 8)
- **Thời lượng:** ~2 phút
- **Mục tiêu:** Trình bày kết quả đo đạc thực tế trên phần cứng GPU H100: kích thước batch tối đa, thông lượng và sự đánh đổi về thời gian tính toán.
- **Lời thoại:**
  > "Slide số 8 trình bày kết quả đo đạc thực tế trên phần cứng **GPU NVIDIA H100** với độ dài ngữ cảnh 4096 token, định dạng dữ liệu bfloat16:
  >
  > **Sức chứa kích thước batch (Batch capacity):**
  > Nhờ tiết kiệm bộ nhớ KV cache, LPT cho phép tăng kích thước batch trên một GPU:
  > - LPT chứa được kích thước batch tương đương 93% đến 98% so với Transformer tiêu chuẩn, và gấp 2.0 lần (ở 2 vòng), 2.9 lần (ở 3 vòng) và **4.7 đến 4.8 lần** so với mô hình vòng lặp thông thường ở 5 vòng lặp.
  > - Mô hình lai H-LPT cho phép tăng kích thước batch lên gấp **3.6 đến 5.0 lần** so với Transformer tiêu chuẩn.
  >
  > **Thông lượng xử lý (Throughput tại batch tối đa):**
  > So với Transformer tiêu chuẩn, H-LPT đạt thông lượng cao hơn 2.1 đến 2.2 lần ở 2 vòng; 1.4 đến 1.5 lần ở 3 vòng; và đạt 0.8 đến 0.9 lần ở 5 vòng lặp.
  >
  > **Đánh đổi về chi phí tính toán (FLOPs và độ trễ):**
  > Tác giả chỉ rõ các chi phí không thể triệt tiêu:
  > - Khi chạy cùng một kích thước batch cố định, LPT có thời gian xử lý chậm hơn mô hình vòng lặp thường từ 3% đến 8%, do cơ chế Attention phải đọc đồng thời cả hai vùng nhớ. Bộ nhớ dùng chung giúp giảm dung lượng lưu trữ, nhưng không làm giảm thời gian truy xuất dữ liệu.
  > - Ở chế độ suy luận giải mã từng token với batch bằng 1, ở 5 vòng lặp, LPT tiêu tốn lượng FLOPs gấp **4.3 đến 4.8 lần** so với Transformer tiêu chuẩn."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào chỉ số sức chứa batch gấp 4.7-4.8 lần]`: Nêu rõ lợi ích trực tiếp của việc giảm KV cache đối với việc phục vụ suy luận (serving).
  - `[Chỉ vào phần FLOPs tăng 4.3-4.8x]`: Thẳng thắn trình bày sự đánh đổi về chi phí tính toán khi chạy tuần hoàn.

---

### Slide 9: 9. LPT: Độ tin cậy thống kê và tính công bằng
- **Slide trong TeX:** `lpt_paper.tex` (Slide 9)
- **Thời lượng:** ~2 phút
- **Mục tiêu:** Trình bày kiểm định thống kê qua 4 hạt giống ngẫu nhiên, phân tích ảnh hưởng của kỹ thuật chuẩn hóa đến khoảng cách hiệu năng giữa các mô hình.
- **Lời thoại:**
  > "Để đảm bảo tính nghiêm ngặt khoa học, tác giả tiến hành kiểm định thống kê qua **4 hạt giống ngẫu nhiên** ở quy mô S, sử dụng kiểm định t ghép cặp một phía (paired t-test):
  >
  > - Độ lệch chuẩn về PPL giữa các lần chạy với các hạt giống khác nhau tối đa chỉ là 0.08 PPL, khẳng định kết quả đo đạc có tính ổn định thống kê cao.
  > - Mức chênh lệch PPL giữa LPT và mô hình vòng lặp thông thường tăng rõ rệt khi tăng số vòng lặp: $-0.40$ ở 2 vòng, $-0.71$ ở 3 vòng và $-1.05$ ở 5 vòng lặp. Ở cấu hình 2 vòng lặp, mức chênh lệch giữa LPT và PLT chưa thực sự lớn ($-0.07$ PPL).
  >
  > **Phân tích về tính công bằng trong điều kiện so sánh:**
  > Tác giả thẳng thắn chỉ ra một yếu tố kỹ thuật có lợi cho LPT trong các bảng so sánh chính:
  > - Cấu hình chuẩn hóa ban đầu của mô hình Ouro khiến mô hình vòng lặp thông thường hoạt động kém tối ưu. Khi thử nghiệm thêm 5 biến thể chuẩn hóa khác nhau, biến thể chuẩn hóa tốt nhất giúp mô hình vòng lặp thông thường giảm được từ 0.13 đến 0.76 PPL.
  > - Khi áp dụng chuẩn hóa tối ưu này cho cả hai bên, khoảng cách chênh lệch giữa LPT và mô hình vòng lặp thường thu hẹp lại, còn **0.31 đến 0.37 PPL** ở quy mô S và **0.06 đến 0.43 PPL** ở quy mô M.
  > - Ngoài ra, chỉ có quy mô S được chạy lặp lại 4 hạt giống; các quy mô M và L chỉ được thực hiện một lần chạy do giới hạn tài nguyên tính toán."
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào bảng kiểm định t]`: Nêu rõ mức cải thiện PPL theo số vòng lặp 2, 3, 5.
  - `[Chỉ vào mục chuẩn hóa]`: Giải thích rõ khoảng cách hiệu năng thực tế khi chuẩn hóa được tối ưu cho baseline.

---

### Slide 10: 10. LPT: Hạn chế, phản biện và kết luận
- **Slide trong TeX:** `lpt_paper.tex` (Slide 10) & `main.tex` (Slide kết thúc & Q&A)
- **Thời lượng:** ~2.5 phút
- **Mục tiêu:** Tóm tắt 3 hạn chế chính của công trình, kết luận các đóng góp cốt lõi và định hướng phát triển tiếp theo.
- **Lời thoại:**
  > "Để tổng kết bài báo, em xin nêu rõ **ba hạn chế kỹ thuật chính**:
  >
  > 1. **Chi phí tính toán chưa tối ưu:** Ở cấu hình 5 vòng lặp, mô hình tiêu tốn lượng FLOPs giải mã gấp 4.3 đến 4.8 lần so với Transformer chuẩn. Các so sánh hiện tại khớp về số lượng tham số nhưng chưa khớp về tổng lượng FLOPs tính toán. Ngoài ra, mọi token đều phải chạy qua cùng một số vòng lặp cố định mà chưa có cơ chế điều chỉnh linh hoạt.
  > 2. **Giới hạn về quy mô và ngữ cảnh thực nghiệm:** Thử nghiệm mới dừng lại ở quy mô tối đa 1 tỷ tham số, trên một tập dữ liệu FineWeb-Edu với ngữ cảnh 4096 token, chưa được kiểm chứng trên các ngữ cảnh siêu dài (32k hay 128k token). Chênh lệch độ chính xác zero-shot dưới 1 điểm phần trăm cần được xem xét thận trọng do phương sai thực nghiệm.
  > 3. **Chưa tối ưu hóa siêu tham số:** Toàn bộ thực nghiệm sử dụng siêu tham số mặc định của nanoGPT, và mã nguồn nghiên cứu chưa được công bố công khai tại thời điểm bài báo xuất bản.
  >
  > ---
  >
  > **Kết luận:**
  > - Cơ chế chia sẻ bộ nhớ đóng vai trò như một thiên kiến quy nạp (inductive bias) có lợi cho kiến trúc Transformer vòng lặp, giúp cải thiện chất lượng dự đoán chứ không chỉ đơn thuần là giải pháp tiết kiệm bộ nhớ.
  > - Mô hình lai H-LPT thiết lập một biên cân bằng mới giữa chất lượng và bộ nhớ ngữ cảnh: đạt PPL thấp hơn Transformer tiêu chuẩn trong khi giảm từ 76% đến 79% dung lượng KV cache.
  > - **Hướng nghiên cứu tiếp theo:** Phát triển cơ chế dừng vòng lặp thích ứng theo từng token (adaptive compute per token), nghiên cứu kiến trúc xen kẽ nhiều vòng trạng thái và vòng dự đoán, và mở rộng đánh giá trên ngữ cảnh siêu dài.
  >
  > Em xin chân thành cảm ơn quý thầy cô và các bạn đã chú ý theo dõi bài thuyết trình. Em xin sẵn sàng nhận các câu hỏi thảo luận ạ!"
- **Ghi chú thao tác (Cues):**
  - `[Chỉ vào 3 gạch đầu dòng hạn chế]`: Thể hiện góc nhìn phản biện khách quan, khoa học.
  - `[Chuyển sang slide Q&A]`: Kết thúc bài trình bày rõ ràng, sẵn sàng cho phần thảo luận.
- **Câu hỏi phản biện dự kiến (Q&A):**
  - *Hỏi:* Nếu FLOPs giải mã của LPT gấp hơn 4 lần Transformer chuẩn, tại sao người ta vẫn muốn dùng nó trong thực tế?
  - *Đáp:* Trong các hệ thống phục vụ mô hình ngôn ngữ lớn (LLM serving), nút thắt cổ chai lớn nhất thường là băng thông bộ nhớ và dung lượng VRAM dành cho KV cache (memory-bound), chứ không phải năng lực tính toán FLOPs của nhân GPU (compute-bound). Việc giảm dung lượng KV cache tới 4.7 lần cho phép tăng kích thước batch phục vụ đồng thời lên gấp nhiều lần, từ đó tối ưu hóa thông lượng tổng thể của hệ thống phần cứng.

