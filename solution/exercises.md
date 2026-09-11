# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> (Chạy bằng Gemini `gemini-3.5-flash-lite` qua endpoint tương thích OpenAI.)
> Ở temperature 0.0 và 0.5, model chọn cùng một chủ đề là cà phê (Việt Nam xuất
> khẩu Robusta số 1, cà phê trứng) với bố cục gần như giống hệt nhau — câu trả
> lời ổn định, "an toàn". Lên 1.0 model chuyển sang chủ đề khác (Hang Sơn Đoòng
> có mây và mưa bên trong) với cách kể giàu hình ảnh hơn, còn ở 1.5 câu trả lời
> nhảy qua nhiều sự thật cùng lúc (hạt điều, hạt tiêu, cà phê, Sơn Đoòng), kém
> tập trung hơn và bắt đầu lẫn chi tiết không chính xác (nhắc cả "hạt dẻ cười"
> như thể xuất xứ Việt Nam). Quy luật: temperature càng cao thì càng đa dạng,
> sáng tạo nhưng càng khó đoán và dễ sai lệch; ngoài ra bản 0.0 bị cắt giữa câu
> vì chạm giới hạn `max_tokens=256`.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt khoảng 0.2. Chatbot hỗ trợ khách hàng cần trả lời nhất quán và
> chính xác: cùng một câu hỏi về chính sách đổi trả thì khách nào cũng phải
> nhận cùng một câu trả lời, và model không nên "sáng tạo" thêm thông tin sai.
> Không để hẳn 0.0 để câu chữ vẫn tự nhiên, bớt cứng nhắc khi trò chuyện.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Mỗi ngày có 10.000 × 3 × 350 = 10,5 triệu token đầu ra. Theo bảng giá trong
> template: GPT-4o = 10.500 × $0.010 = **$105/ngày** (~$3.150/tháng), còn
> GPT-4o-mini = 10.500 × $0.0006 = **$6,3/ngày** (~$189/tháng) → GPT-4o đắt hơn
> khoảng **16,7 lần** (giá input cũng chênh đúng tỉ lệ 0.0025 / 0.00015 ≈ 16,7).
> GPT-4o xứng đáng khi sai sót rất đắt, ví dụ trợ lý phân tích hợp đồng pháp lý
> hoặc tóm tắt hồ sơ y tế cần suy luận nhiều bước. Nên dùng mini cho các tác vụ
> đơn giản, số lượng lớn như phân loại ticket hỗ trợ, trả lời FAQ hay gợi ý
> tiêu đề — chất lượng đủ dùng mà tiết kiệm được hơn 90% chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, model xưng "thầy/cô", gọi người hỏi là "con" và
> giải thích bằng câu chuyện cả lớp cùng ghi chép việc đổi sticker vào những tờ
> giấy giống hệt nhau, mỗi trang giấy là một "block" nối thành cuốn sổ — gần như
> không dùng thuật ngữ. Với persona chuyên gia tài chính, câu trả lời dày đặc
> thuật ngữ (DLT, decentralized, immutable, trustless, hàm băm SHA-256) và được
> chia thành các mục có tiêu đề như một tài liệu kỹ thuật. Độ dài hai bản gần như
> nhau (~185 từ, cả hai đều chạm `max_tokens=256`) nhưng bản chuyên gia tốn nhiều
> token hơn (239 so với 218) vì chứa nhiều thuật ngữ tiếng Anh. Cùng một câu hỏi
> nhưng system prompt quyết định giọng điệu, cách xưng hô, từ vựng, loại ví dụ và
> cấu trúc câu trả lời — model bám theo vai được giao trong toàn bộ phản hồi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Mình dùng một đoạn văn 113 từ giới thiệu về Việt Nam. `count_tokens` (bộ mã
> hóa của gpt-4o) đếm được **154 token**, còn ước lượng 113 / 0.75 = **150,7** →
> chỉ chênh khoảng **2%**. Tuy nhiên cùng nội dung đó dịch sang tiếng Anh
> (87 từ, 529 ký tự) chỉ tốn **104 token**, trong khi bản tiếng Việt ngắn hơn
> (512 ký tự) lại tốn 154 token — nhiều hơn khoảng **48%**; với bản tiếng Anh thì
> ước lượng số từ / 0.75 lại lệch 10%, nên khớp ở bản tiếng Việt là do trùng hợp.
> Tiếng Việt tốn token hơn vì: (1) bộ mã hóa BPE được huấn luyện chủ yếu trên dữ
> liệu tiếng Anh nên từ tiếng Anh thường là 1 token, còn âm tiết tiếng Việt hay bị
> cắt nhỏ; (2) chữ có dấu (ư, ơ, ệ, ặ...) chiếm 2–3 byte UTF-8 và các tổ hợp dấu
> ít gặp hơn nên bị tách thành nhiều mảnh; (3) tiếng Việt đơn âm tiết, mỗi âm tiết
> cách nhau bởi dấu cách, nên cùng một ý cần nhiều "từ" hơn tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi có người đang ngồi chờ đọc câu trả lời, nhất là
> câu trả lời dài như trong chatbot hay trợ lý viết: tổng thời gian sinh không
> đổi, nhưng chữ đầu tiên hiện ra sau chưa đến một giây thay vì phải nhìn màn
> hình trống 5–10 giây, người dùng biết hệ thống đang chạy và có thể dừng sớm nếu
> thấy model trả lời sai hướng. Ngược lại, non-streaming phù hợp hơn khi kết quả
> phải được xử lý trọn vẹn rồi mới dùng được — ví dụ model trả về JSON cần parse,
> tác vụ phân loại chỉ trả về một nhãn, job chạy nền xử lý hàng loạt không có ai
> ngồi xem, hoặc khi cần kiểm duyệt toàn bộ nội dung trước khi hiển thị. Lúc đó
> code non-streaming đơn giản hơn và dễ xử lý lỗi, dễ lấy thống kê token hơn.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tự điều chỉnh theo mức độ sự cố: lỗi thoáng qua thì lần thử
> lại đầu tiên (0,1s) gần như không làm người dùng phải chờ, còn nếu server quá
> tải kéo dài thì thời gian chờ tăng dần (0,2s → 0,4s → 0,8s...), số request gửi
> tới giảm mạnh và server có thời gian hồi phục. Với delay cố định, nếu hàng nghìn
> client cùng gặp lỗi một lúc và cùng chờ đúng 1 giây, chúng sẽ đồng loạt gửi lại
> vào cùng một thời điểm, tạo thành từng "đợt sóng" request (thundering herd).
> Server vừa hồi lại đã bị đánh sập tiếp, lại lỗi, lại retry cùng lúc — vòng lặp
> quá tải không dứt. Thực tế người ta còn cộng thêm một khoảng ngẫu nhiên (jitter)
> vào delay để các client tản ra, không retry trùng nhịp với nhau.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Mình chọn persona trợ giảng cho khóa học AI, system prompt:
> "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."
> - **"trả lời ngắn gọn"**: trợ lý chạy trong terminal nên câu trả lời dài rất
>   khó đọc; câu trả lời ngắn cũng tốn ít token output hơn (output đắt gấp 4 lần
>   input), và vì history được gửi lại mỗi lượt nên câu trả lời ngắn giúp chi phí
>   input của các lượt sau không phình to.
> - **"bằng tiếng Việt"**: học viên hay hỏi lẫn thuật ngữ tiếng Anh (API, token,
>   temperature...), nếu không chỉ định ngôn ngữ model dễ trả lời luôn bằng tiếng
>   Anh. Chỉ định rõ giúp câu trả lời nhất quán đúng đối tượng người học.
> - **"thân thiện"** đặt giọng điệu khuyến khích, phù hợp với người mới học.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ 3 lượt gần nhất: nếu ở lượt 1 mình nói tên
> hoặc nói đang học phần nào, đến lượt 5 thông tin đó đã bị cắt và trợ lý "quên"
> mất. Ngoài ra thống kê chi phí hiện chỉ đếm token của tin nhắn user và reply,
> chưa tính system prompt và history được gửi lại mỗi lượt, nên chi phí thật cao
> hơn con số in ra.
> Cải thiện đề xuất — **tóm tắt hội thoại cũ thay vì bỏ hẳn**: mỗi khi history
> vượt 6 message, lấy các message sắp bị cắt, gọi `call_openai_mini` với prompt
> "Tóm tắt ngắn các thông tin quan trọng trong đoạn hội thoại sau..." rồi lưu kết
> quả vào biến `summary`. Khi ghép messages, chèn thêm
> `{"role": "system", "content": "Tóm tắt hội thoại trước: " + summary}` ngay sau
> persona. Như vậy trợ lý vẫn nhớ thông tin quan trọng từ đầu phiên, mà số token
> gửi đi vẫn được giới hạn; chi phí gọi mini để tóm tắt rất nhỏ.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
