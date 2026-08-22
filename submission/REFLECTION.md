# Reflection — Lab 21

**Họ tên:** Ngô Việt Anh — **MSSV:** 2A202601579

## 1. Điều gì làm tôi ngạc nhiên nhất?

Điều làm tôi ngạc nhiên nhất là prompt tối ưu đạt target 0.765 và latency 1045.2 ms,
trong khi fine-tune chỉ đạt 0.535 và chậm hơn. Trước khi xem đủ bốn nhóm metric, tôi dễ
nghĩ rằng adapter đã thành công vì format đạt 0.99 và output nhìn giống JSON. Phân tích
từng mẫu mới cho thấy model thường hiểu đúng ý nhưng tự tạo tên nhãn như `hoi_tien`,
`hỏi_thông_tin` hoặc `thấp`, không tuân thủ enum mà hệ thống yêu cầu.

## 2. Tôi mất nhiều thời gian nhất ở đâu? Nó có phải chỗ tôi dự đoán không?

Phần tốn thời gian nhất là các lượt sinh để đo baseline và chấm bốn adapter, không chỉ
là bước train. Bốn run train mất tổng cộng khoảng 32 phút, còn generation phải load lại
model và chạy đủ tập target/regression nhiều lần. Ban đầu tôi dự đoán huấn luyện sẽ chiếm
gần như toàn bộ thời gian; thực tế việc đánh giá công bằng và lưu đủ bằng chứng cũng là
một phần lớn của chi phí thí nghiệm.

## 3. Trước lab này tôi tin điều gì về fine-tuning mà giờ tôi không còn tin?

Tôi không còn tin rằng fine-tuning trên dữ liệu đúng miền đương nhiên sẽ tốt hơn base
model có prompt tốt. Run này cho kết quả ngược lại và cổng hồi quy đã đúng khi từ chối
deploy. Tôi cũng không còn coi train loss thấp hơn là bằng chứng đủ: `attn_only` có loss
tốt nhất, nhưng toàn bộ quyết định vẫn phải dựa trên target, regression, format và
latency; `wrong_lr` cho thấy một cấu hình vẫn chạy hết nhưng hoàn toàn vô dụng.

## 4. Tôi dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?

Tôi dùng AI assistant để đối chiếu repo tham khảo, phát hiện lỗi line ending trên Windows,
cấu hình `.env`/Colab, kiểm tra artefact, tính các chênh lệch và rà số liệu report với
`results/`. AI ban đầu diễn giải checksum khác nhau như dữ liệu hai sinh viên khác nhau;
sau khi so sánh nội dung và `.gitattributes`, nó mới xác định nguyên nhân thật là CRLF.
Điều này nhắc tôi không tin ngay một phỏng đoán chỉ dựa trên hash, mà phải kiểm tra byte,
nội dung và cơ chế checkout trước khi sửa dữ liệu.

## 5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên tôi làm là gì?

Tôi sẽ định nghĩa acceptance gate và đóng băng một tập eval đại diện trước khi train.
Sau đó tôi đo base model với cả prompt hiện tại và prompt tối ưu, bao gồm chất lượng tác
vụ, regression, format, latency và chi phí. Chỉ khi khoảng cách còn lại đủ lớn và prompt
không giải quyết được, tôi mới đầu tư vào fine-tuning; như lab này cho thấy, đôi khi câu
trả lời đúng về kỹ thuật là không deploy adapter.
