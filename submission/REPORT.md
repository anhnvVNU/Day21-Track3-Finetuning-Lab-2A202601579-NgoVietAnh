# Lab 21 — Evaluation Report

**Họ tên**: Ngô Việt Anh

**MSSV**: 2A202601579

**Ngày chạy**: 22/08/2026

**Tier**: `T4`

**Base model**: `unsloth/Qwen3.5-4B`

**GPU thực tế**: NVIDIA T4 16 GB

> Mọi số liệu trong báo cáo này được lấy từ các artefact trong `results/`.

---

## 1. Setup

| Thuộc tính | Giá trị |
|---|---|
| Dataset | 250 ticket chăm sóc khách hàng tiếng Việt → JSON triage |
| Train / validation | 225 / 25, split cố định seed 42 |
| Target eval / regression eval | 50 / 15, không dùng `EVAL_LIMIT` |
| `max_length` | 256; p95 đo được là 98 token |
| `MASK_MODE` | `assistant-only` |
| Epochs / `max_steps` | 1 / 15 |
| LoRA chính | `text-linear`, r=16, alpha=32, LR=1e-4, fp16 |
| Effective batch | 16, nhỏ hơn giới hạn 32 của rubric |

Chat template **có giữ khối `<think>`**. `template_check.json` ghi nhận đủ thẻ mở và
nội dung reasoning, với verdict `reasoning preserved — safe to train on traces`; do đó
không cần sửa template trước khi tạo loss mask. Corpus hiện tại chỉ có câu trả lời JSON,
không có reasoning trace thật, nên `assistant-only` là lựa chọn phù hợp cho phần core.

---

## 2. Mask proof (NB1)

| Thuộc tính | Kết quả |
|---|---|
| `n_supervised / n_total` | 39 / 94 |
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi không nằm trong loss | `true` |

Đầu đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Tỷ lệ supervised 41.49% thấp hơn nhiều ngưỡng lỗi 95%, đồng thời hai assert đều xanh.
Điều này chứng minh prompt và câu hỏi không bị dùng làm nhãn, còn câu trả lời JSON và
token kết thúc hội thoại thực sự đóng góp vào loss.

---

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | Target | Regression | Format | Latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3302.0 |
| (b) base + optimized prompt | **0.765** | 0.7578 | **1.000** | **1045.2** |
| (c) LoRA fine-tune | 0.535 | **0.7689** | 0.990 | 1644.6 |

Baseline (b) mạnh hơn (a) rất rõ: target tăng 0.765 và format tăng từ 0 lên 1.0, trong
khi regression được giữ nguyên. Tôi không sửa `OPTIMIZED_PROMPT`; SHA được đóng băng là
`719e74d3b6232053`. Việc prompt tối ưu vừa chính xác hơn vừa nhanh hơn prompt naive cho
thấy phần lớn chi phí của (a) đến từ việc model sinh đầu ra dài hoặc sai định dạng. Đây
là mốc công bằng mà fine-tune phải vượt, không phải một baseline cố ý làm yếu.

Fine-tune không vượt mốc này: target giảm 0.230 và latency tăng 599.4 ms so với (b).
Regression tăng nhẹ 0.0111 nhưng không bù được tổn thất trên tác vụ mục tiêu.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | Vị trí | r | Trainable | LR | Train loss | Target | Thời gian (s) | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 1.3422 | 0.535 | 496.0 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | **1.1317** | **0.580** | 420.3 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 2.0513 | 0.000 | 482.6 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 1.5300 | 0.310 | 518.5 | **7.09** |

### 4.1 Vị trí adapter và rank

`attn_only` chỉ ít hơn `correct` 8,192 tham số huấn luyện, tức chênh khoảng 0.025%, nên
phép so sánh ngân sách đạt yêu cầu dưới 5%. Nó thắng `correct` 0.045 điểm target; thứ tự
này cũng giống train loss vì 1.1317 thấp hơn 1.3422. Kết quả không ủng hộ giả định rằng
gắn LoRA lên toàn bộ lớp tuyến tính luôn tốt hơn. Khi giữ gần như cố định ngân sách,
cách phân bổ tham số vào q,v với rank cao có hiệu quả hơn nhẹ trong run này. Tuy nhiên,
độ chênh target chỉ 0.045 và rank phải thay đổi từ 16 lên 283 để khớp ngân sách, nên tôi
không coi đây là bằng chứng phổ quát rằng rank cao là đòn bẩy; cần lặp nhiều seed hoặc
quét rank cố định vị trí để tách hai hiệu ứng rõ hơn.

### 4.2 Learning rate sai

`wrong_lr` chỉ đổi LR từ 1e-4 xuống 1e-5 nhưng final loss tăng từ 1.3422 lên 2.0513,
format và target đều về 0. LR ở thang full fine-tuning quá nhỏ khiến LoRA không học đủ
trong ngân sách 15 step; model tiếp tục sinh đầu ra không theo schema và latency tăng
lên 5104.2 ms khi đánh giá. Nếu chỉ nhìn thấy loss còn giảm theo step mà không biết LR,
tôi có thể kết luận sai rằng chỉ cần train lâu hơn hoặc dữ liệu có vấn đề. Target metric
cho thấy cấu hình này thực tế chưa học được hành vi cần thiết.

### 4.3 QLoRA

QLoRA giảm peak VRAM từ 12.01 GB xuống 7.09 GB, tiết kiệm 4.92 GB, tương đương khoảng
41%. Đổi lại, target giảm từ 0.535 xuống 0.310, final loss tăng 0.1878, thời gian train
tăng 22.5 giây và latency đánh giá tăng từ 1644.6 lên 4198.4 ms. Trên T4, tiết kiệm bộ
nhớ là thật nhưng không cần thiết vì fp16 vẫn vừa 16 GB. Các số đo của tôi vì vậy ủng hộ
khuyến nghị không dùng QLoRA cho model/cấu hình này khi mục tiêu là chất lượng và thời
gian, trừ trường hợp giới hạn VRAM khiến fp16 hoàn toàn không chạy được.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy: FAILED**

- Target Δ: **-0.2300** (`0.535` so với `0.765`)
- Regression Δ: **+0.0111** (`0.7689` so với `0.7578`)
- Format: `0.990`
- `valid_trace_rate`: `0.0`

Tôi không nên deploy adapter này. Cổng thất bại không phải do catastrophic forgetting:
điểm regression thực tế tăng nhẹ 0.0111 và nằm trong tolerance. Nguyên nhân quyết định
là fine-tune thua baseline prompt tối ưu 0.230 điểm trên chính tác vụ target, đồng thời
chậm hơn khoảng 57%. Một epoch giúp tiết kiệm tài nguyên và vẫn cho thấy model đã học
được format, nhưng 15 optimizer step chưa đủ để nội hóa taxonomy cố định. Các dự đoán
sai thường dùng nhãn tiếng Việt tự do như `hỏi_thông_tin`, `giao_hang_chậm`, `hoi_tien`
hoặc dùng giá trị có dấu như `thấp`, thay vì tập enum chính xác. Đây là vấn đề compliance
với label vocabulary chứ không chỉ là JSON syntax, nên format 0.99 không đồng nghĩa chất
lượng cao. Với kết quả hiện tại, prompt engineering là phương án vừa tốt hơn vừa rẻ hơn;
fine-tune chỉ nên thử lại sau khi tăng step/epoch và tăng tỷ lệ ví dụ khó cho taxonomy.
`valid_trace_rate=0` không được dùng để kết luận reasoning collapse vì corpus không chứa
reasoning trace thật.

---

## 6. Phân tích định tính — có cả ca thắng và ca thua

NB2 đóng băng metric tổng nhưng không lưu dự đoán từng mẫu của baseline (b), vì vậy cột
(b) dưới đây ghi mốc aggregate thay vì bịa lại output không còn trong artefact. “FT
thắng/thua” được xác định bằng độ khớp bốn trường với nhãn thật; ba ca thua đều nằm trong
`qualitative.json` và có `ft_score=0.25`.

| # | Ticket rút gọn | Nhãn đúng | (b) prompt | Fine-tune | Nhận xét |
|---:|---|---|---|---|---|
| 1 | Nồi chiên thiếu phụ kiện | `san_pham_loi`, `thap`, đúng product, `trung_tinh` | Không lưu output từng mẫu; target tổng 0.765 | intent=`thiếu phụ kiện`, urgency=`thấp`, score 0.25 | ❌ Dùng mô tả tự do và dấu tiếng Việt thay enum |
| 2 | Chuột không dây hỏi bảo hành | `hoi_thong_tin`, `thap`, đúng product, `tich_cuc` | Không lưu output từng mẫu; target tổng 0.765 | intent=`hỏi_thông_tin`, urgency=`thấp`, score 0.25 | ❌ Hiểu nghĩa nhưng vi phạm vocabulary chính xác |
| 3 | Bình giữ nhiệt chưa nhận hoàn tiền | `hoan_tien`, `trung_binh`, đúng product, `tieu_cuc` | Không lưu output từng mẫu; target tổng 0.765 | intent=`hoi_tien`, urgency=`cao`, thiếu product, score 0.25 | ❌ Sai intent, urgency và thiếu trường đúng |
| 4 | Balo laptop hỏi còn hàng | `hoi_thong_tin`, `thap`, `balo laptop`, `tich_cuc` | Không lưu output từng mẫu; target tổng 0.765 | Khớp đủ 4 trường, score 1.00 | ✅ FT thắng theo ground truth |
| 5 | Tai nghe bluetooth yêu cầu hoàn tiền ngay | `hoan_tien`, `cao`, đúng product, `trung_tinh` | Không lưu output từng mẫu; target tổng 0.765 | Khớp đủ 4 trường, score 1.00 | ✅ FT thắng theo ground truth |

Mẫu chung của các ca thua là model hiểu nội dung ở mức ngữ nghĩa nhưng không tuân thủ
ontology đóng: nó dịch/biến thể tên nhãn thay vì chọn đúng chuỗi được phép. Trường
`urgency` cũng dễ bị suy diễn từ cảm xúc thay vì bám cụm chỉ báo trong dữ liệu. Điều này
giải thích vì sao format gần hoàn hảo nhưng field accuracy chỉ đạt 0.535.

---

## 7. Kết luận và điều tôi học được

**Kết luận.** Tôi không deploy bản fine-tune này. Thí nghiệm cho thấy một adapter có thể
đạt JSON format gần hoàn hảo và không làm giảm regression nhưng vẫn không tạo ra giá trị
so với một prompt được thiết kế tốt. Baseline (b) đạt target 0.765, nhanh 1045.2 ms,
trong khi adapter chỉ đạt 0.535 và mất 1644.6 ms. Nguyên nhân trực tiếp là model chưa
học chắc vocabulary của bốn trường sau 15 step: nhiều output đúng ý nhưng sai chính tả
ontology, và metric chấm đúng khi coi đây là lỗi vì hệ thống downstream cần enum chính
xác. Trong các đòn bẩy đã đo, learning rate là đòn bẩy rõ nhất: giảm 10 lần làm target
sụp về 0. Vị trí adapter cũng có ảnh hưởng vì `attn_only` thắng nhẹ khi ngân sách tham số
được khớp, nhưng một run chưa đủ để khẳng định tổng quát. QLoRA giải quyết VRAM nhưng
đổi bằng chất lượng và thời gian không phù hợp trên T4 này. Mask đúng là điều kiện nền
tảng: nếu prompt lọt vào loss thì mọi so sánh phía sau mất ý nghĩa, nhưng mask đúng một
mình không bảo đảm fine-tune sẽ thắng. Bước tiếp theo hợp lý là giữ nguyên eval/prompt
đã đóng băng, tăng lên 2 epoch và bổ sung các ví dụ hard-negative tập trung vào enum,
sau đó chỉ deploy nếu vượt 0.765 mà regression vẫn nằm trong tolerance.

**Ba điều tôi học được:**

1. Fine-tune không mặc nhiên thắng prompt engineering; baseline prompt tối ưu phải được
   đo và đóng băng trước khi train.
2. JSON hợp lệ chưa đủ: với tác vụ tích hợp hệ thống, sai một chuỗi enum cũng là lỗi có
   tác động thực, dù câu trả lời vẫn “có vẻ đúng” với người đọc.
3. Train loss chỉ là tín hiệu tối ưu hóa. Quyết định cấu hình phải dựa trên target,
   regression, format, latency và VRAM cùng lúc.

**Nếu có thêm 2 giờ**, tôi sẽ chạy lại `correct` ở 2 epoch trên cùng split và seed, thêm
oversampling cho những nhãn dễ bị paraphrase, rồi đánh giá lại đủ 50/15 mẫu. Tôi cũng sẽ
lưu prediction từng mẫu của baseline (b) để phân tích thắng/thua trực tiếp thay vì chỉ có
metric tổng.

---

## Phụ lục — phần thưởng

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 Hugging Face Hub — chưa phát hành công khai
