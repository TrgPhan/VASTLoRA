# Week 4: RIFT-LoRA competitor review

Ngày khóa kết quả: 2026-08-26.

## Kết luận ngắn

RIFT tiếp tục **GO cho hướng nghiên cứu**, nhưng claim phải là **objective safety
cho delayed updates**, không phải "accuracy cao nhất trong mọi thí nghiệm".

- Trên SST-2, `alignfed_calibration` cao hơn RIFT 0.038 điểm phần trăm accuracy,
  một chênh lệch không có ý nghĩa thực tế với 6 seed. RIFT vẫn có loss thấp hơn
  trên 6/6 seed, harmful-update rate thấp hơn 3 lần, và không có harmful late
  update trong 36 event có `tau >= 8`.
- Trên QNLI, RIFT đứng đầu cả accuracy và loss. So với calibration control mạnh
  nhất, RIFT thắng accuracy và loss trên 3/3 seed.
- FedRot, GLoRA-cache và FedSteer-cache không sửa được objective mismatch của
  delayed update trong simulator này. Chúng xử lý gauge hoặc subspace geometry,
  nhưng một update "hợp hình học" vẫn có thể tăng loss hiện tại.

## Thiết kế công bằng

Tất cả phương pháp dùng cùng model, partition, async event trace design, client
rank, latency, local optimizer, số local step, evaluation set và seed. Chỉ
aggregation trajectory thay đổi.

Simulator có `buffer_size=1`: server áp dụng từng delayed update ngay khi nó về.
Do đó mức độ fidelity phải được ghi rõ:

| Method | Mức độ fidelity trong simulator |
|---|---|
| FedEx | Faithful: exact intrinsic innovation chính là `raw` |
| FedRot | Faithful Procrustes operator, cộng async interpolation được công khai |
| GLoRA-cache | Async adaptation từ latest-client consensus projector; không phải giao thức synchronous GLoRA đầy đủ |
| FedSteer-cache | Dynamic cached-vector projection; không phải inactive-client replay protocol đầy đủ |
| AlignFed-calibration | Whole-update calibration/freshness control; không có version-group centering và learned feature transform đầy đủ |
| RIFT | Frozen proposed implementation |

Không được dùng bảng dưới đây để viết rằng RIFT đã đánh bại implementation chính
thức của GLoRA, FedSteer hoặc AlignFed trên benchmark của các paper đó.

## SST-2: 6 seed, 120 measured updates mỗi method

Non-IID label shards, heterogeneous rank 4/8/16, high staleness; seed
59/71/89/101/113/127.

| Method | Final accuracy | Final loss | Harmful | Late harmful (`tau >= 8`) |
|---|---:|---:|---:|---:|
| AlignFed-calibration control | **73.815%** | 0.543452 | 12.50% | 8.33% |
| **RIFT** | 73.777% | **0.543222** | **4.17%** | **0.00%** |
| Projection/MTIP-style | 73.433% | 0.546267 | 56.67% | 52.78% |
| FedSteer-cache | 73.108% | 0.549336 | 51.67% | 44.44% |
| FedRot | 73.051% | 0.547290 | 50.00% | 44.44% |
| GLoRA-cache | 72.649% | 0.551434 | 49.17% | 44.44% |
| Freshness | 72.611% | 0.552068 | 50.00% | 44.44% |
| VAST | 72.515% | 0.551310 | 50.83% | 44.44% |
| FedEx/exact | 72.496% | 0.553451 | 55.00% | 52.78% |

Paired với calibration control: RIFT thấp hơn 0.038 pp accuracy trung bình,
nhưng giảm loss 0.000230 và thắng loss 6/6 seed. Đây là trade-off hòa accuracy,
RIFT thắng safety/loss; không phải RIFT thắng tuyệt đối.

## QNLI: 3 seed, 60 measured updates mỗi method

Non-IID/high-staleness, seed 131/149/163.

| Method | Final accuracy | Final loss | Harmful | Late harmful (`tau >= 8`) |
|---|---:|---:|---:|---:|
| **RIFT** | **74.121%** | **0.538466** | **1.67%** | **5.56%** |
| AlignFed-calibration control | 73.991% | 0.539336 | **1.67%** | **5.56%** |
| FedRot | 71.973% | 0.560158 | 48.33% | 55.56% |
| Projection/MTIP-style | 71.615% | 0.563886 | 45.00% | 38.89% |
| GLoRA-cache | 71.549% | 0.565447 | 45.00% | 38.89% |
| Freshness | 71.517% | 0.565783 | 48.33% | 44.44% |
| VAST | 71.419% | 0.565118 | 46.67% | 44.44% |
| FedSteer-cache | 71.354% | 0.566204 | 48.33% | 44.44% |
| FedEx/exact | 70.475% | 0.573961 | 46.67% | 44.44% |

RIFT thắng calibration control 0.130 pp accuracy và giảm loss 0.000870; cả hai
metric đều thắng 3/3 seed. Với các baseline còn lại, RIFT tăng accuracy từ
2.148 đến 3.646 pp và thắng loss 3/3 seed.

## RIFT đang thêm giá trị gì?

1. FedEx chứng minh exact reconstruction không đủ: algebra đúng nhưng update có
   thể sai objective tại version hiện tại.
2. FedRot chứng minh gauge alignment không đủ: factor cùng orientation không có
   nghĩa là delayed update còn là descent direction.
3. GLoRA/FedSteer controls chứng minh cached/common subspace không đủ: geometry
   ổn định không tự cung cấp dấu của objective utility.
4. Whole-update calibration gần bắt kịp RIFT, chứng minh server calibration là
   thành phần mạnh. Tuy nhiên SST-2 vẫn còn 12.5% harmful updates trên held-out
   eval, trong khi rank-wise filtering của RIFT giảm còn 4.17%.
5. Phần mới cần bảo vệ là **component-level objective filtering trước paired
   acceptance gate**. Gate không chỉ quyết định nhận/bỏ toàn update; filter có
   thể giữ phần có predicted descent signal và bỏ singular component gây hại.

## Verdict và việc còn thiếu

**GO cho thesis prototype** với câu hỏi:

> Can a calibration-assisted, gauge-invariant rank-wise safety layer reduce the
> harmful effect of late updates in heterogeneous-rank asynchronous FedLoRA
> while preserving optimization progress?

Chưa đủ để claim publication-grade superiority. Các gate tiếp theo:

1. Tăng QNLI lên ít nhất 6 seed.
2. Chạy một generative task và báo token NLL/perplexity, không chỉ classification.
3. Calibration-shift và calibration-size ablation để đo overfitting của gate.
4. Faithful buffered AlignFed và synchronous cohort GLoRA ở simulator phù hợp.
5. Báo server FLOPs, calibration forward/backward count, memory và communication.
6. Chạy 3B với frozen hyperparameters; không tune trên test output.

## Artifacts

- Runner: `scripts/run_week4_competitor_matrix.py`
- Matrix: `configs/week4_rift_competitor_matrix.json`
- Analyzer: `scripts/analyze_week4_competitors.py`
- SST-2 report: `outputs/week4_rift_competitor_analysis/competitor_report.md`
- QNLI report: `outputs/week4_rift_qnli_competitor_analysis/competitor_report.md`

## Primary references

- FedEx-LoRA: https://arxiv.org/abs/2410.09432
- FedRot-LoRA code: https://github.com/haoran-zh/FedRot-LoRA
- GLoRA: https://arxiv.org/abs/2605.06733
- AlignFed: https://arxiv.org/abs/2606.08197
- FedSteer paper/code: https://proceedings.mlr.press/v337/zhang26d.html and
  https://github.com/haoran-zh/FedSteer
