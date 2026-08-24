# Bảng mẫu thí nghiệm VAST-LoRA trên mô hình 1.5B

Ngày cập nhật: 2026-08-11

Mục tiêu: chuẩn bị bảng để tự chạy lại các strategy khả dụng trên cùng mô hình khoảng 1.5B, cùng partition dữ liệu, cùng seed latency/staleness, rồi so với VAST-LoRA. File này là **template chưa có data thực nghiệm của mình**.

## 1. Kết luận chọn benchmark

Với thesis VAST-LoRA, bảng chính nên là **LLM / instruction benchmark**, không phải GLUE. Lý do là thesis đang nghiên cứu asynchronous federated LoRA cho LLM, stale innovation, rank heterogeneity và low-rank transport. GLUE vẫn hữu ích, nhưng nên dùng như benchmark phụ để debug nhanh pipeline, không nên là kết quả trung tâm của thesis.

Đề xuất ưu tiên:

| Mức ưu tiên | Nhóm benchmark | Dataset nên dùng | Vai trò trong thesis |
|---:|---|---|---|
| 1 | LLM / instruction | GSM8K, Dolly subset | Bảng chính cho Qwen2.5-1.5B |
| 2 | LLM / commonsense | BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA | Bảng mở rộng nếu muốn so mạnh với FedEx-LoRA và FSLoRA |
| 3 | GLUE / NLU | SST-2, QNLI, RTE, MNLI | Bảng phụ để sanity check và chạy nhanh |
| 4 | Code generation | HumanEval | Chỉ chạy nếu còn thời gian, vì setup đánh giá tốn hơn |

## 2. Strategy có thể đem về chạy thử

| Nhóm | Strategy / framework | Tình trạng khả dụng | Nên đưa vào benchmark 1.5B? | Lý do |
|---|---|---|---|---|
| Baseline nội bộ | Sync FedAvg-LoRA | Tự implement được | Có, bắt buộc | Mốc synchronous cơ bản |
| Baseline nội bộ | Naive Async-LoRA | Tự implement được | Có, bắt buộc | Mốc async không sửa stale update |
| Baseline nội bộ | Freshness-only Async-LoRA | Tự implement được | Có, bắt buộc | Đối thủ trực tiếp nhất của VAST |
| Baseline nội bộ | Buffered Async-LoRA / FedBuff-style | Tự implement được | Có, bắt buộc | Kiểm tra tác dụng của buffer trong async FL |
| Baseline nội bộ | GLoRA-like + freshness | Cần faithful reimplementation | Có, nên ưu tiên | Kiểm tra VAST có hơn geometry-aware aggregation cộng decay hay không |
| Đối thủ có code | FedEx-LoRA | Có repository public | Có | Liên quan trực tiếp đến exact aggregation trong FedLoRA |
| Đối thủ có code | FedRot-LoRA | Có repository public | Có | Liên quan trực tiếp đến alignment của LoRA factors |
| Đối thủ có code | FSLoRA | Có repository public | Có nếu đủ thời gian | Hợp với rank/resource heterogeneity |
| Đối thủ có code | FLoRA / FedIT / Zero-Padding | Có repository public qua FederatedLLM hoặc FedLLM-Factory | Có nếu cần baseline hetero-rank | Rất phù hợp làm baseline rank heterogeneity |
| Đối thủ chưa chắc code | GLoRA official | Chưa thấy official code rõ | Không chạy trực tiếp, chỉ reimplement lõi | Quan trọng về gauge-aware geometry nhưng cần tự tái hiện |
| Đối thủ chưa chắc code | FLoRG official | Chưa thấy official code rõ | Không ưu tiên đầu tiên | Có liên quan nhưng không async, cần tái hiện thêm |
| Đối thủ chưa chắc code | AlignFed | Chưa thấy official code rõ | Không ưu tiên đầu tiên | Có async nhưng dùng calibration/semantic alignment, lệch khỏi data-free VAST |
| Đối thủ chưa chắc code | SDFLoRA | Chưa thấy official code rõ | Không ưu tiên đầu tiên | Nghiêng về shared/private và privacy hơn là stale subspace transport |
| Đối thủ chưa chắc code | PreLort | Chưa thấy official code rõ | Không ưu tiên | Nghiêng về nested rank/prefix LoRA, không sát thesis core |
| Đối thủ chưa chắc code | HetLoRA | Chưa thấy official code rõ, backbone/data khó public | Không ưu tiên | Có thể trích related work, khó reproduce công bằng |

## 3. Bảng mẫu chính: LLM / instruction benchmark

Chạy cùng mô hình:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Dataset nên bắt đầu:

```text
GSM8K, Dolly subset
```

Nếu đủ thời gian thì thêm:

```text
BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA
```

### Template kết quả chính

| Strategy | Code source | Có chạy được trên 1.5B? | GSM8K acc | Dolly score / ROUGE-L | Commonsense avg acc | Mean | Std | Training time | Peak VRAM | Upload MB/round | Ghi chú |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sync FedAvg-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Baseline synchronous |
| Naive Async-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Async không decay |
| Freshness-only Async-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Decay toàn bộ update theo staleness |
| Buffered Async-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | FedBuff-style |
| FLoRA / FedIT / Zero-Padding | Public repo | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Heterogeneous-rank baseline |
| FedEx-LoRA | Public repo | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Exact aggregation baseline |
| FedRot-LoRA | Public repo | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Rotational alignment baseline |
| FSLoRA | Public repo | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Sketching/rank-resource baseline |
| GLoRA-like + freshness | Reimplementation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Gauge-aware + whole-update freshness |
| VAST-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Method chính |
| VAST-LoRA left-only | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Ablation |
| VAST-LoRA right-only | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Ablation |
| VAST-LoRA geometry-only | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Ablation không dùng freshness residual |

### Template theo staleness regime

| Strategy | Dataset | Low staleness acc | Moderate staleness acc | High staleness acc | Extreme staleness acc | Drop từ low tới high | Ghi chú |
|---|---|---:|---:|---:|---:|---:|---|
| Freshness-only Async-LoRA | GSM8K | TBD | TBD | TBD | TBD | TBD | TBD |
| GLoRA-like + freshness | GSM8K | TBD | TBD | TBD | TBD | TBD | TBD |
| FedEx-LoRA | GSM8K | TBD | TBD | TBD | TBD | TBD | TBD |
| FedRot-LoRA | GSM8K | TBD | TBD | TBD | TBD | TBD | TBD |
| FSLoRA | GSM8K | TBD | TBD | TBD | TBD | TBD | TBD |
| VAST-LoRA | GSM8K | TBD | TBD | TBD | TBD | TBD | TBD |
| Freshness-only Async-LoRA | Dolly subset | TBD | TBD | TBD | TBD | TBD | TBD |
| GLoRA-like + freshness | Dolly subset | TBD | TBD | TBD | TBD | TBD | TBD |
| FedEx-LoRA | Dolly subset | TBD | TBD | TBD | TBD | TBD | TBD |
| FedRot-LoRA | Dolly subset | TBD | TBD | TBD | TBD | TBD | TBD |
| FSLoRA | Dolly subset | TBD | TBD | TBD | TBD | TBD | TBD |
| VAST-LoRA | Dolly subset | TBD | TBD | TBD | TBD | TBD | TBD |

## 4. Bảng mẫu phụ: GLUE / NLU benchmark

GLUE không nên là bảng chính của thesis, nhưng rất hữu ích để test nhanh:

```text
SST-2, QNLI, RTE, MNLI
```

Nếu còn compute:

```text
QQP
```

### Template kết quả GLUE

| Strategy | Code source | Có chạy được trên 1.5B prompt-classification? | SST-2 acc | QNLI acc | RTE acc | MNLI acc | QQP acc | Average | Std | Ghi chú |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Sync FedAvg-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Baseline synchronous |
| Naive Async-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Async không decay |
| Freshness-only Async-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Decay toàn bộ update |
| Buffered Async-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | FedBuff-style |
| FLoRA / FedIT / Zero-Padding | Public repo | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Heterogeneous-rank baseline |
| FedEx-LoRA | Public repo | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Exact aggregation baseline |
| FedRot-LoRA | Public repo | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Rotational alignment baseline |
| FSLoRA | Public repo | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Sketching baseline |
| GLoRA-like + freshness | Reimplementation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Gauge-aware + freshness |
| VAST-LoRA | In-house | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Method chính |

## 5. Bảng diagnostic bắt buộc cho thesis

Accuracy một mình chưa đủ để bảo vệ VAST. Vì thesis claim là stale update compatibility, cần thêm bảng diagnostic sau.

| Strategy | Dataset | Mean acc | Mean staleness | Mean compatibility $\rho_i$ | Energy retained by VAST | Harmful update rate | Bytes uploaded | Server correction time | Kết luận |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Freshness-only Async-LoRA | GSM8K | TBD | TBD | N/A | N/A | TBD | TBD | TBD | TBD |
| GLoRA-like + freshness | GSM8K | TBD | TBD | TBD | N/A | TBD | TBD | TBD | TBD |
| FedEx-LoRA | GSM8K | TBD | TBD | N/A | N/A | TBD | TBD | TBD | TBD |
| FedRot-LoRA | GSM8K | TBD | TBD | TBD | N/A | TBD | TBD | TBD | TBD |
| FSLoRA | GSM8K | TBD | TBD | N/A | N/A | TBD | TBD | TBD | TBD |
| VAST-LoRA | GSM8K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 6. Quyết định thực nghiệm nên đóng băng

Để bám sát thesis, nên đóng băng kế hoạch như sau:

1. Bảng chính: Qwen2.5-1.5B-Instruct trên `GSM8K` và `Dolly subset`.
2. Baseline bắt buộc: Sync FedAvg-LoRA, Naive Async-LoRA, Freshness-only Async-LoRA, Buffered Async-LoRA.
3. Đối thủ public-code nên kéo về trước: FedEx-LoRA, FedRot-LoRA, FSLoRA, FLoRA/FedIT/Zero-Padding.
4. Đối thủ cần tự tái hiện lõi: GLoRA-like + freshness.
5. Bảng GLUE chỉ dùng như sanity benchmark, không đặt làm kết quả chính.
6. Bảng diagnostic về staleness và compatibility là bắt buộc, vì đây mới là bằng chứng trực tiếp cho thesis core.

## 7. Nguồn kiểm tra code public

- FedRot-LoRA: https://github.com/haoran-zh/FedRot-LoRA
- FedEx-LoRA: https://github.com/RaghavSinghal10/fedex-lora
- FSLoRA: https://github.com/wenzhifang/Federated-Sketching-LoRA-Implementation
- FLoRA / FederatedLLM: https://github.com/ATP-1010/FederatedLLM
- FedLLM-Factory: https://github.com/boyi-liu/FedLLM-Factory
