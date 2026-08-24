# Bảng dataset và benchmark của các đối thủ VAST-LoRA

Ngày cập nhật: 2026-08-11

Mục tiêu của file này là nhìn nhanh: mỗi đối thủ đã chạy trên dataset nào, metric chính là gì, kết quả bao nhiêu, và code có public hay không. Các ô không tìm thấy số liệu rõ ràng trong paper được ghi là `N/A`.

## 1. Bảng GLUE / NLU

| Framework / strategy | Dataset | Benchmark / metric | Accuracy / score được báo cáo | So sánh chính | Code public? |
|---|---|---|---:|---|---|
| GLoRA | MNLI-m, MNLI-mm, SST-2, QQP, QNLI | Average accuracy | 81.91 | Dir(0.1), homogeneous rank; cao hơn FedEx-LoRA 81.33 | N/A |
| GLoRA | MNLI-m, MNLI-mm, SST-2, QQP, QNLI | Average accuracy | 89.36 | Dir(0.5), homogeneous rank; cao hơn FedEx-LoRA 88.43 | N/A |
| GLoRA | MNLI-m, MNLI-mm, SST-2, QQP, QNLI | Average accuracy | 88.34 | Heterogeneous rank, normal rank distribution | N/A |
| GLoRA | MNLI-m, MNLI-mm, SST-2, QQP, QNLI | Average accuracy | 87.05 | Heterogeneous rank, uniform rank distribution | N/A |
| GLoRA | MNLI-m, MNLI-mm, SST-2, QQP, QNLI | Average accuracy | 88.30 | Heterogeneous rank, heavy-tail rank distribution | N/A |
| FedRot-LoRA | SST-2, QNLI, QQP, RTE, MNLI | Average accuracy | 0.8932 | N=3 clients; cao hơn RoLoRA 0.8862 và FedIT 0.8712 | Có |
| FedRot-LoRA | SST-2, QNLI, QQP, RTE, MNLI | Average accuracy | 0.8818 | N=10 clients; cao hơn RoLoRA 0.8786 và FedIT 0.8362 | Có |
| FLoRG | MNLI | Accuracy | 91.27 | RoBERTa-large | N/A |
| FLoRG | QNLI | Accuracy | 92.58 | RoBERTa-large | N/A |
| FLoRG | WNLI | Accuracy | 66.48 | RoBERTa-large | N/A |
| FLoRG | RTE | Accuracy | 71.26 | RoBERTa-large | N/A |
| FLoRG | MNLI | Accuracy | 93.15 | Llama-3.2-3B | N/A |
| FLoRG | QNLI | Accuracy | 93.12 | Llama-3.2-3B | N/A |
| FLoRG | WNLI | Accuracy | 68.73 | Llama-3.2-3B | N/A |
| FLoRG | RTE | Accuracy | 73.84 | Llama-3.2-3B | N/A |
| FedEx-LoRA | CoLA, RTE, MRPC, SST-2, QNLI, STS-B | Average score | 83.79 | RoBERTa-base r=4; cao hơn FedIT 82.31 và FFA-LoRA 81.95 | Có |
| FedEx-LoRA | CoLA, RTE, MRPC, SST-2, QNLI, STS-B | Average score | 85.10 | RoBERTa-large r=4; cao hơn FedIT 84.50 và FFA-LoRA 83.23 | Có |
| SDFLoRA | QNLI, RTE, QQP, MNLI, SST-2 | Average accuracy | 79.48 | Heterogeneous rank; cao hơn FLoRA 77.47 và padding 75.86 | N/A |
| SDFLoRA | RTE | Accuracy | 99.71 | Điểm mạnh nhất được báo cáo | N/A |
| SDFLoRA | QNLI | Accuracy | 96.96 | Điểm mạnh được báo cáo | N/A |
| SDFLoRA | QQP | Accuracy | 87.97 | Điểm mạnh được báo cáo | N/A |
| FSLoRA | QNLI, MRPC, CoLA, MNLI, RTE, SST-2, QQP | Average accuracy | 83.3 | Cao hơn HeteroLoRA 79.6, FlexLoRA 78.9, FLoRA 78.9, RAVAN 76.6 | Có |

## 2. Bảng LLM / generative / instruction benchmark

| Framework / strategy | Dataset | Benchmark / metric | Accuracy / score được báo cáo | So sánh chính | Code public? |
|---|---|---|---:|---|---|
| AlignFed | GSM8K | Accuracy | 52.69 | Llama3-8B, FFA-LoRA + AlignFed | N/A |
| AlignFed | CodeAlpaca | pass@1 | 40.30 | Llama3-8B, FFA-LoRA + AlignFed | N/A |
| AlignFed | CodeAlpaca | pass@10 | 62.80 | Llama3-8B, FFA-LoRA + AlignFed | N/A |
| AlignFed | GSM8K | Accuracy | 88.32 | Qwen3-8B async | N/A |
| AlignFed | CodeAlpaca | pass@1 | 53.66 | Qwen3-8B async | N/A |
| AlignFed | CodeAlpaca | pass@10 | 72.56 | Qwen3-8B async | N/A |
| AlignFed | Dolly | Score | 66.66 | Qwen3-8B async | N/A |
| FedRot-LoRA | GSM8K | Exact-match accuracy | 0.4437 ± 0.009 | Cao hơn FFA-LoRA và FedIT | Có |
| FedRot-LoRA | HumanEval | pass@1 | 0.4088 ± 0.012 | Cao hơn FFA-LoRA và FedIT | Có |
| FedEx-LoRA | GSM8K | Accuracy | 62.62 | Mistral-7B | Có |
| FedEx-LoRA | MATH | Accuracy | 16.54 | Mistral-7B | Có |
| FedEx-LoRA | GSM8K | Accuracy | 76.19 | Gemma-2 9B | Có |
| FedEx-LoRA | MATH | Accuracy | 39.00 | Gemma-2 9B | Có |
| FedEx-LoRA | BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA | Average accuracy | 85.99 | Cao hơn FedIT 83.57 và FFA-LoRA 77.35 | Có |
| FSLoRA | ARC-c, ARC-e, BoolQ, HellaSwag, OBQA, PIQA, SIQA, WinoGrande | Average accuracy | 79.4 | Cao hơn FlexLoRA 77.4, HeteroLoRA 74.6, FLoRA 74.6, RAVAN 74.7 | Có |
| GLoRA | SuperNI in-domain clients | ROUGE-L | 52.26 | Normal rank distribution | N/A |
| GLoRA | SuperNI in-domain clients | ROUGE-L | 49.61 | Uniform rank distribution | N/A |
| GLoRA | SuperNI in-domain clients | ROUGE-L | 50.42 | Heavy-tail rank distribution | N/A |
| GLoRA | SuperNI unseen clients | ROUGE-L | 37.97 | Normal rank distribution | N/A |
| GLoRA | SuperNI unseen clients | ROUGE-L | 34.95 | Uniform rank distribution | N/A |
| GLoRA | SuperNI unseen clients | ROUGE-L | 34.50 | Heavy-tail rank distribution | N/A |
| PreLort | Dolly | Accuracy / ROUGE-L / perplexity | N/A | Paper báo tốt nhất trên MMLU và ROUGE-L, nhưng chưa trích được bảng số rõ ràng | N/A |
| PreLort | Alpaca | Accuracy / ROUGE-L / perplexity | N/A | Paper báo tốt nhất trên MMLU và ROUGE-L, nhưng chưa trích được bảng số rõ ràng | N/A |
| PreLort | 20 Newsgroups | Accuracy / perplexity | N/A | Paper báo competitive, nhưng chưa trích được bảng số rõ ràng | N/A |
| HetLoRA | Multi-session chat data | N/A | N/A | Dataset/backbone không public đầy đủ nên khó reproduce | N/A |
| HetLoRA | Reddit summarization | N/A | N/A | Dataset/backbone không public đầy đủ nên khó reproduce | N/A |

## 3. Dataset giao nhau đáng dùng cho VAST-LoRA

### GLUE nên ưu tiên

| Dataset | Vì sao nên dùng |
|---|---|
| SST-2 | Có trong GLoRA, FedRot-LoRA, FedEx-LoRA, SDFLoRA, FSLoRA |
| QNLI | Có trong hầu hết các paper GLUE |
| RTE | Có trong FedRot-LoRA, FLoRG, FedEx-LoRA, SDFLoRA, FSLoRA |
| MNLI | Có trong GLoRA, FedRot-LoRA, FLoRG, SDFLoRA, FSLoRA |
| QQP | Có trong GLoRA, FedRot-LoRA, FLoRG, SDFLoRA, FSLoRA |

Đề xuất thực nghiệm chính cho VAST-LoRA:

```text
SST-2, QNLI, RTE, MNLI
```

Thêm `QQP` nếu còn compute.

### LLM benchmark nên ưu tiên

| Dataset | Vì sao nên dùng |
|---|---|
| GSM8K | Có trong AlignFed, FedRot-LoRA, FedEx-LoRA; metric accuracy rõ |
| Dolly | Có trong AlignFed và PreLort; phù hợp test instruction-following nhẹ hơn |
| Commonsense suite | Có trong FedEx-LoRA và FSLoRA; gồm BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA |
| HumanEval | Có trong FedRot-LoRA; tốt để test code generation, nhưng tốn setup hơn |

Đề xuất thực nghiệm phụ cho VAST-LoRA:

```text
GSM8K, Dolly subset
```

Nếu cần so mạnh với FedEx-LoRA và FSLoRA thì thêm commonsense suite.

## 4. Ghi chú nguồn

- GLoRA: https://arxiv.org/html/2605.06733v1
- FedRot-LoRA: https://arxiv.org/html/2602.23638v1
- FLoRG: https://arxiv.org/pdf/2602.17095
- FedEx-LoRA: https://aclanthology.org/2025.acl-long.67.pdf
- FSLoRA: https://arxiv.org/html/2501.19389v4
- SDFLoRA: https://arxiv.org/html/2601.11219v2
- AlignFed: https://arxiv.org/pdf/2606.08197
- HetLoRA: https://aclanthology.org/2024.emnlp-main.717.pdf
- PreLort: https://arxiv.org/pdf/2606.15963
