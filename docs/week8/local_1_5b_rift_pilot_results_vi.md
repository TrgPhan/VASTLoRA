# Week 8 - Qwen2.5-1.5B local RIFT pilot

Ngay chay: 2026-09-04 den 2026-09-05

Status: preliminary GO cho implementation va objective; chua phai thesis GO.

## Cau hoi

Kiem tra tren GPU 4 GB xem RIFT co chay that, co giu accuracy va class NLL trong
non-IID + heterogeneous rank + high staleness, va co giam harmful update so voi
cac control manh hay khong.

## Setup

- Model: `Qwen/Qwen2.5-1.5B-Instruct`, QLoRA 4-bit, `q_proj` va `v_proj`.
- Task: SST-2, label-shard non-IID, rank client `[2, 4, 8, 4]`, compute time
  `[1, 2, 5, 10]`.
- Moi run: 4 warmup + 16 measured returns, 96 validation examples.
- Development: seeds 2101-2103, eval offset 0, 8 methods.
- Holdout: seeds 2201-2203, eval offset 128, 4 doi thu manh.
- Objective chinh: candidate-normalized `class_nll` cho component scoring,
  calibration gate, harmful monitor va final comparison.
- Absolute label NLL, sequence NLL va EOS NLL chi dung de debug.
- Peak CUDA memory quan sat: khoang 1.96-2.25 GiB.

## Development

| Method | Mean accuracy | Mean class NLL | Harmful | Late harmful | Acceptance |
|---|---:|---:|---:|---:|---:|
| RIFT | 87.50% | 0.321890 | 0.00% | 0.00% | 97.92% |
| Spectral filter | 87.50% | 0.321652 | 4.17% | 0.00% | 100.00% |
| AlignFed calibration | 87.50% | 0.328915 | 6.25% | 8.33% | 58.33% |
| FedEx | 86.81% | 0.339258 | 39.58% | 16.67% | 100.00% |
| Freshness | 86.81% | 0.345164 | 41.67% | 25.00% | 100.00% |
| VAST | 86.81% | 0.345806 | 35.42% | 16.67% | 100.00% |
| MTiP adaptive | 87.15% | 0.352232 | 41.67% | 58.33% | 100.00% |
| FedRot | 73.61% | 0.559643 | 56.25% | 50.00% | 100.00% |

Spectral filter co class NLL tot hon RIFT rat nho tren development. Khong duoc
dung ba seed nay de tuyen bo RIFT thang filter-only ablation.

## Held-out confirmation

| Method | Mean accuracy | Mean class NLL | Harmful | Late harmful | Acceptance | Best accuracy | Best class NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| RIFT | 90.28% | 0.213775 | 4.17% | 0.00% | 93.75% | 91.67% | 0.206346 |
| Spectral filter | 90.28% | 0.214077 | 8.33% | 0.00% | 100.00% | 91.67% | 0.206423 |
| AlignFed calibration | 89.58% | 0.219613 | 10.42% | 8.33% | 70.83% | 89.58% | 0.212729 |
| FedRot | 88.89% | 0.273088 | 52.08% | 25.00% | 100.00% | 91.67% | 0.206840 |

Paired theo seed:

- So voi Spectral filter: accuracy hoa 3/3; class NLL RIFT thang 2, hoa 1;
  harmful RIFT thang 2, hoa 1.
- So voi AlignFed calibration: accuracy RIFT thang 1, hoa 2; class NLL thang
  3/3; harmful thang 2, hoa 1.
- So voi FedRot: accuracy va class NLL RIFT thang 2/3; harmful thang 3/3.

FedRot co mot single run manh tai seed 2201 (91.67% accuracy, class NLL
0.206840), nhung dao dong lon va co harmful/late-harmful cao. RIFT seed 2202
cung dat 91.67% accuracy va class NLL thap hon (0.206346). Vi vay khong duoc
cherry-pick seed 2201 de thay cho mean va paired result.

## Ket luan dung muc

RIFT dat muc preliminary GO tren SST-2 1.5B: no khong hy sinh mean accuracy,
class NLL tot hon ba control trong held-out mean, va harmful update thap nhat.
Ket qua cung cho thay calibration gate co loi ich nho nhung lap lai duoc so voi
filter-only: class NLL thang 2/3 held-out seed va harmful thang 2/3.

Chua the ket luan thesis GO vi moi co mot task, 3 held-out seed, 16 measured
returns va runner nay van immediate async (`buffer_size=1`). Can chay matrix 3B
da dong bang tren SST-2/QNLI/MNLI va 6 confirmation seeds. Cac output local tren
duoc tao trong luc code objective dang la uncommitted worktree, nen chi la pilot;
official matrix bat buoc ghi clean Git provenance.
