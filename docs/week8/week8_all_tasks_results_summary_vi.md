# Week 8 - Tong hop ket qua Qwen2.5-1.5B

Ngay cap nhat: 2026-09-06

Tat ca cac dong dung backbone `Qwen/Qwen2.5-1.5B-Instruct`. Ten trong bang
la aggregation/filtering method, khong phai backbone khac nhau. Cot `dev` va
`held-out` duoc tach rieng de khong con nham 87.50% SST-2 dev voi 90.28%
SST-2 held-out.

## Accuracy

Don vi phan tram, cang cao cang tot.

| Method | SST-2 dev | SST-2 held-out | QNLI held-out | MNLI-m held-out | MNLI-mm held-out |
|---|---:|---:|---:|---:|---:|
| RIFT | **87.50** | **90.28** | 78.993 | 80.556 | 75.868 |
| Spectral filter | **87.50** | **90.28** | 78.993 | **80.903** | **76.910** |
| AlignFed calibration | **87.50** | 89.58 | **79.167** | 80.035 | 76.215 |
| FedRot | 73.61 | 88.89 | 61.458 | 64.757 | 63.368 |
| FedEx | 86.81 | - | - | - | - |
| Freshness | 86.81 | - | - | - | - |
| VAST | 86.81 | - | - | - | - |
| MTiP adaptive | 87.15 | - | - | - | - |

## Class NLL

Cang thap cang tot.

| Method | SST-2 dev | SST-2 held-out | QNLI held-out | MNLI-m held-out | MNLI-mm held-out |
|---|---:|---:|---:|---:|---:|
| RIFT | 0.321890 | **0.213775** | 0.441761 | 0.559449 | 0.587634 |
| Spectral filter | **0.321652** | 0.214077 | 0.440928 | **0.545385** | **0.573481** |
| AlignFed calibration | 0.328915 | 0.219613 | **0.440826** | 0.554471 | 0.582853 |
| FedRot | 0.559643 | 0.273088 | 1.114409 | 1.656944 | 1.685658 |
| FedEx | 0.339258 | - | - | - | - |
| Freshness | 0.345164 | - | - | - | - |
| VAST | 0.345806 | - | - | - | - |
| MTiP adaptive | 0.352232 | - | - | - | - |

## Harmful update rate

Don vi phan tram measured updates, cang thap cang tot.

| Method | SST-2 dev | SST-2 held-out | QNLI held-out | MNLI-m held-out | MNLI-mm held-out |
|---|---:|---:|---:|---:|---:|
| RIFT | **0.00** | **4.17** | **32.292** | **13.542** | **13.542** |
| Spectral filter | 4.17 | 8.33 | 34.375 | 28.125 | 28.125 |
| AlignFed calibration | 6.25 | 10.42 | 36.458 | **13.542** | **13.542** |
| FedRot | 56.25 | 52.08 | 62.500 | 70.833 | 70.833 |
| FedEx | 39.58 | - | - | - | - |
| Freshness | 41.67 | - | - | - | - |
| VAST | 35.42 | - | - | - | - |
| MTiP adaptive | 41.67 | - | - | - | - |

## Late harmful update rate

Chi bao cao held-out; cang thap cang tot.

| Method | SST-2 held-out | QNLI held-out | MNLI-m held-out | MNLI-mm held-out |
|---|---:|---:|---:|---:|
| RIFT | **0.00** | **12.500** | 8.333 | 8.333 |
| Spectral filter | **0.00** | 29.167 | 33.333 | 33.333 |
| AlignFed calibration | 8.33 | 37.500 | **4.167** | **4.167** |
| FedRot | 25.00 | 33.333 | 41.667 | 41.667 |

## Protocol

- SST-2 dev: seeds `2101-2103`, offset 0, 4 warmup + 16 measured returns.
- SST-2 held-out: seeds `2201-2203`, offset 128, cung ngan sach returns.
- QNLI/MNLI held-out: seeds `3201-3206`, offset 64, 4 warmup + 16 measured
  returns, 96 eval examples, 4 late events/run va full client coverage.
- QNLI/MNLI dung 72/72 run tai commit sach `1e8bfb9`; khong cherry-pick seed.
- Tat ca task deu dung label-shard non-IID, heterogeneous rank va compute time.
- Spectral filter va AlignFed calibration la matched controls trong simulator,
  khong phai official full-paper implementation.

Chi SST-2 dev co du ca tam method. Dau gach `-` nghia la chua co held-out
evidence, khong phai accuracy/NLL bang 0.

## Ket luan

- RIFT co harmful rate thap nhat hoac dong thap nhat tren ca bon held-out task.
- RIFT khong phai accuracy/NLL winner tren QNLI va MNLI. Spectral filter thang
  RIFT ro nhat tai MNLI-mm accuracy: `+1.042 pp` cho Spectral, paired CI95 cua
  RIFT gain la `[-2.019, -0.064]`.
- AlignFed calibration co late harmful thap hon RIFT tren ca hai MNLI slice.
- RIFT van thang FedRot rat manh, nhung dieu do khong du de pass claim "tot
  hon moi doi thu".
- Formal held-out verdict cho gia thuyet manh/config hien tai la **`NO_GO`**.
  Ket qua day du nam trong
  `docs/week8/local_1_5b_remaining_tasks_confirmation_results_vi.md`.
- Day van chi la bang chung local 1.5B. Khong duoc suy dien thanh ket luan 3B
  hoac claim thang official Spectral Surgery/AlignFed.
