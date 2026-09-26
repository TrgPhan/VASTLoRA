# Week 8 Spectral FedLoRA Confirmation

Ngay cap nhat: 2026-09-26.

## Nguon va tinh toan ven

- Kaggle kernel: `trngphanquang/notebook08d0fd202f`.
- Output moi nhat: `outputs/kaggle_downloads/notebook08d0fd202f_latest`.
- Result root: `week8_qwen15b_spectral_v1_confirmation`.
- Code commit: `2188a5505c4acc668c29fa743d86d90a6829f7c4`.
- Matrix SHA256: `c7751a63425f40c4dc4e627a602a1aa459a50a34dcc21418c7a32b6d1c650da4`.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit.
- Regime: non-IID label shard, heterogeneous rank, high staleness.
- Tasks: SST-2, QNLI, MNLI-m, MNLI-mm.
- Seeds: `6101-6106`; 1024 held-out examples/run.
- Returns: 8 warmup + 32 measured.

Audit dat `72/72` result, `12/12` task-method groups va `6/6` seeds/group.
Moi result co schema >= 5, clean worktree va dung commit. Accuracy/Class NLL
duoc tinh lai tu CSV 1024 dong; harmful/late harmful duoc tinh lai tu 32
measured returns. Held-out examples va event schedule trung nhau giua ba method
cho tung task/seed. Khong phat hien traceback, OOM hay job fail.

Day la cac paper operator duoc thich nghi vao immediate-async Week 8 protocol,
khong phai reproduction synchronous benchmark cua paper. Chi tiet fidelity:
`docs/week8/spectral_fedlora_fidelity_vi.md`.

## Accuracy

Mean +/- sample standard deviation, percentage point. Cao hon tot hon.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| FlexLoRA | **83.757 +/- 2.793** | **73.665 +/- 4.328** | 70.638 +/- 3.521 | 70.671 +/- 3.140 |
| FLoRIST | 82.829 +/- 2.618 | 73.226 +/- 4.215 | 71.191 +/- 3.282 | 71.273 +/- 3.174 |
| FLoRG | 77.230 +/- 0.911 | 72.070 +/- 1.369 | **73.486 +/- 0.819** | **72.493 +/- 1.372** |

## Class NLL

Mean +/- sample standard deviation. Thap hon tot hon.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| FlexLoRA | **0.366979 +/- 0.041030** | **0.522789 +/- 0.064136** | 0.680172 +/- 0.054884 | 0.687575 +/- 0.074353 |
| FLoRIST | 0.378395 +/- 0.039805 | 0.528438 +/- 0.057718 | 0.672029 +/- 0.055242 | 0.681210 +/- 0.078796 |
| FLoRG | 0.464317 +/- 0.014382 | 0.535864 +/- 0.013703 | **0.653796 +/- 0.023395** | **0.673823 +/- 0.039049** |

## Harmful Update Rate

Mean percentage tren 32 measured returns. Thap hon tot hon.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| FlexLoRA | 35.42 | 55.73 | 55.73 | 55.73 |
| FLoRIST | 34.38 | **52.08** | 56.77 | 56.77 |
| FLoRG | **28.65** | 55.73 | **51.04** | **51.04** |

## Late Harmful Update Rate

Mean percentage tren late returns (`staleness >= 4`). Thap hon tot hon.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| FlexLoRA | 71.43 | **35.71** | **57.14** | **57.14** |
| FLoRIST | **69.05** | 38.10 | **57.14** | **57.14** |
| FLoRG | **69.05** | **35.71** | 61.90 | 61.90 |

## Nhan xet

- FlexLoRA manh nhat trong ba method tren SST-2 va QNLI ve accuracy/NLL,
  nhung variance QNLI lon (`4.328` pp).
- FLoRG manh nhat tren MNLI-m va MNLI-mm, voi variance accuracy thap hon hai
  spectral redistribution baseline con lai.
- Khong method nao giai quyet harmful update manh: harmful van khoang
  `28.65%-56.77%`, late harmful `35.71%-71.43%`.
- So voi bang RIFT-Core confirmation matched protocol, RIFT-Core van cao hon
  ca ba method ve mean accuracy va thap hon ve Class NLL tren ca bon task.
- Ket qua nay bo sung baseline coverage, nhung khong tu no chung minh RIFT
  vuot full synchronous implementations trong cac paper goc.
