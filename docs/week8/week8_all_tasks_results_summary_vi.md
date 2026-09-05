# Week 8 - Tong hop ket qua Qwen2.5-1.5B

Ngay tong hop: 2026-09-05

Tat ca cac dong dung cung backbone `Qwen/Qwen2.5-1.5B-Instruct`. Ten trong
bang la aggregation/filtering method, khong phai model backbone khac nhau.

## Accuracy

Don vi: phan tram, cang cao cang tot.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| RIFT | **87.50%** | **78.65%** | **67.71%** | **76.04%** |
| Spectral filter | **87.50%** | **78.65%** | 67.19% | **76.04%** |
| AlignFed calibration | **87.50%** | **78.65%** | **67.71%** | **76.04%** |
| FedRot | 73.61% | 76.04% | 64.06% | 65.10% |
| FedEx | 86.81% | - | - | - |
| Freshness | 86.81% | - | - | - |
| VAST | 86.81% | - | - | - |
| MTiP adaptive | 87.15% | - | - | - |

## Class NLL

Cang thap cang tot.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| RIFT | 0.321890 | 0.458886 | 0.729945 | 0.639211 |
| Spectral filter | **0.321652** | **0.457380** | **0.724801** | 0.630478 |
| AlignFed calibration | 0.328915 | 0.459272 | 0.724837 | **0.626093** |
| FedRot | 0.559643 | 0.467754 | 0.848227 | 0.791774 |
| FedEx | 0.339258 | - | - | - |
| Freshness | 0.345164 | - | - | - |
| VAST | 0.345806 | - | - | - |
| MTiP adaptive | 0.352232 | - | - | - |

## Harmful Update Rate

Don vi: phan tram measured updates, cang thap cang tot.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| RIFT | **0.00%** | **16.67%** | **25.00%** | **25.00%** |
| Spectral filter | 4.17% | 33.33% | 29.17% | 29.17% |
| AlignFed calibration | 6.25% | **16.67%** | 37.50% | 37.50% |
| FedRot | 56.25% | 33.33% | 50.00% | 50.00% |
| FedEx | 39.58% | - | - | - |
| Freshness | 41.67% | - | - | - |
| VAST | 35.42% | - | - | - |
| MTiP adaptive | 41.67% | - | - | - |

## Protocol

- SST-2 trong bang la development seeds 2101-2103, 4 warmup + 16 measured
  returns va 96 eval examples. Day la protocol duy nhat co du ca 8 method.
- QNLI/MNLI la development seeds 3101-3103, 2 warmup + 8 measured returns va
  64 eval examples. Chi bon method manh da duoc chay.
- Cac task deu dung label-shard non-IID, heterogeneous rank va heterogeneous
  compute time. Day la bang tong hop descriptive, khong phai mot statistical
  cross-task leaderboard.
- QNLI/MNLI chi co mot event `staleness >= late_tau` moi run. Vi vay late
  harmful rate co the nhay 0%/100% theo mot event va chua du tin cay de lam
  thesis verdict.

## SST-2 held-out confirmation

Tren seeds 2201-2203 va eval offset 128, RIFT dat 90.28% accuracy, class NLL
0.213775, harmful 4.17% va late harmful 0%. Spectral cung dat 90.28% accuracy
nhung class NLL 0.214077 va harmful 8.33%. AlignFed dat 89.58% accuracy,
class NLL 0.219613 va harmful 10.42%. FedRot dat 88.89% accuracy, class NLL
0.273088 va harmful 52.08%.

## Ket luan

- RIFT co accuracy cao nhat hoac dong cao nhat tren ca bon task.
- RIFT co harmful rate thap nhat hoac dong thap nhat tren ca bon task. Day la
  tin hieu cross-task manh nhat hien tai.
- Spectral filter va AlignFed van co class NLL tot hon RIFT tren mot so task,
  dac biet MNLI. RIFT khong phai NLL winner tuyet doi.
- FedRot kem on dinh va co harmful rate cao. FedEx, Freshness, VAST va MTiP
  chua duoc chay tren QNLI/MNLI trong protocol moi, nen khong duoc suy dien ket
  qua tu SST-2 sang cac task do.
- Verdict dung muc la `preliminary GO` cho kha nang generalize accuracy va
  update safety cua implementation. Final thesis van `INCONCLUSIVE` cho den
  khi matrix 3B co 92 measured returns, 6 held-out seed, du late events, full
  client coverage va paired confidence interval.
