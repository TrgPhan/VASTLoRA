# Week 8 - Tong hop ket qua Qwen2.5-1.5B

Ngay tong hop: 2026-09-05

Tat ca cac dong dung cung backbone `Qwen/Qwen2.5-1.5B-Instruct`. Ten trong
bang la aggregation/filtering method, khong phai model backbone khac nhau.

Trong moi o, thu tu metric la `Accuracy / class NLL / Harmful / Late harmful`.
Accuracy cang cao cang tot; ba metric con lai cang thap cang tot.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---|---|---|---|
| RIFT | **87.50% / 0.321890 / 0.00% / 0.00%** | **78.65% / 0.458886 / 16.67% / 33.33%** | **67.71% / 0.729945 / 25.00% / 0.00%** | **76.04% / 0.639211 / 25.00% / 0.00%** |
| Spectral filter | **87.50%** / **0.321652** / 4.17% / 0.00% | **78.65%** / **0.457380** / 33.33% / 33.33% | 67.19% / **0.724801** / 29.17% / **0.00%** | **76.04%** / 0.630478 / 29.17% / **0.00%** |
| AlignFed calibration | **87.50%** / 0.328915 / 6.25% / 8.33% | **78.65%** / 0.459272 / **16.67%** / **0.00%** | **67.71%** / 0.724837 / 37.50% / 33.33% | **76.04%** / **0.626093** / 37.50% / 33.33% |
| FedRot | 73.61% / 0.559643 / 56.25% / 50.00% | 76.04% / 0.467754 / 33.33% / 33.33% | 64.06% / 0.848227 / 50.00% / 33.33% | 65.10% / 0.791774 / 50.00% / 33.33% |
| FedEx | 86.81% / 0.339258 / 39.58% / 16.67% | Chua test | Chua test | Chua test |
| Freshness | 86.81% / 0.345164 / 41.67% / 25.00% | Chua test | Chua test | Chua test |
| VAST | 86.81% / 0.345806 / 35.42% / 16.67% | Chua test | Chua test | Chua test |
| MTiP adaptive | 87.15% / 0.352232 / 41.67% / 58.33% | Chua test | Chua test | Chua test |

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
