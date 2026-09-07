# Week 8 - RIFT positive-filter development summary

Ngay cap nhat: 2026-09-07

## Ket luan ngan

- Verdict chinh thuc gan nhat van la **`NO_GO`** cho RIFT gain-mass v1 tren
  clean held-out offset 256, 6 seeds va 72/72 runs.
- RIFT positive-filter + extreme-staleness rescue dat **development GO to
  confirmation** truoc Spectral filter tren ca QNLI, MNLI-m va MNLI-mm.
- Day chua phai thesis `GO`: candidate moi chi co 3 development seeds, chua co
  paired CI95 va chua dung held-out moi.
- Khong seed nao bi loai theo performance.

## Official clean held-out cua RIFT gain-mass v1

Backbone `Qwen/Qwen2.5-1.5B-Instruct`, offset 256, seeds 3301-3306, non-IID
label shard, heterogeneous rank va high staleness. Cumulative la mean tong
late-harm moi run.

| Task | Method | Acc % | Class NLL | Harmful % | Late harmful % | Cumulative |
|---|---|---:|---:|---:|---:|---:|
| QNLI | RIFT v1 | 78.819 | 0.420961 | 28.12 | 12.50 | 0.000547 |
| QNLI | Spectral filter | 78.819 | 0.420623 | 30.21 | 12.50 | 0.000336 |
| QNLI | AlignFed calibration | 78.125 | 0.426418 | 17.71 | 25.00 | 0.001893 |
| QNLI | FedRot | 58.854 | 0.994735 | 57.29 | 50.00 | 0.157888 |
| MNLI-m | RIFT v1 | 74.653 | 0.641314 | 35.42 | 25.00 | 0.006117 |
| MNLI-m | Spectral filter | 74.479 | 0.645470 | 38.54 | 33.33 | 0.006550 |
| MNLI-m | AlignFed calibration | 73.958 | 0.642557 | 26.04 | 25.00 | 0.011462 |
| MNLI-m | FedRot | 60.590 | 1.768007 | 63.54 | 54.17 | 0.474664 |
| MNLI-mm | RIFT v1 | 75.694 | 0.626717 | 35.42 | 25.00 | 0.006117 |
| MNLI-mm | Spectral filter | 76.042 | 0.630020 | 38.54 | 33.33 | 0.006550 |
| MNLI-mm | AlignFed calibration | 75.174 | 0.629837 | 26.04 | 25.00 | 0.011462 |
| MNLI-mm | FedRot | 58.160 | 1.916066 | 63.54 | 54.17 | 0.474664 |

Analyzer tra `NO_GO` vi paired CI95 khong pass dong thoi accuracy/NLL
non-inferiority va hai safety gates truoc moi doi thu. RIFT v1 chi pass day du
truoc FedRot tren hai MNLI slice.

## SST-2 held-out truoc do

Offset 128, seeds 2201-2203. Bang nay khong duoc gop seed voi confirmation
QNLI/MNLI.

| Method | Acc % | Class NLL | Harmful % | Late harmful % |
|---|---:|---:|---:|---:|
| RIFT | 90.28 | 0.213775 | 4.17 | 0.00 |
| Spectral filter | 90.28 | 0.214077 | 8.33 | 0.00 |
| AlignFed calibration | 89.58 | 0.219613 | 10.42 | 8.33 |
| FedRot | 88.89 | 0.273088 | 52.08 | 25.00 |

## Final development candidate

Candidate tai
`configs/local_1_5b_rift_positive_filter_candidate_matrix.json` giu moi
singular component co predicted gain duong, dung paired gate tu staleness 4,
va chi rescue rejected high-gain update tu staleness 8 voi scale bi chan boi
`min(0.75, 6 / tau)`.

Ket qua offset 512, seeds 4101-4103:

| Task | Method | Acc % | Class NLL | Harmful % | Late harmful % | Cumulative | Accept % |
|---|---|---:|---:|---:|---:|---:|---:|
| QNLI | RIFT candidate | 82.639 | 0.419593 | 12.50 | 16.67 | 0.000814 | 100.00 |
| QNLI | Spectral filter | 82.986 | 0.417924 | 18.75 | 25.00 | 0.002094 | 100.00 |
| MNLI-m | RIFT candidate | 73.611 | 0.605197 | 27.08 | 33.33 | 0.003308 | 97.92 |
| MNLI-m | Spectral filter | 73.958 | 0.601625 | 27.08 | 41.67 | 0.007816 | 100.00 |
| MNLI-mm | RIFT candidate | 67.708 | 0.691651 | 27.08 | 33.33 | 0.003308 | 97.92 |
| MNLI-mm | Spectral filter | 67.014 | 0.687518 | 27.08 | 41.67 | 0.007816 | 100.00 |

Paired mean quy uoc: accuracy duong la RIFT cao hon; NLL va safety duong la
RIFT giam duoc so voi Spectral.

| Task | Acc delta pp | NLL improvement | Harmful reduction pp | Late reduction pp | Cumulative reduction |
|---|---:|---:|---:|---:|---:|
| QNLI | -0.347 | -0.001669 | +6.25 | +8.33 | +0.001281 |
| MNLI-m | -0.347 | -0.003572 | 0.00 | +8.33 | +0.004508 |
| MNLI-mm | +0.694 | -0.004133 | 0.00 | +8.33 | +0.004508 |

Ca ba task deu pass **point-estimate** margins da khoa:

- accuracy delta >= -0.5 pp;
- NLL improvement >= -0.005;
- late-harm reduction > 0;
- cumulative late-harm reduction > 0.

## Development comparison voi AlignFed calibration

AlignFed companion da hoan tat 9/9 runs voi cung offset, seeds, schedule,
calibration sizes va local-training config. Khac biet quan trong la AlignFed
reject toan bo update khi gate khong an toan, con RIFT co gang giu utility bang
component filtering va bounded rescue.

| Task | Method | Acc % | Class NLL | Harmful % | Late harmful % | Normalized cumulative | Accept % |
|---|---|---:|---:|---:|---:|---:|---:|
| QNLI | RIFT candidate | 82.639 | 0.419593 | 12.50 | 16.67 | 0.000203 | 100.00 |
| QNLI | AlignFed calibration | 81.250 | 0.423322 | 10.42 | 0.00 | 0.000000 | 70.83 |
| MNLI-m | RIFT candidate | 73.611 | 0.605197 | 27.08 | 33.33 | 0.000827 | 97.92 |
| MNLI-m | AlignFed calibration | 71.528 | 0.630075 | 8.33 | 8.33 | 0.000059 | 62.50 |
| MNLI-mm | RIFT candidate | 67.708 | 0.691651 | 27.08 | 33.33 | 0.000827 | 97.92 |
| MNLI-mm | AlignFed calibration | 64.931 | 0.719745 | 8.33 | 8.33 | 0.000059 | 62.50 |

RIFT co mean accuracy va class NLL tot hon AlignFed tren ca ba task, dong thoi
acceptance cao hon `29.17 pp` tren QNLI va `35.42 pp` tren hai MNLI slice.
Nguoc lai, AlignFed co raw harmful/late-harm thap hon vi reject nhieu update.
Do do RIFT **khong dominate AlignFed tren moi metric**; hai method la hai diem
khac nhau tren utility-safety-utilization frontier.

## Best observed cua candidate

Chi mang tinh mo ta, khong dung cho verdict.

| Task | Best accuracy | Seed | Best class NLL | Seed |
|---|---:|---:|---:|---:|
| QNLI | 86.458% | 4103 | 0.407667 | 4103 |
| MNLI-m | 73.958% | 4101/4102 | 0.601689 | 4103 |
| MNLI-mm | 68.750% | 4103 | 0.685878 | 4103 |

## Cac vong tuning da chay

- Delay-budget rescue v2, budget 2: giam magnitude harm tren MNLI nhung con
  thua Spectral ve MNLI-m accuracy/NLL va QNLI cumulative harm.
- Gate-first: an toan hon nhung mat utility tren MNLI.
- Fixed rescue 0.25/0.50: cho thay scale can phu thuoc staleness; mot scale
  khong phu hop moi task.
- Extreme rescue grid mass 0.90/0.95/1.00: mass 0.95 bi dominated; mass 0.90
  tot ve NLL, mass 1.00 lay lai accuracy seed 4103.
- Extreme-only pruning: khong cai thien accuracy va lam NLL seed 4102 te hon.
- Positive-filter candidate: frontier development tot nhat hien tai.

## Trang thai nghien cuu

Trang thai hop le la **PROMISING / GO to a new confirmation**, khong phai final
`GO`. Protocol moi da duoc khoa tai
`docs/week8/rift_positive_filter_confirmation_protocol_vi.md`: offset 768,
seeds 4301-4306, RIFT candidate, Spectral filter, AlignFed calibration va
FedRot. Analyzer dung relative-safety gate voi throughput-matched controls va
constrained-utilization gate voi reject-heavy AlignFed. Neu bat ky hard slice
khong pass, verdict van la `NO_GO` hoac `INCONCLUSIVE` dung theo analyzer.
