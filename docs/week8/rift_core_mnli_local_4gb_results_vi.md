# RIFT-Core MNLI-m local 4GB development results

Ngay 2026-09-09, chay tren clean worktree rieng:
`C:\Users\Ad\Downloads\RIFTLoRA-mnli-clean`.

Day la development cohort, khong phai held-out verdict. Cau hinh nay giu logic
method hien tai, chi doi experiment setup de chay duoc tren GPU 4 GiB:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit.
- Task: MNLI matched validation (`mnli_m`), offset 1024, 96 eval examples.
- Regime: non-IID + high staleness (`label_shard`, ranks 2/4/8/4, compute 1/2/5/10).
- Returns: 4 warmup + 16 measured.
- Calibration: 24 gradient examples, 48 gate examples, 48 monitor examples.
- Max length: 128 cho tat ca methods. Ket qua max length 192 truoc do bi OOM o seed 5102 va khong duoc tron vao cohort nay.
- Seeds: 5101, 5102, 5103.
- Provenance: 15/15 result co `git_worktree_dirty=false`; dry-run bao `skip completed` cho tat ca entries.

## Accuracy

| Method | Seed 5101 | Seed 5102 | Seed 5103 | Mean |
| --- | ---: | ---: | ---: | ---: |
| RIFT-Core | 73.96% | 76.04% | 72.92% | **74.31%** |
| RIFT-Diag | 66.67% | 68.75% | 64.58% | 66.67% |
| Spectral Filter | 66.67% | 65.62% | 65.62% | 65.97% |
| RIFT gate-only | 66.67% | 65.62% | 64.58% | 65.62% |
| AlignFed calibration | 61.46% | 61.46% | 62.50% | 61.81% |

## Class NLL

Thap hon la tot hon.

| Method | Seed 5101 | Seed 5102 | Seed 5103 | Mean |
| --- | ---: | ---: | ---: | ---: |
| RIFT-Core | 0.691914 | 0.603847 | 0.627339 | **0.641033** |
| RIFT-Diag | 0.755463 | 0.714368 | 0.744068 | 0.737966 |
| Spectral Filter | 0.775721 | 0.762058 | 0.770488 | 0.769422 |
| RIFT gate-only | 0.773145 | 0.761708 | 0.773433 | 0.769428 |
| AlignFed calibration | 0.783779 | 0.797490 | 0.771448 | 0.784239 |

## Harmful

Thap hon la tot hon. Harmful la ti le measured update lam monitor loss tang.

| Method | Seed 5101 | Seed 5102 | Seed 5103 | Mean |
| --- | ---: | ---: | ---: | ---: |
| RIFT-Core | 0.00% | 6.25% | 0.00% | **2.08%** |
| RIFT-Diag | 6.25% | 0.00% | 0.00% | **2.08%** |
| AlignFed calibration | 0.00% | 12.50% | 18.75% | 10.42% |
| RIFT gate-only | 37.50% | 12.50% | 6.25% | 18.75% |
| Spectral Filter | 43.75% | 18.75% | 0.00% | 20.83% |

## Late harmful

Thap hon la tot hon. Late harmful chi tinh tren update co staleness >= `late_tau`.

| Method | Seed 5101 | Seed 5102 | Seed 5103 | Mean |
| --- | ---: | ---: | ---: | ---: |
| RIFT-Core | 0.00% | 0.00% | 0.00% | **0.00%** |
| RIFT-Diag | 25.00% | 0.00% | 0.00% | 8.33% |
| AlignFed calibration | 0.00% | 25.00% | 25.00% | 16.67% |
| RIFT gate-only | 25.00% | 0.00% | 25.00% | 16.67% |
| Spectral Filter | 50.00% | 25.00% | 0.00% | 25.00% |

## Runtime

| Method | Mean runtime |
| --- | ---: |
| Spectral Filter | 357.86 s |
| RIFT gate-only | 576.23 s |
| AlignFed calibration | 851.47 s |
| RIFT-Diag | 1154.39 s |
| RIFT-Core | 1319.95 s |

## Ket luan tam thoi

RIFT-Core thang ro tren MNLI-m non-IID + high-staleness development cohort:

- Acc thang 3/3 seeds so voi Spectral Filter, gap mean +8.33 pp.
- Class NLL tot hon Spectral Filter tren 3/3 seeds, gap mean -0.128389.
- Harmful mean giam tu 20.83% xuong 2.08%.
- Late harmful mean giam tu 25.00% xuong 0.00%.

RIFT gate-only gan nhu bang Spectral Filter ve Acc/NLL va van harmful cao. Vi vay,
tin hieu MNLI nay ung ho gia thuyet rang phan Core repair moi la nguon loi ich
chinh, khong chi la spectral gating/filtering.

Van chua nen goi la GO cuoi cung. Buoc tiep theo can chay confirmation voi eval
lon hon, it nhat 512 examples, va giu winner/config da freeze. Confirmation khong
duoc chon seed dep sau khi nhin ket qua.
