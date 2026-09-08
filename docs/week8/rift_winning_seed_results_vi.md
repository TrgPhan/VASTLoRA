# Week 8 - Cac held-out seed RIFT thang Spectral Filter

Ngay tong hop: 2026-09-08

## Pham vi va tieu chi

- Nguon: positive-filter confirmation, held-out offset `768`.
- Backbone: `Qwen/Qwen2.5-1.5B-Instruct`.
- Regime: non-IID label shard, heterogeneous rank, high staleness.
- Seeds da chay: `4301-4306`; khong seed nao bi loai.
- Tieu chi xep hang: accuracy cao hon; neu accuracy bang nhau thi class NLL
  thap hon se thang.
- Spectral Filter la matched Spectral-Surgery-style control trong simulator,
  khong phai official Spectral Surgery implementation.

## Bang cac seed RIFT thang

Moi cap trong bang co accuracy bang nhau; RIFT thang bang class NLL thap hon.
`Cum` la normalized cumulative late harm. Tat ca metric safety va NLL deu lower
is better.

| Task | Seed | Acc RIFT / Spectral (%) | NLL RIFT / Spectral | Harmful RIFT / Spectral (%) | Late harmful RIFT / Spectral (%) | Cum RIFT / Spectral |
|---|---:|---:|---:|---:|---:|---:|
| MNLI-m | 4302 | 68.750 / 68.750 | **0.680238** / 0.683891 | **43.75** / 50.00 | **0.00** / 25.00 | **0.000000** / 0.002642 |
| MNLI-m | 4303 | 73.958 / 73.958 | **0.653143** / 0.653211 | 12.50 / **6.25** | 0.00 / 0.00 | 0.000000 / 0.000000 |
| MNLI-m | 4304 | 73.958 / 73.958 | **0.742627** / 0.749242 | **75.00** / 81.25 | **0.00** / 25.00 | **0.000000** / 0.002527 |
| MNLI-mm | 4302 | 66.667 / 66.667 | **0.714781** / 0.718150 | **43.75** / 50.00 | **0.00** / 25.00 | **0.000000** / 0.002642 |
| MNLI-mm | 4303 | 70.833 / 70.833 | **0.692967** / 0.693155 | 12.50 / **6.25** | 0.00 / 0.00 | 0.000000 / 0.000000 |
| MNLI-mm | 4304 | 71.875 / 71.875 | **0.765625** / 0.771265 | **75.00** / 81.25 | **0.00** / 25.00 | **0.000000** / 0.002527 |
| QNLI | 4305 | 81.250 / 81.250 | **0.471976** / 0.472032 | 6.25 / 6.25 | 0.00 / 0.00 | 0.000000 / 0.000000 |

## Tong ket

| Task | RIFT win / 6 seeds | Winning seeds |
|---|---:|---|
| QNLI | 1/6 | 4305 |
| MNLI-m | 3/6 | 4302, 4303, 4304 |
| MNLI-mm | 3/6 | 4302, 4303, 4304 |
| **Tong** | **7/18 task-seed pairs** | - |

RIFT khong co strict accuracy win trong 18 task-seed pairs nay. Bay ket qua
thang deu la accuracy tie va NLL tie-break. Trong so do, `4302` va `4304` tren
ca hai MNLI slices la bang chung mo ta manh hon vi RIFT dong thoi co NLL,
harmful, late harmful va cumulative late harm tot hon Spectral Filter.

Seed `4303` tren hai MNLI slices chi thang ve NLL; harmful rate cua RIFT cao
hon Spectral Filter. QNLI `4305` chi co NLL advantage rat nho (`0.000055`).

Bang nay chi la seed-level descriptive analysis. Week 8 strong verdict van la
`NO_GO`, vi mean paired confidence intervals va hard-slice gates khong pass
dong thoi tren tat ca tasks/opponents. Khong duoc chi bao cao bay seed nay va
bo qua cac seed con lai khi viet ket luan thesis.
