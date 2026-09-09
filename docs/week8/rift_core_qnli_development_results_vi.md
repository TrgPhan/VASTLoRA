# RIFT-Core QNLI development results

Ngay chay: 2026-09-09. Day la development evidence, khong phai held-out
verdict. Backbone la Qwen/Qwen2.5-1.5B-Instruct 4-bit. Tat ca methods dung
cung QNLI validation shuffle, offset 1024, 96 eval examples, 4 warmup + 16
measured returns, non-IID label shard, client ranks [2, 4, 8, 4] va compute
times [1, 2, 5, 10]. Seeds: 5101, 5102, 5103.

Artifacts: `outputs/rift_core_dev_1_5b/qnli/noniid_high_staleness`.

## Ket qua trung binh

| Method | Accuracy % | Class NLL | Harmful % | Late harmful % | Acceptance % | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| RIFT-Core | **80.903** | **0.434355** | 2.083 | **0.000** | 81.250 | 838.67 |
| RIFT-Diag | 79.861 | 0.443852 | **0.000** | **0.000** | 97.917 | 883.18 |
| Spectral Filter | 74.653 | 0.459969 | 6.250 | **0.000** | 100.000 | 313.54 |
| RIFT gate-only | 74.653 | 0.460055 | 6.250 | **0.000** | 100.000 | 354.15 |
| AlignFed calibration | 73.958 | 0.461148 | 10.417 | 8.333 | 68.750 | 535.46 |

## Accuracy theo seed

| Method | 5101 | 5102 | 5103 | Mean |
|---|---:|---:|---:|---:|
| RIFT-Core | **80.208** | **80.208** | **82.292** | **80.903** |
| RIFT-Diag | 79.167 | **80.208** | 80.208 | 79.861 |
| Spectral Filter | 75.000 | 73.958 | 75.000 | 74.653 |
| RIFT gate-only | 75.000 | 73.958 | 75.000 | 74.653 |
| AlignFed calibration | 75.000 | 71.875 | 75.000 | 73.958 |

RIFT-Core hon Spectral tren 3/3 seeds, delta accuracy trung binh +6.250 pp.
RIFT-Diag cung hon Spectral tren 3/3 seeds, delta trung binh +5.208 pp.
Core hon Diag tren 2/3 seeds va hoa 1/3, delta trung binh +1.042 pp.

## Dien giai

RIFT gate-only gan nhu trung Spectral Filter. Do do accuracy gain cua Core va
Diag khong duoc giai thich boi positive filtering/gate mot minh. Full core co
48 measured events; route gom 32 `rank_filtered`, 9 `reject` va 7
`spectral_comparator`. Mean off-diagonal norm la 0.3085 (min 0.0860, max
0.4174), nen bien cross-component da thuc su hoat dong thay vi co ve diagonal.

Tin hieu hien tai du de tiep tuc development, nhung chua du GO:

- Chi co mot task, mot regime, ba seeds va 96 eval examples.
- Moi seed chi co 4 late events, nen late-harm rate con rat nhieu bat dinh.
- Core cham hon Spectral khoang 2.67 lan; Diag cham hon khoang 2.82 lan.
- Artifacts co `git_worktree_dirty=true` vi tracked file tam
  `tmp_pyproject_view.txt` da bi xoa san trong workspace. `git diff` xac nhan
  day la thay doi duy nhat va file khong duoc code tham chieu. Ket qua hop le
  de development, nhung khong dung lam clean confirmation artifact.

## Quyet dinh tiep theo

1. Chay MNLI-m development tren clean worktree voi cung 3 seeds va 5 methods.
2. Neu Core/Diag giu duoc tin hieu, khoa mot candidate theo mean accuracy,
   class NLL va harmful, khong chon theo seed dep.
3. Tao confirmation rieng voi eval 512 tren range chua dung va clean commit.
   Khong thay eval size ben trong cohort hien tai.
4. Tang measured returns de co nhieu late events truoc khi claim safety.

