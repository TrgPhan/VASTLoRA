# Week 8 - RIFT positive-filter held-out confirmation

Ngay chay: 2026-09-08

## Protocol va integrity

- Backbone: `Qwen/Qwen2.5-1.5B-Instruct`.
- Tasks: QNLI, MNLI matched va MNLI mismatched.
- Regime: non-IID label shard, heterogeneous ranks `[2, 4, 8, 4]`, high staleness.
- Held-out offset: `768`; seeds: `4301-4306`; khong loai seed.
- Methods: RIFT, Spectral-Surgery-style matched filter, AlignFed calibration
  control va matched FedRot operator.
- Hoan tat `72/72` runs, schema `4`, clean commit
  `25dd949f8c3ac6e2afc04a38aace6249f4ecb514`.
- Spectral Filter va AlignFed trong bang la matched controls, khong phai full
  official paper implementations.

## Accuracy (%)

Higher is better. Moi o la mean cua 6 seeds.

| Method | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|
| RIFT | 81.250 | 71.528 | 70.660 |
| Spectral Filter | **81.424** | 71.528 | 71.007 |
| AlignFed calibration | 80.556 | **72.569** | **71.528** |
| FedRot | 67.882 | 55.382 | 55.556 |

## Class NLL

Lower is better. Moi o la mean cua 6 seeds.

| Method | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|
| RIFT | 0.477342 | 0.673730 | 0.710504 |
| Spectral Filter | **0.477093** | 0.674585 | 0.710742 |
| AlignFed calibration | 0.481435 | **0.667121** | **0.708016** |
| FedRot | 0.662944 | 2.244059 | 2.292207 |

## Harmful updates

Lower is better. Moi o la `harmful % / late harmful % / acceptance %`.

| Method | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|
| RIFT | 16.67 / 8.33 / 98.96 | 30.21 / **4.17** / 96.88 | 30.21 / **4.17** / 96.88 |
| Spectral Filter | 16.67 / 8.33 / 100.00 | 31.25 / 12.50 / 100.00 | 31.25 / 12.50 / 100.00 |
| AlignFed calibration | **13.54 / 4.17** / 71.88 | **19.79** / 8.33 / 70.83 | **19.79** / 8.33 / 70.83 |
| FedRot | 52.08 / 20.83 / 100.00 | 69.79 / 50.00 / 100.00 | 69.79 / 50.00 / 100.00 |

## RIFT versus Spectral Filter theo seed

Neu dung accuracy lam primary metric va class NLL lam tie-break khi accuracy
bang nhau, RIFT thang `7/18` task-seed pairs:

- QNLI: seed `4305` (`1/6`).
- MNLI-m: seeds `4302`, `4303`, `4304` (`3/6`); accuracy deu hoa, RIFT thang
  bang class NLL.
- MNLI-mm: seeds `4302`, `4303`, `4304` (`3/6`); accuracy deu hoa, RIFT thang
  bang class NLL.

Day la thong ke mo ta, khong thay the paired mean va confidence interval. RIFT
khong co strict accuracy win truoc Spectral Filter: MNLI-m hoa accuracy o ca 6
seed; MNLI-mm va QNLI co mot so seed thua accuracy.

## Verdict

Analyzer tra **`NO_GO`** cho strong Week 8 claim. RIFT pass day du truoc FedRot
tren hai MNLI slices, nhung khong pass dong thoi accuracy/NLL non-inferiority va
safety gates truoc Spectral Filter va AlignFed tren moi hard slice.

RIFT co tin hieu co ich tren MNLI so voi Spectral Filter:

- MNLI-m: accuracy hoa, mean class NLL giam `0.000855`, late harmful giam
  `8.33 pp`.
- MNLI-mm: accuracy thap hon `0.347 pp`, mean class NLL giam `0.000238`, late
  harmful giam `8.33 pp`.

Nhung CI95 cua late-harm va cumulative-harm reduction van cat qua 0. Vi vay
khong duoc claim RIFT dominate Spectral Filter.

Khong tune tiep tren offset `768` hoac seeds `4301-4306`. Neu sua method, phai
dung development offset moi va sau do dong bang mot confirmation offset/seeds
moi. Mot seed thang khong phai tieu chuan GO.
