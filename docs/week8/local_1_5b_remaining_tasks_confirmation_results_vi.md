# Week 8 - Held-out confirmation QNLI va MNLI

Ngay hoan tat: 2026-09-06

## Protocol va provenance

- Backbone: `Qwen/Qwen2.5-1.5B-Instruct` 4-bit.
- Tasks: QNLI, MNLI matched (`MNLI-m`) va mismatched (`MNLI-mm`).
- Methods: RIFT, Spectral filter, AlignFed calibration control va FedRot.
- Moi cell co 6 seed `3201-3206`, 96 held-out examples tai offset 64.
- Moi run co 4 warmup + 16 measured returns, 4 late events va full client
  coverage.
- Tat ca 72/72 run den tu commit sach `1e8bfb9`, matrix SHA
  `f60637aad609f11f42d438a9c3959d520b8c1d0ebe3e4d90b5bbd279428ce356`.
- Nguon phan tich:
  `outputs/local_1_5b_remaining_tasks_confirmation_v3_analysis/`.

Khong seed nao bi loai theo performance. Vong v1/v2 bi OOM hoac tron commit
khong duoc dua vao bat ky con so nao ben duoi.

## Accuracy

Don vi phan tram. Trong ngoac la median cua 6 seed; cang cao cang tot.

| Method | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|
| RIFT | 78.993 (77.604) | 80.556 (80.729) | 75.868 (75.521) |
| Spectral filter | 78.993 (77.604) | **80.903 (81.250)** | **76.910 (76.562)** |
| AlignFed calibration | **79.167 (78.125)** | 80.035 (80.729) | 76.215 (76.562) |
| FedRot | 61.458 (57.812) | 64.757 (68.229) | 63.368 (68.229) |

## Class NLL

Trong ngoac la median cua 6 seed; cang thap cang tot.

| Method | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|
| RIFT | 0.441761 (0.445003) | 0.559449 (0.551205) | 0.587634 (0.581319) |
| Spectral filter | 0.440928 (0.443897) | **0.545385 (0.544993)** | **0.573481 (0.569425)** |
| AlignFed calibration | **0.440826 (0.439731)** | 0.554471 (0.546327) | 0.582853 (0.575501) |
| FedRot | 1.114409 (0.848339) | 1.656944 (1.798480) | 1.685658 (1.823565) |

## Harmful updates

Don vi phan tram measured updates; cang thap cang tot. Moi o la
`harmful / late harmful`.

| Method | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|
| RIFT | **32.292 / 12.500** | **13.542 / 8.333** | **13.542 / 8.333** |
| Spectral filter | 34.375 / 29.167 | 28.125 / 33.333 | 28.125 / 33.333 |
| AlignFed calibration | 36.458 / 37.500 | **13.542 / 4.167** | **13.542 / 4.167** |
| FedRot | 62.500 / 33.333 | 70.833 / 41.667 | 70.833 / 41.667 |

RIFT co harmful rate tot nhat hoac dong tot nhat tren ca ba task, va late
harm tot nhat tren QNLI. Tren MNLI, AlignFed calibration co late harmful thap
hon RIFT, nen RIFT khong phai safety winner tuyet doi.

## Best observed

Day chi la mo ta, khong duoc dung cho verdict va khong thay the trung binh 6
seed.

| Task | Method | Best accuracy (seed) | Best NLL (seed) |
|---|---|---:|---:|
| QNLI | RIFT | 87.500 (3205) | 0.401958 (3205) |
| QNLI | Spectral filter | 86.458 (3205) | 0.407319 (3205) |
| QNLI | AlignFed calibration | **89.583 (3205)** | **0.386505 (3205)** |
| QNLI | FedRot | 78.125 (3203) | 0.456943 (3203) |
| MNLI-m | RIFT | 81.250 (3202) | 0.542147 (3202) |
| MNLI-m | Spectral filter | 82.292 (3204) | **0.512985 (3206)** |
| MNLI-m | AlignFed calibration | **83.333 (3202)** | 0.521848 (3202) |
| MNLI-m | FedRot | 73.958 (3205) | 0.631001 (3205) |
| MNLI-mm | RIFT | 78.125 (3204) | 0.566135 (3202) |
| MNLI-mm | Spectral filter | **79.167 (3204)** | **0.539734 (3204)** |
| MNLI-mm | AlignFed calibration | 78.125 (3201) | 0.552657 (3202) |
| MNLI-mm | FedRot | 76.042 (3202) | 0.689248 (3202) |

## Paired result

`W/T/L` la so seed RIFT thang/hoa/thua cung seed cua doi thu.

| Task | Opponent | Accuracy W/T/L | NLL W/T/L | Harmful W/T/L | Late W/T/L |
|---|---|---:|---:|---:|---:|
| QNLI | Spectral filter | 1/4/1 | 2/0/4 | 3/0/3 | 3/3/0 |
| QNLI | AlignFed calibration | 2/1/3 | 3/0/3 | 3/1/2 | 5/1/0 |
| QNLI | FedRot | 4/1/1 | 6/0/0 | 6/0/0 | 4/1/1 |
| MNLI-m | Spectral filter | 1/2/3 | 1/0/5 | 4/0/2 | 4/2/0 |
| MNLI-m | AlignFed calibration | 3/1/2 | 2/0/4 | 3/1/2 | 1/3/2 |
| MNLI-m | FedRot | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| MNLI-mm | Spectral filter | 0/2/4 | 1/0/5 | 4/0/2 | 4/2/0 |
| MNLI-mm | AlignFed calibration | 2/1/3 | 2/0/4 | 3/1/2 | 1/3/2 |
| MNLI-mm | FedRot | 5/0/1 | 6/0/0 | 6/0/0 | 6/0/0 |

Ket qua ro nhat chong RIFT la MNLI-mm accuracy so voi Spectral filter:
`-1.042 pp`, CI95 `[-2.019, -0.064]`. Khoang tin cay nam hoan toan duoi
0, va cung vuot margin non-inferiority `-0.5 pp`.

## Verdict

Analyzer chinh thuc tra **`NO_GO`** cho claim hien tai: "RIFT non-inferior
accuracy/NLL va giam late harm so voi moi doi thu tren moi hard slice".

- RIFT pass day du gate truoc FedRot tren MNLI-m, va co loi the rat lon truoc
  FedRot nhin chung.
- RIFT khong pass accuracy/NLL non-inferiority truoc Spectral filter va
  AlignFed calibration tren cac hard slice.
- RIFT khong giam late harm so voi AlignFed calibration tren MNLI.
- QNLI cho thay trade-off safety huu ich, nhung khong du de chung minh RIFT la
  best overall.

Day la `NO_GO` cho gia thuyet manh va config hien tai, khong phai bang chung
rang moi y tuong rank-wise filtering deu vo dung. Claim co the bao ve hon la
RIFT giam harmful updates so voi ung dung spectral khong gate trong khi giu
accuracy gan tuong duong tren QNLI/MNLI-m; claim nay van can protocol moi duoc
pre-register, them baseline day du va xac nhan 3B.
