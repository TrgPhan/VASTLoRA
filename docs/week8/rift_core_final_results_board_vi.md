# RIFT-Core Final Results Board

Ngay cap nhat: 2026-09-26

> Canh bao fidelity (2026-09-26): cac ket qua `fedavg_lora` va `ffa_lora`
> tu batch extended cu la legacy implementation, KHONG duoc dung de khang
> dinh RIFT thang hai baseline paper. Ban sua `persistent_factors_v2` can
> rerun rieng hai method. Da co 48/48 corrected runs tu notebook7b4f16575b:
> [ket qua da audit](week8_factor_kaggle_recovery_results_vi.md). Giu nguyen so lieu
> cu de audit. Chi tiet: [FedAvg/FFA fidelity](fedavg_ffa_fidelity_vi.md).

Day la file tong hop legit hien tai cho cohort confirmation moi nhat. Bang
RIFT-Core chinh chi dung cac run nam trong:

`outputs/rift_core_confirmation_v3_1_5b`

Cac experiment Week 9 bo sung, nhu RAVAN async, duoc dat thanh section rieng va
khong tron truc tiep vao mean/std cua cohort RIFT-Core.

## Scope Hien Tai

Da co ket qua day du:

| Task | Regime | Runs | Seeds | Eval examples |
|---|---|---:|---|---:|
| SST-2 | non-IID + high staleness | 48/48 | 6101-6106 | 1024 |
| QNLI | non-IID + high staleness | 48/48 | 6101-6106 | 1024 |
| MNLI-m | non-IID + high staleness | 48/48 | 6101-6106 | 1024 |
| MNLI-mm | non-IID + high staleness | 48/48 | 6101-6106 | 1024 |

Kaggle output da kiem tra:

| Kernel | Status luc kiem tra | Output local | Ket qua |
|---|---|---|---|
| `trngphanquang/notebookea4aef1e67` | `COMPLETE` | `outputs/kaggle_downloads/notebookea4aef1e67` | Da merge MNLI-m 48/48 |
| `trngphanquang/notebookc6e6ff3b22` | downloaded | `outputs/kaggle_downloads/notebookc6e6ff3b22` | Da merge MNLI-mm 48/48 |
| `trngphanquang/notebook08d0fd202f` | `COMPLETE`, downloaded latest | `outputs/kaggle_downloads/notebook08d0fd202f_latest` | Spectral FedLoRA: 72/72 hop le |

## Protocol

- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Methods: `raw`, `freshness`, `fedrot`, `spectral_surgery`,
  `alignfed_calibration`, `rift`, `rift_diag`, `rift_core`.
- Seeds: `6101, 6102, 6103, 6104, 6105, 6106`.
- Regime: label-shard non-IID, heterogeneous rank, high staleness.
- Evaluation: held-out evaluation with 1024 examples per task.
- Result source: `result.json` files under each task/method/seed directory.
- No cherry-picking trong cohort chinh: mean/std use all six seeds for every
  method.

Luu y integrity:

- Tat ca run cung commit `5d34c8414b295d8a013658d8fd94355063033db9`.
- Mot phan SST-2 va QNLI local cu co `git_worktree_dirty=true`.
- Cac QNLI job tai tu Kaggle cho `rift_diag`, `rift_core`, va phan con lai cua
  `rift` co provenance sach hon, nhung board nay van ghi chu dirty issue vi
  cohort tong the la pha local/Kaggle.
- Moi run co `late_event_count=7`; neu strict thesis gate yeu cau toi thieu 8
  late events thi safety gate coverage van can duoc trinh bay rieng.

## FedLoRA Extended Cohort

Batch mo rong dung matrix artifact rieng nhung giu cung task, seed, held-out
split va protocol non-IID + high staleness voi cohort chinh:

Luu y: section nay la snapshot legacy da tai truoc do tai
`outputs/kaggle_notebook08d0fd202f`. Output hien tai cua cung kernel slug da
duoc thay bang spectral suite 72/72 va duoc bao cao o section tiep theo.

- Extended matrix SHA256:
  `7dc2939ad759fd621be3b5cdcccc8f398569af58184904787123178b4d5db7ad`.
- Main cohort matrix SHA256:
  `973723fc52d7e0923e0f6768004e977cfd364662062bc30440fab637daaf5a9b`.
- Commit: `8b347cab1a553708c3616f429a9da4bedd2c54a6`; tat ca 84 result co
  `git_worktree_dirty=false`.
- `fedavg_lora`, `flora_lora`, `ffa_lora`: du 4 task x 6 seed, tong 72/72.
- `fedex_lora`: chi co 12/24 result, moi task 3/6 seed. Cac job con lai bi
  `SIGKILL 9` do RAM host tang manh khi luu dense residual snapshot; day khong
  phai ket qua metric that bai.
- `florg`: 0/24 result do adapter tao tren CPU trong khi model o CUDA, gay loi
  device mismatch trong forward hook. Khong co metric hop le de bao cao.

Ky hieu trong cac bang duoi:

- `[E]`: extended baseline da du 6/6 seed, co the so sanh trong simulator nay.
- `[E-legacy]`: du 6/6 seed nhung factor implementation da bi thay the; chi
  giu de audit, khong dung xep hang hay claim paper baseline.
- `[E-dagger]`: so lieu tam thoi chi tu 3/6 seed; khong dung de xep hang hay
  dua ra final claim.
- `N/A`: khong co run hop le.

- `[S]`: spectral FedLoRA suite moi, du 6/6 seed, commit sach va da audit lai
  tu output moi nhat cua `notebook08d0fd202f`.

Day la so sanh protocol-matched cross-batch, khong phai cung executable commit.
Neu can claim paper-level chat nhat, phai rerun RIFT va cac extended baseline
tren cung commit/matrix sau khi sua FLoRG va gioi han bo nho FedEx.

## Spectral FedLoRA Confirmation Cohort

Output moi nhat cua `trngphanquang/notebook08d0fd202f` co `72/72` result:
FlexLoRA, FLoRIST va FLoRG x 4 task x 6 seed. Tat ca cung commit sach
`2188a5505c4acc668c29fa743d86d90a6829f7c4`, cung held-out 1024 examples va
cung async schedule theo tung task/seed. Accuracy/NLL va safety metrics da
duoc tinh lai tu CSV. Report chi tiet:
[Week 8 Spectral FedLoRA](week8_spectral_fedlora_kaggle_results_vi.md).

Day la so sanh protocol-matched cross-batch. Ba paper operator duoc thich nghi
vao immediate async va heterogeneous prefix ranks; khong goi day la full
synchronous paper reproduction.

## Accuracy

Don vi: mean +/- std percentage point. Cao hon la tot hon.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| raw | 90.120 +/- 1.843 | 74.658 +/- 9.710 | 68.392 +/- 4.167 | 68.083 +/- 3.679 |
| freshness | 87.679 +/- 3.131 | 74.593 +/- 6.616 | 69.010 +/- 5.771 | 68.620 +/- 5.307 |
| fedrot | 82.389 +/- 2.491 | 73.014 +/- 4.079 | 71.354 +/- 3.021 | 71.354 +/- 2.632 |
| fedavg_lora [E-legacy] | 83.089 +/- 1.763 | 73.128 +/- 3.779 | 71.126 +/- 2.178 | 71.370 +/- 1.930 |
| flora_lora [E] | 90.137 +/- 1.644 | 74.495 +/- 9.081 | 68.343 +/- 3.847 | 67.952 +/- 3.463 |
| ffa_lora [E-legacy] | 78.385 +/- 1.114 | 71.517 +/- 2.168 | 73.568 +/- 0.749 | 72.689 +/- 1.218 |
| fedex_lora [E-dagger] | 84.863 +/- 0.832 | 79.948 +/- 0.332 | 65.267 +/- 3.579 | 67.936 +/- 2.641 |
| flexlora [S] | 83.757 +/- 2.793 | 73.665 +/- 4.328 | 70.638 +/- 3.521 | 70.671 +/- 3.140 |
| florist [S] | 82.829 +/- 2.618 | 73.226 +/- 4.215 | 71.191 +/- 3.282 | 71.273 +/- 3.174 |
| florg [S] | 77.230 +/- 0.911 | 72.070 +/- 1.369 | 73.486 +/- 0.819 | 72.493 +/- 1.372 |
| spectral_surgery | 90.316 +/- 1.720 | 74.023 +/- 10.501 | 68.066 +/- 4.312 | 67.660 +/- 3.590 |
| alignfed_calibration | 90.088 +/- 1.516 | 79.492 +/- 2.225 | 76.937 +/- 3.037 | 77.425 +/- 2.136 |
| rift | 90.609 +/- 1.450 | 79.329 +/- 1.517 | 78.483 +/- 1.839 | 79.020 +/- 1.128 |
| rift_diag | 92.285 +/- 1.024 | 80.680 +/- 0.721 | 80.762 +/- 1.406 | 81.038 +/- 0.944 |
| rift_core | **92.285 +/- 0.865** | **80.843 +/- 1.065** | **82.210 +/- 1.197** | **81.901 +/- 0.717** |

## Class NLL

Thap hon la tot hon.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| raw | 0.263043 +/- 0.034779 | 0.601147 +/- 0.265166 | 0.780 +/- 0.115 | 0.776886 +/- 0.108301 |
| freshness | 0.312909 +/- 0.040382 | 0.541523 +/- 0.148201 | 0.735 +/- 0.120 | 0.732654 +/- 0.110383 |
| fedrot | 0.384843 +/- 0.037355 | 0.529501 +/- 0.055123 | 0.667 +/- 0.047 | 0.676905 +/- 0.070920 |
| fedavg_lora [E-legacy] | 0.376959 +/- 0.029326 | 0.526808 +/- 0.052354 | 0.667283 +/- 0.030010 | 0.677603 +/- 0.055037 |
| flora_lora [E] | 0.261784 +/- 0.031203 | 0.606935 +/- 0.249505 | 0.784698 +/- 0.108652 | 0.781097 +/- 0.101157 |
| ffa_lora [E-legacy] | 0.452468 +/- 0.020310 | 0.541976 +/- 0.022165 | 0.650189 +/- 0.021011 | 0.668026 +/- 0.039411 |
| fedex_lora [E-dagger] | 0.350062 +/- 0.016356 | 0.431238 +/- 0.003293 | 0.812470 +/- 0.072410 | 0.749433 +/- 0.028928 |
| flexlora [S] | 0.366979 +/- 0.041030 | 0.522789 +/- 0.064136 | 0.680172 +/- 0.054884 | 0.687575 +/- 0.074353 |
| florist [S] | 0.378395 +/- 0.039805 | 0.528438 +/- 0.057718 | 0.672029 +/- 0.055242 | 0.681210 +/- 0.078796 |
| florg [S] | 0.464317 +/- 0.014382 | 0.535864 +/- 0.013703 | 0.653796 +/- 0.023395 | 0.673823 +/- 0.039049 |
| spectral_surgery | 0.255270 +/- 0.033749 | 0.640903 +/- 0.317734 | 0.799 +/- 0.126 | 0.793632 +/- 0.112417 |
| alignfed_calibration | 0.277694 +/- 0.020418 | 0.444717 +/- 0.032554 | 0.563 +/- 0.039 | 0.564158 +/- 0.029795 |
| rift | 0.262853 +/- 0.024434 | 0.441927 +/- 0.020712 | 0.536 +/- 0.022 | 0.531404 +/- 0.019812 |
| rift_diag | 0.205356 +/- 0.022635 | 0.427704 +/- 0.011417 | 0.489 +/- 0.021 | 0.487065 +/- 0.018642 |
| rift_core | **0.202688 +/- 0.021634** | **0.423123 +/- 0.018457** | **0.459 +/- 0.015** | **0.452990 +/- 0.011390** |

## Harmful Update Rate

Don vi: percent measured updates. Thap hon la tot hon.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| raw | 28.12 | 56.25 | 60.94 | 60.94 |
| freshness | 26.04 | 51.56 | 58.33 | 58.33 |
| fedrot | 33.33 | 54.69 | 54.69 | 54.69 |
| fedavg_lora [E-legacy] | 35.94 | 51.04 | 53.65 | 53.65 |
| flora_lora [E] | 28.13 | 57.81 | 59.90 | 59.90 |
| ffa_lora [E-legacy] | 35.42 | 56.25 | 51.56 | 51.56 |
| fedex_lora [E-dagger] | 41.67 | 42.71 | 58.33 | 59.38 |
| flexlora [S] | 35.42 | 55.73 | 55.73 | 55.73 |
| florist [S] | 34.38 | 52.08 | 56.77 | 56.77 |
| florg [S] | 28.65 | 55.73 | 51.04 | 51.04 |
| spectral_surgery | 28.12 | 56.77 | 60.42 | 60.42 |
| alignfed_calibration | 9.90 | 17.71 | 17.71 | 17.71 |
| rift | **2.60** | 25.52 | 23.44 | 23.44 |
| rift_diag | 11.98 | **11.98** | **5.21** | **5.21** |
| rift_core | 9.90 | 12.50 | 8.85 | 8.85 |

## Late Harmful Update Rate

Don vi: percent late updates. Thap hon la tot hon.

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| raw | 59.52 | 38.10 | 52.38 | 52.38 |
| freshness | 57.14 | 28.57 | 52.38 | 52.38 |
| fedrot | 61.90 | 35.71 | 57.14 | 57.14 |
| fedavg_lora [E-legacy] | 69.05 | 35.71 | 52.38 | 52.38 |
| flora_lora [E] | 59.52 | 42.86 | 52.38 | 52.38 |
| ffa_lora [E-legacy] | 61.90 | 38.10 | 59.52 | 59.52 |
| fedex_lora [E-dagger] | 71.43 | 52.38 | 52.38 | 52.38 |
| flexlora [S] | 71.43 | 35.71 | 57.14 | 57.14 |
| florist [S] | 69.05 | 38.10 | 57.14 | 57.14 |
| florg [S] | 69.05 | 35.71 | 61.90 | 61.90 |
| spectral_surgery | 57.14 | 40.48 | 52.38 | 52.38 |
| alignfed_calibration | 16.67 | 11.90 | 21.43 | 21.43 |
| rift | **4.76** | 21.43 | 7.14 | 7.14 |
| rift_diag | 19.05 | **4.76** | **2.38** | **2.38** |
| rift_core | **4.76** | 19.05 | 9.52 | 9.52 |

## Per-task Verdict

| Task | Best accuracy | Best class NLL | Best harmful | Best late harmful | Verdict |
|---|---|---|---|---|---|
| SST-2 | rift_diag / rift_core | rift_core | rift | rift / rift_core | GO signal for RIFT family |
| QNLI | rift_core | rift_core | rift_diag | rift_diag | GO signal for RIFT-core/RIFT-diag |
| MNLI-m | rift_core | rift_core | rift_diag | rift_diag | GO signal for RIFT-core/RIFT-diag |
| MNLI-mm | rift_core | rift_core | rift_diag | rift_diag | GO signal for RIFT-core/RIFT-diag |

## Ket Luan Hien Tai

Hien tai, voi bon task held-out da du 6 seeds:

- `rift_core` la method manh nhat ve accuracy trung binh tren SST-2, QNLI,
  MNLI-m va MNLI-mm.
- `rift_core` cung co class NLL thap nhat tren ca bon task.
- `rift_diag` dac biet tot tren QNLI, MNLI-m va MNLI-mm harmful/late harmful.
- Ban `rift` goc co harmful thap nhat tren SST-2, nhung khong manh bang
  `rift_core`/`rift_diag` ve accuracy va NLL.
- Trong legacy extended cohort, `flora_lora` co SST-2 accuracy 90.137%.
  Cac so `fedavg_lora`/`ffa_lora [E-legacy]` khong con duoc dung de xep hang;
  ban corrected-factor nam trong report rieng da link o dau file.
- Trong so sanh protocol-matched cross-batch, `rift_core` van cao hon cac
  baseline hop le da du 6 seed ve accuracy va thap hon ve class NLL tren ca bon
  task. `rift_diag`/`rift_core` cung giu harmful update rate thap hon ro ret.
- Trong spectral suite moi, FlexLoRA tot nhat tren SST-2/QNLI con FLoRG tot
  nhat tren MNLI-m/MNLI-mm. Tuy vay, `rift_core` van cao hon ca ba ve mean
  accuracy va thap hon ve Class NLL tren ca bon task; safety gap cung con lon.
- Khong dung hang `fedex_lora [E-dagger]` de claim FedEx thang/thua cho den khi
  co du 6/6 seed. FLoRG `[S]` da sua device mismatch va co du 6/6 seed; chi
  claim trong immediate-async simulator, khong claim full paper reproduction.

Dung muc claim:

- Co the noi RIFT family dang co tin hieu GO tren SST-2, QNLI, MNLI-m va MNLI-mm
  held-out.
- Co the claim RIFT-core/RIFT-diag vuot cac control hien tai trong simulator
  nay tren cohort 4 task, 6 seeds/task.
- Chua nen claim vuot official full-paper implementations cua AlignFed hay
  Spectral Surgery; trong simulator hien tai, chung la matched proxy/control.

## Paper-Baseline Audit Moi

Cap nhat them sau khi implement baseline gan paper hon:

- Output root: `outputs/paper_baseline_development`.
- Matrix: `configs/paper_baseline_audit_matrix.json`.
- Day la audit development rieng, khong tron truc tiep vao bang confirmation v3
  phia tren.
- Khac v3: seed `7201`, homogeneous rank `[8,8,8,8]`, train target modules
  day du `q/k/v/o/gate/up/down`, spectral post-hoc edit tren `o_proj/down_proj`.
- `spectral_surgery_posthoc` da chay du 4 task cho seed `7201`.
- `alignfed_reference` da thu SST-2 seed `7201` tren GPU local 4GB nhung chay
  qua lau do buffered semantic transform; job bi dung thu cong, chua co
  `result.json`. Nen chua co so lieu hop le de dien vao bang.

### Spectral Surgery Posthoc Seed7201

| Task | Accuracy | Class NLL | Harmful | Late harmful | Runtime |
|---|---:|---:|---:|---:|---:|
| SST-2 | 94.336% | 0.882676 | 71.88% | 71.43% | 306s |
| QNLI | 88.574% | 1.171470 | 65.62% | 42.86% | 513s |
| MNLI-m | 71.191% | 2.140890 | 50.00% | 28.57% | 661s |
| MNLI-mm | 73.828% | 1.787798 | 50.00% | 28.57% | 658s |

Nhan xet tam thoi:

- Spectral post-hoc co the tang accuracy tren SST-2, QNLI va MNLI-mm voi seed
  nay, nhung class NLL tang rat manh.
- Tren MNLI-m, accuracy giam va NLL tang, nen chua the coi day la baseline on
  dinh.
- Ket qua nay la canary mot seed, khong du thay cho confirmation 6 seeds.
- De so sanh paper-faithful dung nghia, can chay them `alignfed_reference` tren
  may co VRAM lon hon hoac Kaggle/T4x2, vi local 4GB khong thuc te cho full
  buffered AlignFed.

## Week 9 RAVAN Async Held-Out Supplement

Nguon ket qua rieng:

- Report chi tiet: `docs/week9/ravan_async_heldout_confirmation_results_vi.md`.
- Output root: `outputs/week9_ravan_confirmation_heldout`.
- Runner: `scripts/run_ravan_confirmation.py`.
- Lenh da chay:

```powershell
python scripts/run_ravan_confirmation.py --output-root outputs/week9_ravan_confirmation_heldout --mode confirmation --gpu 0
```

Trang thai: hoan thanh du `24/24` job, khong OOM, khong crash runtime. Tong
runtime worker ghi nhan khoang `3.15` gio tren GPU local 4GB.

Pham vi:

- Method: `ravan_async` only.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit.
- Tasks: SST-2, QNLI, MNLI-m, MNLI-mm.
- Seeds: `6101-6106`.
- Eval: `1024` held-out examples moi task.
- Regime: label-shard non-IID, heterogeneous rank, high staleness.
- Returns: `8` warmup + `32` measured.
- Ghi chu fidelity: day la RAVAN async adaptation trong simulator hien tai, khong
  phai full-paper RAVAN reproduction.

### RAVAN Aggregate

| Task | Final Acc mean | Acc std | Baseline Acc | Acc win seeds | Final NLL mean | Baseline NLL | NLL win seeds | Harmful mean | Late harmful mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SST-2 | 90.49% | 2.54 | 74.12% | 6/6 | 0.2376 | 0.5363 | 6/6 | 41.15% | 57.14% |
| QNLI | 71.35% | 13.95 | 72.07% | 4/6 | 1.1311 | 0.5328 | 4/6 | 58.33% | 40.48% |
| MNLI-m | 64.34% | 7.53 | 72.56% | 2/6 | 1.0932 | 0.6828 | 0/6 | 58.85% | 40.48% |
| MNLI-mm | 63.95% | 5.90 | 71.39% | 0/6 | 1.0587 | 0.7094 | 0/6 | 58.85% | 40.48% |

### Bang Tham Chieu Voi Cohort Chinh

Bang nay chi de doc nhanh vi `ravan_async` duoc chay trong output root rieng
`outputs/week9_ravan_confirmation_heldout`. Cac row con lai lay tu cohort
confirmation chinh `outputs/rift_core_confirmation_v3_1_5b`.

Accuracy, don vi mean +/- std percentage point:

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| raw | 90.120 +/- 1.843 | 74.658 +/- 9.710 | 68.392 +/- 4.167 | 68.083 +/- 3.679 |
| freshness | 87.679 +/- 3.131 | 74.593 +/- 6.616 | 69.010 +/- 5.771 | 68.620 +/- 5.307 |
| fedrot | 82.389 +/- 2.491 | 73.014 +/- 4.079 | 71.354 +/- 3.021 | 71.354 +/- 2.632 |
| spectral_surgery | 90.316 +/- 1.720 | 74.023 +/- 10.501 | 68.066 +/- 4.312 | 67.660 +/- 3.590 |
| alignfed_calibration | 90.088 +/- 1.516 | 79.492 +/- 2.225 | 76.937 +/- 3.037 | 77.425 +/- 2.136 |
| rift | 90.609 +/- 1.450 | 79.329 +/- 1.517 | 78.483 +/- 1.839 | 79.020 +/- 1.128 |
| rift_diag | 92.285 +/- 1.024 | 80.680 +/- 0.721 | 80.762 +/- 1.406 | 81.038 +/- 0.944 |
| rift_core | **92.285 +/- 0.865** | **80.843 +/- 1.065** | **82.210 +/- 1.197** | **81.901 +/- 0.717** |
| ravan_async | 90.495 +/- 2.543 | 71.354 +/- 13.951 | 64.339 +/- 7.532 | 63.949 +/- 5.900 |

Class NLL, thap hon la tot hon:

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| raw | 0.263043 +/- 0.034779 | 0.601147 +/- 0.265166 | 0.780 +/- 0.115 | 0.776886 +/- 0.108301 |
| freshness | 0.312909 +/- 0.040382 | 0.541523 +/- 0.148201 | 0.735 +/- 0.120 | 0.732654 +/- 0.110383 |
| fedrot | 0.384843 +/- 0.037355 | 0.529501 +/- 0.055123 | 0.667 +/- 0.047 | 0.676905 +/- 0.070920 |
| spectral_surgery | 0.255270 +/- 0.033749 | 0.640903 +/- 0.317734 | 0.799 +/- 0.126 | 0.793632 +/- 0.112417 |
| alignfed_calibration | 0.277694 +/- 0.020418 | 0.444717 +/- 0.032554 | 0.563 +/- 0.039 | 0.564158 +/- 0.029795 |
| rift | 0.262853 +/- 0.024434 | 0.441927 +/- 0.020712 | 0.536 +/- 0.022 | 0.531404 +/- 0.019812 |
| rift_diag | 0.205356 +/- 0.022635 | 0.427704 +/- 0.011417 | 0.489 +/- 0.021 | 0.487065 +/- 0.018642 |
| rift_core | **0.202688 +/- 0.021634** | **0.423123 +/- 0.018457** | **0.459 +/- 0.015** | **0.452990 +/- 0.011390** |
| ravan_async | 0.237574 +/- 0.045440 | 1.131052 +/- 0.983486 | 1.093225 +/- 0.403297 | 1.058682 +/- 0.306024 |

Harmful update rate, don vi percent measured updates:

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| raw | 28.12 | 56.25 | 60.94 | 60.94 |
| freshness | 26.04 | 51.56 | 58.33 | 58.33 |
| fedrot | 33.33 | 54.69 | 54.69 | 54.69 |
| spectral_surgery | 28.12 | 56.77 | 60.42 | 60.42 |
| alignfed_calibration | 9.90 | 17.71 | 17.71 | 17.71 |
| rift | **2.60** | 25.52 | 23.44 | 23.44 |
| rift_diag | 11.98 | **11.98** | **5.21** | **5.21** |
| rift_core | 9.90 | 12.50 | 8.85 | 8.85 |
| ravan_async | 41.15 | 58.33 | 58.85 | 58.85 |

Late harmful update rate, don vi percent late updates:

| Method | SST-2 | QNLI | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|
| raw | 59.52 | 38.10 | 52.38 | 52.38 |
| freshness | 57.14 | 28.57 | 52.38 | 52.38 |
| fedrot | 61.90 | 35.71 | 57.14 | 57.14 |
| spectral_surgery | 57.14 | 40.48 | 52.38 | 52.38 |
| alignfed_calibration | 16.67 | 11.90 | 21.43 | 21.43 |
| rift | **4.76** | 21.43 | 7.14 | 7.14 |
| rift_diag | 19.05 | **4.76** | **2.38** | **2.38** |
| rift_core | **4.76** | 19.05 | 9.52 | 9.52 |
| ravan_async | 57.14 | 40.48 | 40.48 | 40.48 |

### RAVAN Verdict

RAVAN async co tin hieu rat manh tren SST-2: `6/6` seed thang baseline ca
accuracy va class NLL.

QNLI la ket qua mixed: `4/6` seed thang baseline, nhung seed `6104` va `6106`
sup nang lam mean accuracy thap hon baseline va mean NLL tang rat manh.

MNLI-m va MNLI-mm la **NO-GO voi RAVAN async config hien tai**: accuracy mean
thap hon baseline, NLL xau hon baseline, MNLI-mm khong co seed nao thang
accuracy baseline.

Dung muc claim:

- RAVAN async khong duoc gop vao bang RIFT-Core main result de claim RIFT thang.
- Ket qua nay nen dung nhu mot competitor/related-method diagnostic cho Week 9.
- Gia tri nghien cuu chinh: no cho thay multi-head/frozen-basis async update co
  the rat hop voi binary SST-2, nhung can stability gate/aggregation moi neu
  muon dung cho entailment 3 lop nhu MNLI.

## Cach Them Ket Qua Sau Nay

Khi co output confirmation hoac supplement moi, them vao file nay theo cung
format va giu tach bach giua cohort chinh voi experiment bo sung:

1. Dat output vao root rieng, khong ghi de output root da dung de lap bang cu.
2. Kiem tra `completion.json`, so job expected/completed/missing va moi
   `result.json` co dung config/source hash.
3. Chi dien cot/bang khi moi method trong cohort do co du seeds da dang ky.
4. Bao cao mean +/- std tren tat ca seeds, khong cherry-pick.
5. Neu la supplement nhu RAVAN async, dat thanh section rieng va khong tron
   vao bang RIFT-Core main result.
6. Cap nhat `Per-task Verdict` va `Ket Luan Hien Tai` chi khi cohort chinh thay
   doi; supplement thi cap nhat verdict rieng trong section supplement.
