# Week 4 Debug va Rescue - ket qua cuoi

Ngay chay: 2026-08-24

## Verdict

**VAST goc: NO-GO tai gate Week 4.**

Khong con du bang chung de quy ket qua te cho optimizer hay mot loi factor don gian. Sau khi sua pipeline, `rho_two_sided` van khong du bao utility tren held-out data va VAST khong thang freshness.

**Huong thesis: co the cuu bang pivot sang corrective projection, nhung chua phai GO cho claim VAST goc.** Projection-only thang freshness va VAST tren 3/3 held-out seed trong regime non-IID/high-staleness. Claim moi phai duoc doi chieu ky voi FedSteer, GLoRA va cac projection/alignment baseline.

## Loi va confound da tim thay

1. Client cu khoi tao `A` moi va `B=0` o moi return. Day khong phai continuation tu adapter server va lam right overlap gan random. Pipeline moi factorize server adapter va gui dung `B0,A0`; message la `B1 A1 - B0 A0`.
2. Cac gia tri `last_loss` tai step 4/20/100 thuoc cac batch khac nhau, nen khong tao thanh mot loss curve tang dan. Probe co dinh cho thay local loss giam.
3. Checkpoint `gokuls/BERT-tiny-sst2` da fine-tune tren SST-2 va evaluator 128 mau qua nho. Checkpoint sach moi warm-start tu `google/bert_uncased_L-2_H-128_A-2` tren 5.000 mau; 62.349 mau con lai danh cho FL; evaluator dung du 872 validation examples.
4. Config cu `LR=5e-3, 4 steps` gay client drift: accuracy `73.97% -> 71.44%`, `A` doi 53.6% va chi giu 80.1% initial subspace. `LR=1e-3, 1 step` on dinh hon va dat `74.43%` trong IID dev run.
5. Temporal reference cu coi moi singular direction ngang nhau. Da them `reference_singular_power`, nhung weighted reference chi lam rho dep hon, khong lam task loss tot hon; khong chon no.

## Baseline cung protocol

Day la baseline theo nguyen ly tren cung code/trace, khong phai faithful reproduction day du cua paper ngoai.

- `raw`: naive asynchronous intrinsic innovation; cung la gauge-invariant single-return control.
- `freshness`: whole-update exponential decay.
- `projection`: two-sided corrective projection, FedSteer-inspired control.
- `vast`: projected component cong freshness-scaled residual.

### Dev seeds 17/31/43, non-IID/high-staleness

VAST hon freshness o ca ba seed, mean accuracy gain `+1.26 pp`. Ket qua nay khong lap lai tren held-out seeds, nen duoc xem la development overfit, khong phai confirmation.

### Held-out seeds 59/71/89

| Method | Mean end loss | Mean loss drift | Mean end accuracy | Mean accuracy drift |
|---|---:|---:|---:|---:|
| Freshness | 0.563904 | +0.019625 | 71.75% | -1.22 pp |
| VAST | 0.564928 | +0.020649 | 71.02% | -1.95 pp |
| Projection-only | **0.543968** | **-0.000311** | **73.74%** | **+0.77 pp** |

Projection-only thang tren 3/3 seed. VAST kem freshness trung binh `-0.73 pp` accuracy.

## Kill-test cua rho

Tren 180 stale returns cua held-out freshness trajectories:

- partial Spearman `rho_two_sided` vs utility, control `tau`: **-0.174**;
- bootstrap CI95%: **[-0.312, -0.031]**;
- positive seed fraction: **1/3**;
- them rho vao tau lam CV R2 gain **-0.015**;
- harmful-update AUROC gain **-0.054**.

Do do menh de "historical-subspace compatibility cao thi update huu ich hon" bi bac bo trong protocol nay. Projection van co ich vi no loai stale residual, khong phai vi rho la mot utility score tot.

## Thu rescue residual

Da thu `mu_residual = mu^p`:

- dev seed 17: `p=2` va `p=4` giam loss so voi `p=1`;
- unseen seed 101: `p=2` dat 66.51%, kem freshness 68.81%, loss kem 0.018.

Vi gate unseen fail, khong tiep tuc tune theo seed 101.

## Viec con thieu truoc khi khang dinh thesis moi

1. Reproduce Sync FedAvg-LoRA, FedBuff, GLoRA-like + freshness va mot projection baseline faithful hon.
2. Chay projection pivot tren IID/heterogeneous-rank va them task/backbone thu hai.
3. Dung validation split rieng cho hyperparameter selection, confirmation va final report.
4. Neu projection tiep tuc thang, viet lai novelty quanh stale-residual suppression/corrective transport; neu khong thang strong baselines thi chuyen thanh negative empirical study.

## Reproducibility

Checkpoint sach duoc tao boi `scripts/prepare_week4_backbone.py`; manifest nam tai `outputs/models/bert_tiny_sst2_clean/warmstart_manifest.json`. Config da khoa nam tai `configs/week4_rescue.json`. Raw artifacts nam trong `outputs/week4_rescue_confirmation/` va `outputs/week4_residual_confirmation/`.
