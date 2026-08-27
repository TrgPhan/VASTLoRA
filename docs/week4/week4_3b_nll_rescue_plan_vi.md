# Week 4 - Ke hoach cuu NLL cho MTIP tren 3B

Ngay cap nhat: 2026-08-25

## Ket luan tu run 3B cu

MTIP 3B hien tai la **INCONCLUSIVE**, chua phai GO va cung chua du bang chung de
ket luan NO-GO tuyet doi.

Tren ba seed `2026/2027/2028`:

| Method | Accuracy | Sequence NLL | Binary NLL |
|---|---:|---:|---:|
| Freshness | 94.401% | 4.6147 | 0.14636 |
| VAST | 94.271% | **2.3825** | 0.14930 |
| MTIP | **95.443%** | 11.8754 | 0.23481 |
| MTIP adaptive | 94.922% | 11.0940 | 0.24368 |

Fixed MTIP tang trung binh `+1.042 pp` accuracy, nhung accuracy gain theo seed la
`-0.391/+2.344/+1.172 pp`. MTIP lam sequence NLL xau hon `+7.2607` va binary
NLL xau hon `+0.08846`. Trong 768 cap du doan, MTIP chi thay doi 22 prediction:
sua 15 loi cua Freshness va lam hong 7 prediction dung.

## Chan doan

Day khong giong divergence hay loi so hoc:

- khong co NaN/Inf hay runtime failure;
- MTIP van ha NLL so voi pretrained baseline;
- `rho` trung binh chi khoang `0.129`, nen projection-only loai khoang 87% nang
  luong innovation;
- VAST giu residual da freshness-scale va cai thien sequence likelihood rat ro.

Do do chan doan hien tai la **projection qua manh dan den under-learning
likelihood**. Accuracy tang vi mot so mau gan decision boundary doi dau, trong
khi confidence va margin cua phan lon mau bi nen.

Run 3B cu cung chua scale faithful MTIP da chon o tiny experiment: no dung
`history_size=4/reference_rank=4`, trong khi MTIP confirmation dung
`history_size=8/reference_rank=2`. Config rescue sua lai diem nay.

## Method rescue

Voi innovation `D`, temporal projection `P(D)`, residual `R=D-P(D)` va
freshness `mu=exp(-lambda*tau)`:

```text
Hybrid-MTIP: T_beta(D) = P(D) + beta * mu * R
```

- `beta=0`: MTIP projection-only;
- `beta=1`: VAST;
- tune truoc `beta in {0.05, 0.1, 0.2, 0.4, 0.7}`.

Nhanh exploratory thu hai chi mo residual cho update con moi:

```text
Routed-MTIP: T(D) = P(D) + mu * sigmoid((tau0 - tau) / T) * R
```

Routed-MTIP co ly do co hoc ro rang, nhung khong duoc xem la target mac dinh.
No chi duoc confirm neu duoc selector chon tren development holdout truoc khi
xem confirmation result.

## Protocol chong leakage

1. Reserve deterministic holdout tu SST-2 `train`; cac index nay bi loai khoi
   client training.
2. Tune cac variant tren development seed `2024/2025`.
3. Ghi `variant` rieng de cac beta khong ghi de artifact cua nhau.
4. Freeze dung mot target variant tu `tradeoff_selection.json`.
5. Chay target cung Freshness, VAST, MTIP va adaptive MTIP tren official
   validation; moi seed dung cung 872 mau.
6. Reporting chi ra verdict khi `--target-variant` trung target da freeze.

Official SST-2 validation da duoc xem trong cac experiment truoc, nen mot GO o
day van chi la regime-specific evidence. Claim task-general can mot task moi va
khong duoc tune lai target theo task do.

## Pareto gate

Full GO can it nhat nam confirmation seed va dong thoi:

- mean balanced-accuracy gain `>= +0.5 pp`;
- lower 95% CI cua balanced-accuracy gain `> 0`;
- upper 95% CI cua sequence-NLL relative degradation `<= 10%`;
- upper 95% CI cua binary-NLL relative degradation `<= 5%`;
- upper 95% CI cua Brier relative degradation `<= 5%`;
- khong seed nao giam balanced accuracy qua `0.5 pp`;
- thang balanced accuracy tren it nhat 4/5 seed.

Neu selector khong co candidate pass development gate, notebook van co the chay
candidate gan gate nhat de chan doan, nhung ket qua do khong duoc tu dong goi la
GO. Neu target da freeze tiep tuc fail NLL/calibration gate tren nam seed, huong
residual-recovery nay la **NO-GO** va khong tiep tuc tune tren confirmation data.

## Artifact/code

- `configs/kaggle_3b_mtip_tradeoff.json`
- `scripts/tune_kaggle_mtip_tradeoff.py`
- `scripts/run_kaggle_3b.py`
- `src/riftlora/scale/tradeoff.py`
- `src/riftlora/scale/reporting.py`
- `notebooks/kaggle_qwen_3b_mtip_scale.ipynb`

