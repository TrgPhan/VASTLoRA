# Week 9 Qwen 7B: FedLoRA [E] + [S]

## Pham vi

Profile `qwen7b-fedlora` chay 12 job tren cung protocol Dolly development da
dung cho convergence 7B:

- `[E]`: `fedavg_lora`, `ffa_lora`, `flora_lora`.
- `[S]`: `flexlora`, `florist`, `florg`.
- Seeds: `9101`, `9102`.
- Regime: `noniid_high_staleness`.
- Budget: 8 warmup + 512 measured returns; milestone 64/128/256/512.
- Model: `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, NF4, `q_proj/v_proj`.

Day la development extension, khong phai held-out confirmation va khong duoc
chon checkpoint tot nhat tren test set.

## Audit fidelity

| Method | Operator giu lai | Bien thich nghi cua simulator |
|---|---|---|
| FedAvg-LoRA | Trung binh truc tiep persistent `A/B` | Mot arrival/luc; prefix rank + zero padding |
| FFA-LoRA | Chung `A0` bat bien, chi train/aggregate `B` | Mot arrival/luc; prefix rank cho `B` |
| FLoRA | Stack `B` ngang, `A` doc de tao tong co trong so cua `BA` | Stack server + arriving innovation, sau do common rank cap |
| FlexLoRA | Trung binh product va SVD phan phoi top-r | Convex mix server/client moi arrival |
| FLoRIST | Factor SVD, core SVD, squared-energy SVT | `tau=0.9`, sau do common rank cap |
| FLoRG | Mot matrix `A`, `A.T@A`, semi-orthogonal `L/R`, Procrustes | Immediate async va heterogeneous active-rank |

Nguon code/paper doi chieu ngay 2026-09-26:

- FedEx-LoRA official commit `2fa2e4a243f93c829e2b601e459e45aa748e4149`.
- FLoRA official commit `7bea15826bc0de37da35e44bc34a39274e3cc09f`.
- FlexLoRA official FederatedScope branch commit
  `1cb1ab76bf4c9394d617c4c5a800cb0df145796d`.
- FLoRIST official commit `670a2247a32b4fc8ac165d0f9691416e2cc41c15`.
- FLoRG duoc doi chieu voi paper ICLR 2026; khong tim thay official code repo
  duoc paper cong bo, nen day la equation-level reimplementation co unit test.

Khong goi cac run nay la full reproduction cua paper: cac paper FedLoRA tren
chu yeu aggregate theo communication round, con RIFT simulator xu ly mot stale
return moi lan. Claim hop le la `paper aggregation operator adapted to the
matched immediate-async protocol`.

## Tai sao khong co FedEx-LoRA

FedEx dung nghia tinh residual
`mean(B_i A_i) - mean(B_i) mean(A_i)` va fold residual vao frozen base weight.
Voi Qwen 7B NF4, sua truc tiep base quantized khong con la cung operator. Giu
dense residual rieng cho moi stale server version lai tang host RAM theo so
layer va so version; batch 1.5B truoc da bi `SIGKILL 9`. Low-rank truncate
residual de ep chay se mat tinh exact, nen khong dua mot baseline gan nhan sai
vao cohort 7B.

## Lenh chay

Sau khi deploy clean release, tao dung environment da pin roi moi preflight:

```bash
bash scripts/bootstrap_week9_remote.sh
bash scripts/run_week9_7b_fedlora.sh preflight
bash scripts/run_week9_7b_fedlora.sh smoke
bash scripts/run_week9_7b_fedlora.sh run
```

Neu job bi ngat, runner giu cac `result.json` hop le va chi khoi dong lai job
chua day du:

```bash
bash scripts/run_week9_7b_fedlora.sh retry
bash scripts/run_week9_7b_fedlora.sh report
```

Mac dinh mot worker/GPU de FLoRG co du VRAM cho fixed `L/R` bases. GPU tu
48 GB co the thu hai worker sau smoke:

```bash
WEEK9_FEDLORA_WORKERS_PER_GPU=2 bash scripts/run_week9_7b_fedlora.sh run
```

Khong doi workers giua cac method vi no chi la throughput setting, nhung phai
bao cao wall time va peak memory rieng cho FLoRG.

## Metric bat buoc

Bao cao token NLL, ROUGE-L precision/recall/F1, exact match, harmful update,
late harmful, response length/limit rate, runtime, peak GPU memory, transmitted
parameters va retained rank. Khong ket luan chi tu ROUGE-L hoac mot seed.
