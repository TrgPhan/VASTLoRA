# Week 4 - RIFT-LoRA novelty review

Ngay review: 2026-08-26

## Ket luan ngan

RIFT-LoRA co novelty o muc **co the bao ve cho thesis**, nhung khong nen claim
rang tung thanh phan deu moi.

Claim nen giu:

> RIFT la mot calibration-assisted, gauge-invariant, rank-wise safety layer cho
> delayed LoRA innovations trong heterogeneous-rank asynchronous FedLoRA. No loc
> tung intrinsic singular component theo first-order objective gain va dung mot
> paired held-out loss gate de scale/reject late updates.

Khong nen claim:

- "SVD cua LoRA la moi";
- "calibration gradient de cham singular component la moi";
- "gate bang validation/calibration loss la moi";
- "RIFT da thang full official GLoRA/FedSteer/AlignFed".

## Overlap gan nhat

### Spectral Surgery

Day la doi thu novelty gan nhat ve mat ky thuat.

Spectral Surgery decomposes a trained LoRA update by SVD, estimates
per-component sensitivity using gradients on a small calibration set, and
reweights singular values while keeping directions fixed.

Vi vay, neu RIFT chi duoc viet la "gradient-guided singular component filtering
for LoRA", novelty se yeu.

Khac biet can nhan manh:

- Spectral Surgery la post-hoc refinement cho trained adapter rieng le.
- RIFT la server-side online safety operator cho delayed client updates trong
  asynchronous federated LoRA.
- RIFT xu ly exact stale innovation `B_final A_final - B_dispatch A_dispatch`,
  khong chi edit mot adapter da train xong.
- RIFT co paired held-out gate tach khoi gradient split de chap nhan, scale,
  hoac reject update.
- Metric trung tam cua RIFT la late harmful-update reduction, cumulative harm,
  trajectory loss safety, khong phai chi final task score.

### AdaLoRA

AdaLoRA da dung SVD-style parameterization va importance-aware rank pruning.
No la inspiration hop ly cho viec xem singular directions nhu cac don vi co
importance khac nhau. RIFT khong nen claim rank pruning la moi.

Khac biet:

- AdaLoRA la local PEFT training/rank allocation.
- RIFT la server aggregation/safety trong async FL.
- RIFT cham component bang gradient cua objective hien tai doi voi stale
  innovation, khong phai budget allocation trong qua trinh local fine-tuning.

### FedEx-LoRA

FedEx-LoRA giai quyet exact aggregation: average factor `A/B` khong bang average
intrinsic update `BA`. RIFT nen dung exact innovation lam input, khong doi dau
voi FedEx nhu mot replacement.

Khac biet:

- FedEx sua loi algebra/aggregation exactness.
- RIFT sua objective mismatch cua update da cu so voi server hien tai.

### FedRot-LoRA / FLoRG / GLoRA

Nhom nay giai quyet factor gauge, rotational misalignment, consensus subspace,
rank-compatible readout, hoac Gram/Procrustes aggregation.

Khac biet:

- Chung tra loi "lam sao represent/aggregate LoRA dung hinh hoc?"
- RIFT tra loi "trong mot update tre, thanh phan nao con la descent direction
  cua objective hien tai?"
- Mot update dung gauge/subspace van co the harmful neu stale va non-IID.

### FedSteer

FedSteer la doi thu gan nhat ve staleness. No dung cached gradient subspace va
corrective projection/caching de steer outdated gradients.

Khac biet:

- FedSteer nham vao gradient staleness va inactive-client replay trong FL noi
  chung.
- RIFT nham vao LoRA intrinsic update, SVD components, heterogeneous rank, va
  objective-gated late update acceptance.
- RIFT khong replay inactive clients; no loc update thuc su vua ve server.

### AlignFed / OrthoFL

AlignFed va OrthoFL la doi thu gan ve async calibration/alignment.

Khac biet:

- AlignFed dung version-aware grouping, cross-version semantic alignment tren
  calibration mini-batch, va fairness/freshness aggregation.
- OrthoFL decouples global/local progress va calibrates global shifts de giam
  interference.
- RIFT khong hoc transform semantic/cross-version; no scoring rank-one LoRA
  innovation components bang first-order loss signal va dung paired gate.

## Novelty verdict

RIFT **khong moi tuyet doi** o tung building block. Building blocks da co:

- SVD/rank-wise LoRA importance: AdaLoRA va nhieu PEFT spectral methods.
- Calibration-gradient singular sensitivity: Spectral Surgery.
- Exact intrinsic LoRA update: FedEx-LoRA.
- Gauge/subspace-aware FedLoRA: FedRot-LoRA, FLoRG, GLoRA.
- Staleness correction: FedSteer, AlignFed, OrthoFL, freshness-weighted AsyncFL.
- Validation/calibration gate: robust FL va async FL co lien quan.

Nhung RIFT **co novelty ket hop va bai toan**:

1. Dinh nghia delayed LoRA innovation theo intrinsic dense update:
   `D = B_final A_final - B_dispatch A_dispatch`.
2. Tach `D` thanh SVD rank-one components.
3. Cham moi component bang first-order current-objective gain
   `gain_j = -sigma_j u_j^T G v_j`.
4. Giu component co predicted descent signal, bo component predicted harmful.
5. Dung split calibration rieng de scale/reject update bang paired loss gate.
6. Danh gia bang late harmful-update rate/cumulative harm trong non-IID,
   high-staleness, heterogeneous-rank Async FedLoRA.

Day la du de GO cho **thesis prototype** neu claim duoc viet hep:

> RIFT is an objective-safety layer for stale LoRA updates, complementary to
> exact/gauge-aware aggregation methods.

## Claim nen dung trong thesis

Tieng Anh:

> We propose RIFT-LoRA, a calibration-assisted safety layer for asynchronous
> federated LoRA. Unlike exact or gauge-aware aggregation methods that improve
> how LoRA updates are represented, RIFT asks whether each intrinsic singular
> component of a delayed client innovation is still aligned with the current
> server objective. It filters rank-one components using a first-order
> calibration-gradient score and accepts only scaled candidates that improve a
> disjoint paired calibration-loss gate.

Tieng Viet:

> RIFT-LoRA khong thay the FedEx, FedRot hay GLoRA. No la mot lop an toan cho
> update tre: sau khi co update LoRA dung ve mat intrinsic/gauge, RIFT kiem tra
> tung singular component co con giam loss cua server hien tai hay khong, roi
> scale/reject bang mot calibration gate doc lap.

## Gate can lam de bao ve novelty tot hon

1. So sanh RIFT wrapper tren FedEx/FedRot/GLoRA-like candidate update.
2. Them ablation: Spectral-Surgery-style reweighting without async gate.
3. Them ablation: whole-update calibration gate only.
4. Bao cao late harmful rate, cumulative late harm, worst-step loss increase,
   acceptance rate, calibration shift, calibration size.
5. Chay generative task co token NLL/perplexity va classification task co
   trajectory loss.
6. Viet ro chi can 512 calibration examples va kiem tra calibration/client/eval
   disjoint.

## Sources checked

- FedEx-LoRA: https://arxiv.org/abs/2410.09432
- FedRot-LoRA: https://arxiv.org/abs/2602.23638
- FLoRG: https://arxiv.org/abs/2602.17095
- GLoRA: https://arxiv.org/abs/2605.06733
- AlignFed: https://arxiv.org/abs/2606.08197
- FedSteer: https://arxiv.org/abs/2606.10124
- FSLoRA: https://arxiv.org/abs/2501.19389
- SDFLoRA: https://arxiv.org/abs/2601.11219
- AdaLoRA: https://arxiv.org/abs/2303.10512
- Spectral Surgery: https://arxiv.org/abs/2603.03995
- OrthoFL / Taming Update Drift: https://doi.org/10.1145/3770855.3817907
