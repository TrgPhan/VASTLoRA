# Week 1 - Tong Hop Paper Lien Quan Cho VAST-LoRA

Ngay tong hop: 2026-08-10  
Muc tieu: dong bang tam thoi novelty map cho Week 1, tranh claim qua tay, va rut ra baseline bat buoc cho VAST-LoRA.

## 1. Cau hoi nghien cuu cua VAST-LoRA

VAST-LoRA nen duoc dinh vi quanh mot cau hoi hep:

> Khi client tra ve mot LoRA innovation bi stale trong asynchronous FL, ta co nen lam giam toan bo update chi vi no cu, hay nen tach phan van nam trong current optimization subspace va phan residual da lech khoi geometry hien tai?

Cong thuc loi:

$$
D_i = G_{i,E}^{(v_i)} - G_{i,0}^{(v_i)}
$$

voi LoRA adapter:

$$
G_{i,E}^{(v_i)} = B_{i,E}A_{i,E},
\qquad
G_{i,0}^{(v_i)} = B_{i,0}A_{i,0}.
$$

Innovation co low-rank representation chinh xac:

$$
D_i
=
\begin{bmatrix}
B_{i,E} & B_{i,0}
\end{bmatrix}
\begin{bmatrix}
A_{i,E}\\
-A_{i,0}
\end{bmatrix}.
$$

Sau do VAST tach:

$$
D_i = D_i^\parallel + D_i^\perp,
\qquad
D_i^\perp = D_i - D_i^\parallel.
$$

Operator trung tam:

$$
\mathcal{T}_t(D_i)
=
D_i^\parallel + \mu_iD_i^\perp,
\qquad
\mu_i=e^{-\lambda\tau_i}.
$$

Tuong duong:

$$
\mathcal{T}_t(D_i)
=
\mu_iD_i + (1-\mu_i)D_i^\parallel.
$$

## 2. Bang literature map

| Paper | Nam / nguon | Van de chinh | Y tuong chinh | Anh huong den VAST-LoRA |
|---|---:|---|---|---|
| GLoRA | 2026, arXiv:2605.06733 | Gauge ambiguity trong FedLoRA | Server aggregate bang gauge-aware low-rank representation, consensus subspace, rank-compatible readout | Threat lon nhat cho claim "gauge-aware FedLoRA". VAST chi nen muon principle gauge-invariant, khong claim day la novelty. |
| AlignFed | 2026, arXiv:2606.08197 | Async federated fine-tuning cho LLM, stale model drift, fairness | Version-aware grouping, semantic alignment bang calibration mini-batch, freshness + fairness aggregation | Threat lon nhat cho claim "async FFT for LLM". VAST nen nhan manh data-free, adapter-factor-only geometry. |
| FedSteer | 2026, arXiv:2606.10124 | Extreme gradient staleness | Dynamic gradient subspace, corrective projections, caching representative clients | Threat cho claim "subspace correction for stale FL". VAST khac o LoRA matrix innovation, rank heterogeneity, two-sided geometry. |
| FedRot-LoRA | 2026, arXiv:2602.23638 | Rotational misalignment trong LoRA factors | Orthogonal transformation de align client updates truoc aggregation | Khong claim "align LoRA factors" la moi. Nen dung lam strong related work/baseline neu co code. |
| FLoRG | 2026, arXiv:2602.17095 | Aggregation error va decomposition drift | Low-rank Gram matrix + Procrustes alignment | Threat ve Procrustes/alignment va decomposition drift. VAST khac vi sua stale residual theo current temporal reference. |
| HetLoRA | 2024, EMNLP / arXiv:2401.06432 | Heterogeneous LoRA rank cho on-device foundation models | Local rank self-pruning, sparsity-weighted aggregation | Khong claim heterogeneous rank la novelty. Dung lam baseline/setting. |
| FedEx-LoRA | 2025, ACL / arXiv:2410.09432 | Inexactness cua FedAvg tren LoRA adapters | Them residual error term vao frozen weight de dat exact aggregation | Bat buoc doc ky vi VAST cung thao tac tren exact update/innovation. |
| FLoRA | 2024, NeurIPS / arXiv:2409.05976 | Heterogeneous LoRA aggregation noise | Stacking-based aggregation ho tro heterogeneous adapters | Threat cho exact/heterogeneous LoRA aggregation. Can so sanh neu thesis noi ve rank heterogeneity. |
| SDFLoRA | 2026, arXiv:2601.11219 | Rank heterogeneity + privacy/personalization | Tach adapter thanh shared/global module va private/local module | Khong dua personalization/privacy vao core VAST. Chi cite la extension/threat. |
| PreLort | 2026, arXiv:2606.15963 | Rank heterogeneity trong FedLoRA | Prefix-nested LoRA, segment-wise aggregation | Threat cho rank-compatible heterogeneous client design. VAST khac vi stale correction. |
| FSLoRA | 2025, arXiv:2501.19389 | Resource heterogeneity | Sketching mechanism de client update submatrix cua global LoRA | Khong claim resource-aware rank scheduling. Dung lam related work. |
| QLoRA | 2023, NeurIPS / arXiv:2305.14314 | Memory-efficient LLM finetuning | Frozen 4-bit quantized base + LoRA adapters, NF4, double quantization, paged optimizers | Nen dung lam engineering tool, khong phai contribution. |

## 3. Ghi chu tung paper

### 3.1 GLoRA

Nguon: https://arxiv.org/abs/2605.06733

GLoRA chi ra loi can ban cua viec aggregate raw LoRA factors: cung mot update noi tai co vo so factorization tuong duong do gauge ambiguity:

$$
BA = (BQ)(Q^{-1}A),
$$

voi moi ma tran kha nghich $Q$.

Neu server average truc tiep $B_i$ va $A_i$, ket qua phu thuoc vao toa do factor chu khong chi phu thuoc vao update matrix $B_iA_i$. GLoRA xu ly bang gauge-aware server representation, uoc luong consensus update subspace tu projectors va aggregate trong shared coordinates. Paper nay con co rank-compatible readout cho client co rank khac nhau.

Tac dong den VAST:

- VAST phai dung representation gauge-invariant, vi raw factor comparison nhu $\cos(B_i,B_j)$ khong co nghia noi tai.
- VAST khong nen claim "giai quyet gauge ambiguity" la dong gop chinh.
- Baseline can co: GLoRA-like gauge-aware aggregation va GLoRA + freshness decay.

### 3.2 AlignFed

Nguon: https://arxiv.org/abs/2606.08197

AlignFed tap trung vao asynchronous federated fine-tuning cho LLM trong edge heterogeneity. Paper nay noi thang cac van de:

- straggler effect trong synchronous FL;
- stale updates gay model drift;
- non-IID gay client drift;
- fast clients co the dominate aggregation.

AlignFed dung:

- version-aware update grouping;
- cross-version semantic alignment bang mini-batch calibration set;
- freshness + fairness aggregation.

Tac dong den VAST:

- Khong claim "asynchronous federated LLM fine-tuning" la moi.
- VAST nen khac biet bang dieu kien **server-data-free**: khong can calibration mini-batch.
- Neu so sanh truc tiep, can ghi ro AlignFed dung semantic/calibration signal, con VAST dung adapter-factor geometry.

### 3.3 FedSteer

Nguon: https://arxiv.org/abs/2606.10124

FedSteer sua extreme gradient staleness bang dynamic gradient subspace. No tao subspace tu cache recent gradients, project active gradient vao subspace de lay coordinates, roi tai su dung coordinates voi evolved subspace cho inactive/stale clients.

Tac dong den VAST:

- Khong claim "subspace projection de sua stale update" la moi.
- VAST khac vi:
  - object la LoRA matrix innovation, khong phai dense/vector gradient chung;
  - co factorization/gauge ambiguity;
  - co heterogeneous rank;
  - can two-sided column/row geometry;
  - can low-rank algebra de khong materialize dense matrix.

### 3.4 FedRot-LoRA

Nguon: https://arxiv.org/abs/2602.23638 va https://openreview.net/forum?id=2X8Qi3VdjA

FedRot-LoRA noi ve rotational misalignment:

$$
(B_iR_i)(R_i^\top A_i)=B_iA_i
$$

khi $R_i$ la orthogonal matrix. Neu cac client dung latent bases khac nhau, factor-wise averaging co the pha huy semantic update.

Tac dong den VAST:

- Alignment cua LoRA factors khong con la novelty.
- VAST nen aggregate/sua tren innovation intrinsic form:

$$
D_i = U_i\Sigma_iV_i^\top.
$$

### 3.5 FLoRG

Nguon: https://arxiv.org/abs/2602.17095

FLoRG neu hai challenge:

- aggregating two LoRA factors rieng biet gay aggregation error;
- ngay ca khi aggregate product, factor recovery qua decomposition khong duy nhat va co decomposition drift.

Paper de xuat Gram matrix va Procrustes alignment de giam drift.

Tac dong den VAST:

- Can ghi ro VAST khong giai bai decomposition drift tong quat.
- VAST dung compact SVD canonical form de tinh geometry cua **innovation** va sua stale residual.

### 3.6 HetLoRA

Nguon: https://aclanthology.org/2024.emnlp-main.717/ va https://arxiv.org/abs/2401.06432

HetLoRA cho phep different client ranks cho on-device foundation models. No dung rank self-pruning local va sparsity-weighted aggregation server.

Tac dong den VAST:

- Heterogeneous rank la setting, khong phai novelty.
- VAST can co experiment homogeneous vs heterogeneous rank de xem benefit co tang khi rank diversity tang khong.

### 3.7 FedEx-LoRA

Nguon: https://aclanthology.org/2025.acl-long.67/ va https://arxiv.org/abs/2410.09432

FedEx-LoRA chi ra FedAvg tren LoRA adapters co the inexact. Huong chinh la them residual error term vao pretrained frozen weight matrix de dat exact update.

Tac dong den VAST:

- VAST phai rat can than voi local innovation:

$$
D_i = G_{i,E}^{(v_i)} - G_{i,0}^{(v_i)}.
$$

- Neu dung final adapter $G_{i,E}^{(v_i)}$ nhu update moi, server se double count kien thuc da co o dispatch version.

### 3.8 FLoRA

Nguon: https://proceedings.neurips.cc/paper_files/paper/2024/hash/28312c9491d60ed0c77f7fff4ad86dd1-Abstract-Conference.html va https://arxiv.org/abs/2409.05976

FLoRA de xuat stacking-based aggregation, ho tro heterogeneous LoRA adapters va tranh aggregation noise cua cach aggregate LoRA thong thuong.

Tac dong den VAST:

- VAST nen tach ro hai lop van de:
  - exact/heterogeneous aggregation: da co FLoRA/FedEx-LoRA/GLoRA;
  - stale innovation correction: VAST tap trung vao day.

### 3.9 SDFLoRA

Nguon: https://arxiv.org/abs/2601.11219

SDFLoRA tach client adapter thanh global/shared module va local/private module. Shared module duoc align/aggregate; private module giu local. Paper cung noi ve privacy-aware optimization bang DP noise tren global module.

Tac dong den VAST:

- Khong nen them personalization/privacy vao MVP.
- Co the cite trong limitation/extension: VAST hien chua xu ly private personalization module.

### 3.10 PreLort

Nguon: https://arxiv.org/pdf/2606.15963

PreLort dung prefix-nested LoRA de rank-thap va rank-cao tuong thich hon trong federated setting. Segment-wise aggregation chi average tren clients co dong gop cho segment rank tuong ung.

Tac dong den VAST:

- Rank-compatible design dang rat active.
- VAST nen xem rank heterogeneity la stress test, khong claim la contribution doc lap.

### 3.11 FSLoRA

Nguon: https://arxiv.org/abs/2501.19389

FSLoRA dung sketching mechanism de client update submatrices cua global LoRA modules. Sketching ratio dieu khien rank/submatrix tren client, phu hop resource constraints.

Tac dong den VAST:

- Resource-aware rank adaptation khong nen dua vao core thesis.
- Neu them rank scheduler se bien thanh paper khac.

### 3.12 QLoRA

Nguon: https://arxiv.org/abs/2305.14314 va https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html

QLoRA la engineering foundation de chay LLM fine-tuning voi GPU han che:

- frozen 4-bit quantized base model;
- LoRA trainable adapters;
- NF4;
- double quantization;
- paged optimizers.

Tac dong den VAST:

- Dung QLoRA de tiet kiem VRAM.
- QLoRA khong lien quan truc tiep den novelty.

## 4. Novelty threat map

| Claim co the viet | Muc do nguy hiem | Ly do |
|---|---:|---|
| "Phuong phap dau tien cho heterogeneous-rank FedLoRA" | Cao | HetLoRA, FLoRA, FSLoRA, PreLort, GLoRA da bao phu. |
| "Phuong phap dau tien giai gauge ambiguity trong FedLoRA" | Cao | GLoRA da noi truc dien. |
| "Phuong phap dau tien align LoRA subspace/factors" | Cao | FedRot-LoRA, FLoRG va cac paper alignment da co. |
| "Phuong phap dau tien asynchronous federated LLM fine-tuning" | Cao | AlignFed da dung async FFT cho LLM. |
| "Phuong phap data-free low-rank correction cho stale LoRA innovation dua tren intrinsic current temporal subspace" | Trung binh/thap hon | Chua thay exact match trong review hien tai, nhung can search lai truoc thesis defense. |

## 5. Working novelty statement an toan

> Theo literature review hien tai, cac huong gan day da xu ly rieng le gauge-aware heterogeneous-rank FedLoRA aggregation va asynchronous cross-version federated LLM alignment. VAST-LoRA tap trung vao giao diem hep hon: stale client innovations trong asynchronous FedLoRA khi client co rank khac nhau. Phuong phap de xuat mot correction rule data-free, thao tac tren intrinsic low-rank update geometry, giu phan innovation con tuong thich voi current temporal reference subspace va chi attenuate stale residual.

## 6. Baseline bat buoc

Minimum viable baselines:

1. Sync FedAvg-LoRA.
2. Naive Async-LoRA.
3. Freshness-only Async-LoRA:

$$
D_i^{\text{fresh}}=\mu_iD_i,
\qquad
\mu_i=e^{-\lambda\tau_i}.
$$

4. Buffered Async-LoRA / FedBuff-style.
5. Heterogeneous-rank baseline theo HetLoRA/FLoRA-inspired implementation.
6. GLoRA-like gauge-aware aggregation.
7. GLoRA-like + freshness-only.

Desirable neu con thoi gian:

8. AlignFed-compatible comparison neu co calibration set.
9. FedSteer-inspired projection baseline tren vectorized LoRA update.
10. FedRot-LoRA/FLoRG-inspired alignment baseline neu co code.

## 7. Ket luan Week 1

Huong VAST-LoRA van co kha nang thanh thesis tot neu giu scope that hep:

- update phai la local innovation, khong phai whole adapter;
- moi phep so sanh geometry phai gauge-invariant;
- reference subspace nen den tu recent accepted innovations, khong phai current adapter position;
- correction nen la selective residual attenuation, khong phai whole-update decay;
- Week 3-4 phai co kill-test de chung minh $\rho_i^t$ co signal ngoai $\tau_i$.

